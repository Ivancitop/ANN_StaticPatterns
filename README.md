# StaticPatterns — detección de patrones estáticos + PI de velocidad en S32K312

Tres potenciómetros forman un patrón; una red 9-8-6-4 entrenada **en el micro** lo clasifica y
fija la referencia de un PI de velocidad para un motor DC con encoder.

| Clase | pot1 | pot2 | pot3 | Acción |
|---|---|---|---|---|
| 1 CW | H | H | L | +120 rpm |
| 2 CCW | L | H | H | −120 rpm |
| 3 Paro | H | L | H | PI regulando a 0 rpm (freno activo) |
| 0 Default | cualquier otra combinación, niveles intermedios, potes en movimiento o p<0.70 | | | PWM apagado, integrador a cero |

## Estructura

```
firmware/     main.c (StaticPatterns integrado), preproc.c/.h, mlp.c/.h, pi_ctrl.c/.h,
              pesos_iniciales.h (lo genera run_pc.py), pesos_entrenados.h (lo genera run_mcu.py)
python/       mlp_core.py      modelo de referencia, dataset, preprocesado (gemelo del C)
              run_pc.py        entrenamiento PC float64/float32, reducción, int8, pesos iniciales
              run_mcu.py       entrenamiento en el MCU por UART (o --host / --simular)
              capturar.py      dataset REAL anotado con los potes (comando C)
              evaluar.py       PC vs MCU sobre el dataset real (matrices de confusión)
              simulacion_lazo.py  lazo completo con modelo del motor (diseño del PI, latencia)
              telemetria.py    identificación, escalón, demo en RUN y análisis de métricas
              graficas.py      figuras del informe
tests/        host_mcu.c       gemelo del firmware para PC (mismo C, mismo protocolo)
              test_equivalencia.py  C vs Python: preprocesado y red
informe/      informe.tex + figuras/
```

## 1. Cambios en el proyecto de S32DS (StaticPatterns.mex)

El .mex actual **no tiene UART**. Agrega en S32 Configuration Tools:

1. **Pins**: LPUART6_TX y LPUART6_RX. En la S32K312EVB-Q172 el puente OpenSDA/USB-serie va a
   PTA15 (TX) y PTA16 (RX) — confírmalo con el esquemático de tu tarjeta.
2. **Peripherals → Lpuart_Uart**: instancia LPUART6, 115200 8N1, callback `Uart_Callback`,
   nombre de configuración que genere `Lpuart_Uart_Ip_xHwConfigPB_6` (igual que en el
   proyecto BackPropagation, puedes copiar el componente de ahí).
3. **IntCtrl_Ip → IntCtrlConfig_0**: agrega `LPUART6_IRQn`, handler `LPUART_UART_IP_6_IRQHandler`,
   prioridad **2** (encoder 0, ADC 1, UART 2: una trama nunca debe retrasar un flanco del encoder).
4. Regenera código, copia `firmware/*.c *.h` a `src/` y enlaza con `-lm`.

No hace falta `-u _scanf_float`: el firmware ya no usa `sscanf`; los floats viajan en hex.

Ajusta en `main.c` antes de compilar: `ENC_CPR` (cuentas por vuelta en x4 del eje que mides),
`ENC_SIGN` (si CW da cuentas negativas) y, tras identificar el motor, `PI_KP_DEFAULT/PI_KI_DEFAULT`.

## 2. Flujo de trabajo (una tarea del plan por paso)

```bash
cd python
python run_pc.py                       # 1) dataset sintético + referencia PC + pesos_iniciales.h
cd ../tests && gcc -O2 -std=c99 -I../firmware host_mcu.c ../firmware/preproc.c \
      ../firmware/mlp.c ../firmware/pi_ctrl.c -lm -o host_mcu
python test_equivalencia.py            # 2) el C del firmware == Python (antes de flashear)
cd ../python
python run_mcu.py --host && mv resultados_mcu.json resultados_host.json   # gemelo C
python simulacion_lazo.py              # 3) diseño del PI y latencia esperada
# --- con la tarjeta ---
python run_mcu.py --puerto COM6        # 4) entrena en el S32K312, genera pesos_entrenados.h
python capturar.py --puerto COM6       # 5) dataset real anotado (el MCU conserva la red)
python evaluar.py --puerto COM6        # 6) PC vs MCU sobre datos reales
python telemetria.py identificar --duty 500   # 7) K, tau y ganancias sugeridas
python telemetria.py ganancias --kp ... --ki ...
python telemetria.py escalon --rpm 120 --seg 2
python telemetria.py demo --seg 60     # 8) latencia detección→acción, carga CPU, ciclos
python graficas.py --salida ../informe/figuras
```

**Binario autónomo para la demo**: con `pesos_entrenados.h` en `src/`, compila con
`-DUSE_PRETRAINED`; arranca directo en modo RUN sin PC.

**Si reseteas el micro** después de entrenar, la red vuelve a los pesos iniciales (viven en RAM).
Captura y evalúa en la misma sesión, o usa el binario con `USE_PRETRAINED`.

## 3. Protocolo UART (115200 8N1)

| Comando | Respuesta |
|---|---|
| `T,<9 hex>,<c>` entrena / `V,...` evalúa | `cls,p0,p1,p2,p3,loss,cicF,cicB` (p y loss en hex) |
| `R` recarga pesos iniciales | `OK` |
| `W` vuelca los 162 parámetros | 21 líneas `W,idx,hex...` y `END` |
| `C` captura | `raw1,raw2,raw3,<9 hex>` o `N` si la ventana aún no se llena |
| `I` modo RUN · `F,<rpm>` PI con ref. fija · `O,<‰>` lazo abierto | `OK` + telemetría cada 10 ms |
| `X` a IDLE · `P,<kp·1e6>,<ki·1e6>` · `S,<rpm>` | `OK` |

Telemetría: `D,t_ms,clsRaw,cls,ref,rpm·10,u·1000,raw1,raw2,raw3,pmax·1000,carga·1000,cicF`

## 4. Mediciones que solo salen del laboratorio

- **Memoria**: `arm-none-eabi-size` del .elf (Debug y Release) → tabla del informe. Toma el
  tamaño de `int_pflash`/`int_sram` del linker script del proyecto.
- **CPU**: la telemetría trae la carga (ciclos ocupados / ciclos del periodo, ventana de 1 s) y
  los ciclos del forward (DWT). `telemetria.py demo` los resume.
- **Energía aproximada**: medidor USB o multímetro en serie con la alimentación de la tarjeta y,
  por separado, con la fuente del motor. Registra corriente en IDLE, en RUN con motor parado
  (clase Paro) y en RUN a 120 rpm. P = V·I; la diferencia IDLE→RUN sin motor es lo que cuesta
  la inferencia + control. Con la carga de CPU medida puedes estimar energía por inferencia.
- **Video de la demo**: potes → clase activa (telemetría en pantalla) → motor.

