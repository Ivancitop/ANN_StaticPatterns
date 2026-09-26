/*==================================================================================================
* Project : StaticPatterns
* Platform : S32K3XX
* Modulo   : preproc  (preprocesado de los 3 potenciometros)
*
* Pipeline por canal:
*   crudo ADC (12 b) -> media movil de FEAT_MA_LEN muestras (acumulador entero)
*                    -> ventana circular de FEAT_WIN muestras filtradas [0..1]
*                    -> media, desviacion estandar, maximo
*                    -> normalizacion fija (sin depender del dataset)
*
* Vector de salida (FEAT_N = 9):
*   [ 2*media1-1, 2*media2-1, 2*media3-1,
*     min(desv1/STD_REF, STD_CLIP), ... ,
*     2*max1-1,   2*max2-1,   2*max3-1 ]
*
* Gemelo de python/mlp_core.py (moving_average, features_from_filtered).
* Codigo C puro (sin RTD) para poder compilarlo en PC y probarlo.
==================================================================================================*/
#ifndef PREPROC_H
#define PREPROC_H

#include <stdint.h>

#define FEAT_N_CH     3U
#define FEAT_MA_LEN   8U       /* media movil: 8 x 5 ms = 40 ms          */
#define FEAT_WIN      32U      /* ventana:    32 x 5 ms = 160 ms         */
#define FEAT_N        (3U * FEAT_N_CH)
#define FEAT_ADC_FS    2700.0f
#define FEAT_STD_REF  0.02f
#define FEAT_STD_CLIP 5.0f

typedef struct {
    uint16_t maBuf[FEAT_N_CH][FEAT_MA_LEN];
    uint32_t maSum[FEAT_N_CH];
    uint8_t  maIdx;
    uint8_t  maCnt;
    float    win[FEAT_N_CH][FEAT_WIN];
    uint8_t  wIdx;
    uint8_t  wCnt;
} FeatState;

void    Feat_Init(FeatState *s);
/* Agrega una lectura cruda (3 canales). Devuelve el ultimo valor filtrado
 * de cada canal en filtOut (puede ser NULL).                              */
void    Feat_Push(FeatState *s, const uint16_t raw[FEAT_N_CH], float *filtOut);
uint8_t Feat_Ready(const FeatState *s);
void    Feat_Compute(const FeatState *s, float out[FEAT_N]);

#endif
