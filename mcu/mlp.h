/*==================================================================================================
* Project : StaticPatterns
* Platform : S32K3XX
* Modulo   : mlp  (perceptron multicapa 9-8-6-4 con retropropagacion)
*
*   entrada (9 caracteristicas)
*     -> capa oculta 1: 8 neuronas, sigmoide parametrica por neurona
*     -> capa oculta 2: 6 neuronas, sigmoide parametrica por neurona
*     -> salida: 4 neuronas lineales + softmax (probabilidad por clase)
*
*   Perdida: entropia cruzada  L = -ln(p_y)
*   Entrenamiento: SGD por muestra (un ejemplo a la vez)
*
* Evolucion directa de la red 1-2-3-1 del proyecto BackPropagation: mismas
* sigmoides a/(1+b e^{-cz})+d, misma derivada sin expf() (desde la activacion),
* mismo esquema Forward / Backward / TrainStep. Cambian la topologia y la capa
* de salida (softmax en lugar de sigmoide + MSE, porque ahora hay 4 clases).
*
* Gemelo de python/mlp_core.py (clase MLP). Codigo C puro para pruebas en PC.
==================================================================================================*/
#ifndef MLP_H
#define MLP_H

#include <stdint.h>

#define MLP_N_IN   9U
#define MLP_N_H1   8U
#define MLP_N_H2   6U
#define MLP_N_OUT  4U

#define MLP_N_PARAMS ((MLP_N_IN*MLP_N_H1 + MLP_N_H1) + \
                      (MLP_N_H1*MLP_N_H2 + MLP_N_H2) + \
                      (MLP_N_H2*MLP_N_OUT + MLP_N_OUT))      /* = 162 */

typedef struct {
    float a;
    float b;
    float c;
    float d;
} Zigmoid;

/* Parametros entrenables, en el mismo orden que el volcado 'W':
 * W1, b1, W2, b2, W3, b3. W es fila por neurona: W[j*n_in + i].          */
typedef struct {
    float W1[MLP_N_H1 * MLP_N_IN];
    float b1[MLP_N_H1];
    float W2[MLP_N_H2 * MLP_N_H1];
    float b2[MLP_N_H2];
    float W3[MLP_N_OUT * MLP_N_H2];
    float b3[MLP_N_OUT];
} MlpParams;

/* Activaciones que el backward necesita */
typedef struct {
    float a1[MLP_N_H1];
    float a2[MLP_N_H2];
    float p[MLP_N_OUT];
} MlpCache;

void    Mlp_Load(MlpParams *n,
                 const float *W1, const float *b1,
                 const float *W2, const float *b2,
                 const float *W3, const float *b3);
void    Mlp_Forward(const MlpParams *n, const float *x, MlpCache *c);
void    Mlp_Backward(MlpParams *n, const float *x, const MlpCache *c,
                     uint8_t label, float lr);
float   Mlp_Loss(const MlpCache *c, uint8_t label);
uint8_t Mlp_Argmax(const MlpCache *c);
/* Acceso plano a los 162 parametros (para volcado por UART) */
float   Mlp_GetParam(const MlpParams *n, uint16_t idx);

#endif
