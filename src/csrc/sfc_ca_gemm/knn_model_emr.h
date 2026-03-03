/******************************************************************************
 * Copyright (c) Intel Corporation - All rights reserved.                      *
 * This file is part of the LIBXSMM library.                                   *
 *                                                                             *
 * For information on the license, see the LICENSE file.                       *
 * Further information: https://github.com/libxsmm/libxsmm/                    *
 * SPDX-License-Identifier: BSD-3-Clause                                       *
 ******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
/*
 * k-NN Model for GEMM Configuration Prediction (EMR Platform)
 * Training samples: 1573
 * k value: 1
 * Distance metric: euclidean
 *
 * Usage:
 *   #include "knn_model_emr.h"
 *
 *   int main() {
 *       int M = 4096, N = 4096, K = 4096;
 *       int kbf, K_layers;
 *       predict_config_knn_emr(M, N, K, &kbf, &K_layers);
 *       printf("Predicted: kbf=%d, K_layers=%d\n", kbf, K_layers);
 *   }
 *
 */

#ifndef KNN_MODEL_EMR_H
#define KNN_MODEL_EMR_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Predict optimal kbf and K_layers configuration for given GEMM dimensions.
 *
 * Parameters:
 *   M, N, K    - GEMM dimensions
 *   out_kbf    - Output: predicted kbf value (1-8)
 *   out_K_layers - Output: predicted K_layers value (1-8)
 *
 * Example:
 *   int kbf, K_layers;
 *   predict_config_knn_emr(4096, 4096, 4096, &kbf, &K_layers);
 */
void predict_config_knn_emr(int M, int N, int K, int* out_kbf, int* out_K_layers);

/* Get model information */
int get_knn_k_value_emr(void);           /* Returns k (number of neighbors) */
int get_knn_training_samples_emr(void);  /* Returns number of training samples */

#ifdef __cplusplus
}
#endif

#endif /* KNN_MODEL_EMR_H */
