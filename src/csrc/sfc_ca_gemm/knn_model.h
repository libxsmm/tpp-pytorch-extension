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
 * Unified k-NN Model Interface for GEMM Configuration Prediction
 * 
 * Automatically detects CPU platform (EMR or GNR) and dispatches to
 * the appropriate platform-specific model.
 * 
 */

#ifndef KNN_MODEL_H
#define KNN_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Predict optimal kbf and K_layers configuration for given GEMM dimensions.
 * 
 * Automatically detects CPU platform and uses appropriate model.
 *
 * Parameters:
 *   M, N, K    - GEMM dimensions
 *   out_kbf    - Output: predicted kbf value (1-8)
 *   out_K_layers - Output: predicted K_layers value (1-8)
 *
 * Example:
 *   int kbf, K_layers;
 *   predict_config_knn(4096, 4096, 4096, &kbf, &K_layers);
 */
void predict_config_knn(int M, int N, int K, int* out_kbf, int* out_K_layers);

/*
 * Detect CPU platform
 * Returns: "EMR", "GNR", or "UNKNOWN"
 */
const char* detect_cpu_platform(void);

#ifdef __cplusplus
}
#endif

#endif /* KNN_MODEL_H */
