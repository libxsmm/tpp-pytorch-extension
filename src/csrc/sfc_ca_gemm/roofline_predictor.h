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

#ifndef ROOFLINE_PREDICTOR_H
#define ROOFLINE_PREDICTOR_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Predict optimal kbf and K_layers (c_copies) configuration using roofline model
 * 
 * @param M                  First GEMM dimension
 * @param N                  Second GEMM dimension
 * @param K                  Contraction dimension
 * @param kbf                Output: predicted kbf value
 * @param K_layers           Output: predicted K_layers (c_copies) value
 * @param threads            Number of threads (default: 64)
 * @param bm                 Block size M (default: 32)
 * @param bn                 Block size N (default: 32)
 * @param bw_per_core        Bandwidth per core in GB/s (default: 6.54)
 * @param c_bw_per_core      C bandwidth per core in GB/s (default: 6.54)
 * @param compute_per_core   Compute throughput per core in GFLOPS (default: 1517.505278)
 * @param c_copies_limit     Maximum K_layers to consider (default: 8)
 * 
 * @return 0 on success, -1 on error
 */
int predict_config_roofline(
    long long M, 
    long long N, 
    long long K,
    int* kbf,
    int* K_layers,
    long long threads,
    long long bm,
    long long bn,
    double bw_per_core,
    double c_bw_per_core,
    double compute_per_core,
    long long c_copies_limit
);

/**
 * Simplified roofline predictor with default hardware parameters
 * Uses default values optimized for the target platform
 */
int predict_config_roofline_simple(
    long long M,
    long long N,
    long long K,
    int* kbf,
    int* K_layers
);

/**
 * Get platform-specific hardware parameters for roofline model
 * Automatically detects CPU platform (EMR or GNR) and returns appropriate values
 * 
 * @param threads            Output: Number of threads
 * @param bm                 Output: Block size M
 * @param bn                 Output: Block size N
 * @param bw_per_core        Output: Bandwidth per core in GB/s
 * @param c_bw_per_core      Output: C bandwidth per core in GB/s
 * @param compute_per_core   Output: Compute throughput per core in GFLOPS
 * @param c_copies_limit     Output: Maximum K_layers to consider
 * 
 * @return 0 on success, -1 on error
 */
int get_platform_roofline_params(
    long long* threads,
    long long* bm,
    long long* bn,
    double* bw_per_core,
    double* c_bw_per_core,
    double* compute_per_core,
    long long* c_copies_limit
);

#ifdef __cplusplus
}
#endif

#endif /* ROOFLINE_PREDICTOR_H */
