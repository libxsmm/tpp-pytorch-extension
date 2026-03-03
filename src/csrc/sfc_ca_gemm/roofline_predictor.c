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

#include "roofline_predictor.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

// Forward declaration for CPU detection from knn_model
#ifdef __cplusplus
extern "C" {
#endif
extern const char* detect_cpu_platform(void);
#ifdef __cplusplus
}
#endif

/* Helper function for max */
static inline double max_double(double a, double b) {
    return (a > b) ? a : b;
}

static inline long long max_ll(long long a, long long b) {
    return (a > b) ? a : b;
}

static inline long long min_ll(long long a, long long b) {
    return (a < b) ? a : b;
}

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
) {
    /* Input validation */
    if (M <= 0 || N <= 0 || K <= 0 || !kbf || !K_layers) {
        return -1;
    }

    double dtype_size_in_bytes = 2.0;  /* BF16 */
    double l2_in_mb = 2.0;
    long long best_kbf = 1;
    long long best_c_copies = 1;
    double best_roofline = 0.0;

    /* Iterate over c_copies (K_layers) */
    for (int ic = 1; ic <= c_copies_limit; ic++) {
        long long threads_per_layer = (threads + ic - 1) / ic;
        long long m_tasks = M / bm;
        long long n_tasks = N / bn;
        long long K_per_layer = (K + ic - 1) / ic;
        double gflops_per_brgemm = 2.0 * ((double)bm * (double)bn * (double)K_per_layer) / 1e9;
        long long calculate_reduction_time = (ic > 1) ? 1 : 0;

        /* Iterate over m_teams */
        for (long long m_teams_val = 1; m_teams_val <= threads_per_layer; m_teams_val++) {
            long long n_teams_val = threads_per_layer;
            while (m_teams_val * n_teams_val > threads_per_layer) {
                n_teams_val--;
            }

            long long m_tasks_per_team = (m_tasks + m_teams_val - 1) / m_teams_val;
            long long n_tasks_per_team = (n_tasks + n_teams_val - 1) / n_teams_val;
            long long brgemms_per_thread = m_tasks_per_team * n_tasks_per_team;
            long long brgemms_ab_mem = 1;
            long long brgemms_a_only_mem = m_tasks_per_team - 1;
            long long brgemms_b_only_mem = n_tasks_per_team - 1;
            long long brgemms_ab_l2 = brgemms_per_thread - brgemms_a_only_mem - brgemms_b_only_mem - brgemms_ab_mem;

            double a_slice_size_in_gb = (double)bm * (double)K_per_layer * dtype_size_in_bytes / (1024.0 * 1024.0 * 1024.0);
            double b_slice_size_in_gb = (double)bn * (double)K_per_layer * dtype_size_in_bytes / (1024.0 * 1024.0 * 1024.0);
            double c_slice_size_in_gb = (double)bm * (double)bn * dtype_size_in_bytes / (1024.0 * 1024.0 * 1024.0);
            double time_per_brgemm = gflops_per_brgemm / compute_per_core;

            /* Time calculations */
            double time_brgemms_ab_mem = brgemms_ab_mem * max_double(time_per_brgemm, (a_slice_size_in_gb + b_slice_size_in_gb) / bw_per_core);
            double time_brgemms_a_only_mem = brgemms_a_only_mem * max_double(time_per_brgemm, a_slice_size_in_gb / bw_per_core);
            double time_brgemms_b_only_mem = brgemms_b_only_mem * max_double(time_per_brgemm, b_slice_size_in_gb / bw_per_core);
            double time_brgemms_c_write = (double)brgemms_per_thread * c_slice_size_in_gb / c_bw_per_core;
            double time_brgemms_compute = (double)brgemms_ab_l2 * max_double(time_per_brgemm, gflops_per_brgemm / compute_per_core);
            double total_time_in_sec = time_brgemms_ab_mem + time_brgemms_a_only_mem + time_brgemms_b_only_mem + time_brgemms_c_write + time_brgemms_compute;

            /* Add reduction time if needed */
            if (calculate_reduction_time == 1) {
                long long reduction_tasks = m_tasks * n_tasks;
                long long reduction_tasks_per_thread = (reduction_tasks + threads - 1) / threads;
                double reduction_volume_in_gb = (double)reduction_tasks_per_thread * (double)ic * c_slice_size_in_gb;
                double reduction_time_in_sec = reduction_volume_in_gb / c_bw_per_core;
                total_time_in_sec += reduction_time_in_sec;
            }

            /* Calculate kbf to fit in L2 cache */
            long long kbf_val = 1;
            double blocked_a_slice = a_slice_size_in_gb;
            double blocked_b_slice = b_slice_size_in_gb;
            long long reuse_a = min_ll(4LL, m_tasks_per_team);
            long long reuse_b = min_ll(4LL, n_tasks_per_team);

            while ((reuse_a * blocked_a_slice + reuse_b * blocked_b_slice) > 0.5 * (l2_in_mb / 1024.0)) {
                kbf_val++;
                long long K_blocked = (K_per_layer + kbf_val - 1) / kbf_val;
                blocked_a_slice = (double)bm * (double)K_blocked * dtype_size_in_bytes / (1024.0 * 1024.0 * 1024.0);
                blocked_b_slice = (double)bn * (double)K_blocked * dtype_size_in_bytes / (1024.0 * 1024.0 * 1024.0);
            }

            /* Calculate roofline performance */
            double total_gflops = (2.0 * (double)M * (double)N * (double)K) / 1e9;
            double roofline = total_gflops / total_time_in_sec;

            if (best_roofline < roofline) {
                best_roofline = roofline;
                best_c_copies = ic;
                best_kbf = kbf_val;
            }
        }
    }

    /* Set output values */
    *kbf = (int)best_kbf;
    *K_layers = (int)best_c_copies;

    return 0;
}

int get_platform_roofline_params(
    long long* threads,
    long long* bm,
    long long* bn,
    double* bw_per_core,
    double* c_bw_per_core,
    double* compute_per_core,
    long long* c_copies_limit
) {
    /* Input validation */
    if (!threads || !bm || !bn || !bw_per_core || !c_bw_per_core || !compute_per_core || !c_copies_limit) {
        return -1;
    }
    
    /* Detect CPU platform */
    const char* platform = detect_cpu_platform();

    /* Print the name of the detected platform */
    printf("Detected CPU platform: %s\n", platform);
    
    /* Set parameters based on platform */
    if (strcmp(platform, "GNR") == 0) {
        /* GNR (Granite Rapids) parameters */
        *threads = 128;                    /* Typical GNR core count */
        *bm = 32;
        *bn = 32;
        *bw_per_core = 5.15625;                /* GB/s */
        *c_bw_per_core = 5.15625;              /* GB/s */
        *compute_per_core = 1304.389445;   /* GFLOPS */
        *c_copies_limit = 8;
    } else {
        /* EMR (Emerald Rapids) or default parameters */
        *threads = 64;
        *bm = 32;
        *bn = 32;
        *bw_per_core = 6.54;               /* GB/s */
        *c_bw_per_core = 6.54;             /* GB/s */
        *compute_per_core = 1517.505278;   /* GFLOPS */
        *c_copies_limit = 8;
    }
    
    return 0;
}

int predict_config_roofline_simple(
    long long M,
    long long N,
    long long K,
    int* kbf,
    int* K_layers
) {
    /* Get platform-specific parameters */
    long long threads, bm, bn, c_copies_limit;
    double bw_per_core, c_bw_per_core, compute_per_core;
    
    int result = get_platform_roofline_params(
        &threads, &bm, &bn,
        &bw_per_core, &c_bw_per_core, &compute_per_core,
        &c_copies_limit
    );
    
    if (result != 0) {
        return result;
    }

    return predict_config_roofline(
        M, N, K, kbf, K_layers,
        threads, bm, bn,
        bw_per_core, c_bw_per_core, compute_per_core,
        c_copies_limit
    );
}
