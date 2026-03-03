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
 * Automatically detects CPU platform and dispatches to appropriate model.
 */

#include <stdio.h>
#include <string.h>
#include "knn_model.h"
#include "knn_model_emr.h"
#include "knn_model_gnr.h"

#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif

/*
 * Detect CPU platform based on CPUID information
 * Returns: "EMR", "GNR", or "UNKNOWN"
 */
const char* detect_cpu_platform(void) {
    static char platform[16] = {0};
    static int detected = 0;
    
    // Return cached result if already detected
    if (detected) {
        return platform;
    }
    
    unsigned int eax, ebx, ecx, edx;
    char vendor[13] = {0};
    
    // Get vendor string
#ifdef _WIN32
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    memcpy(vendor, &cpuInfo[1], 4);
    memcpy(vendor + 4, &cpuInfo[3], 4);
    memcpy(vendor + 8, &cpuInfo[2], 4);
#else
    __cpuid(0, eax, ebx, ecx, edx);
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
#endif
    
    // Check if it's an Intel CPU
    if (strcmp(vendor, "GenuineIntel") != 0) {
        strcpy(platform, "UNKNOWN");
        detected = 1;
        return platform;
    }
    
    // Get CPU family, model, and stepping
#ifdef _WIN32
    __cpuid(cpuInfo, 1);
    eax = cpuInfo[0];
#else
    __cpuid(1, eax, ebx, ecx, edx);
#endif
    
    int family = ((eax >> 8) & 0x0F);
    int ext_family = (eax >> 20) & 0xFF;
    int model = ((eax >> 4) & 0x0F);
    int ext_model = ((eax >> 16) & 0x0F);
    
    int display_family = family;
    int display_model = model;
    
    if (family == 0x0F) {
        display_family = family + ext_family;
    }
    
    if (family == 0x06 || family == 0x0F) {
        display_model = (ext_model << 4) | model;
    }
    
    // Detect platform based on family and model
    // EMR (Emerald Rapids): Family 6, Model 0xCF (207)
    // GNR (Granite Rapids): Family 6, Model 0xAD (173)
    
    if (display_family == 6) {
        if (display_model == 0xCF || display_model == 207) {
            strcpy(platform, "EMR");
        } else if (display_model == 0xAD || display_model == 173) {
            strcpy(platform, "GNR");
        } else if (display_model >= 0x8F && display_model <= 0x9F) {
            // SPR (Sapphire Rapids) range - treat as EMR for compatibility
            strcpy(platform, "EMR");
        } else {
            // Default to EMR for unknown Intel CPUs
            strcpy(platform, "EMR");
        }
    } else {
        strcpy(platform, "UNKNOWN");
    }
    
    detected = 1;
    return platform;
}

/*
 * Predict configuration using platform-appropriate k-NN model
 */
void predict_config_knn(int M, int N, int K, int* out_kbf, int* out_K_layers) {
    const char* platform = detect_cpu_platform();
    /* Print the name of the detected platform */
    // printf("Detected CPU platform: %s\n", platform);
    
    if (strcmp(platform, "GNR") == 0) {
        predict_config_knn_gnr(M, N, K, out_kbf, out_K_layers);
    } else {
        // Default to EMR model for EMR, UNKNOWN, and other platforms
        predict_config_knn_emr(M, N, K, out_kbf, out_K_layers);
    }
}
