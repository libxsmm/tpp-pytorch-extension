/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#ifndef _TPP_FLOAT8_H_
#define _TPP_FLOAT8_H_

#ifdef PYTORCH_SUPPORTS_FLOAT8

namespace at {

constexpr auto kBFloat8 = kFloat8_e5m2;
constexpr auto kHFloat8 = kFloat8_e4m3fn;
}; // namespace at

#endif // PYTORCH_SUPPORTS_FLOAT8

#endif // _TPP_BFLOAT8_H_
