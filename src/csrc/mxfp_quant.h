//  MIT License
//  Copyright (c) Microsoft Corporation.
//
//   Permission is hereby granted, free of charge, to any person obtaining a
//   copy of this software and associated documentation files (the "Software"),
//   to deal in the Software without restriction, including without limitation
//   the rights to use, copy, modify, merge, publish, distribute, sublicense,
//   and/or sell copies of the Software, and to permit persons to whom the
//   Software is furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be included in
//   all copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//   DEALINGS IN THE SOFTWARE
//
// This is a modified version of the routines here:
// https://github.com/microsoft/microxcaling/blob/main/mx/cpp/funcs.cpp

#include <math.h>
#include <stdio.h>
#include <cassert>
#include <iostream>
#include <vector>

#define FLOAT32_EXP_BIAS 127
#define FLOAT32_EXP_MAX 255
#define FLOAT32_TRAILING_MBITS 23
#define FLOAT32_IMPLIED1 (1 << FLOAT32_TRAILING_MBITS)
#define FLOAT32_FULL_MBITS (FLOAT32_TRAILING_MBITS + 1)
#define FLOAT32_INF 0x7fe00000
#define FLOAT32_EXP_OFFSET 23
#define FLOAT32_SIGN_OFFSET 31
#define FLOAT32_EXP_MASK 0x7f800000
#define FLOAT32_MANTISSA_MASK 0x007fffff

#define FLOAT16_MIN_NORMAL_EXP -14
#define FLOAT16_MAX_EXP 15
#define FLOAT16_EXP_BIAS 15

//---------------------------------------------------------
// Helper types/structs
//---------------------------------------------------------
typedef union {
  unsigned int i;
  float f;
} u_float_int;

typedef enum _RoundingMode {
  rd_away = 0, // round nearest, ties to away
  rd_floor = 1, // floor
  rd_even = 2 // round nearest, ties to even
} RoundingMode;

typedef enum _mxfp_type {
  MXFP6_E3M2 = 0,
  MXFP6_E2M3 = 1,
  MXFP4_E2M1 = 2,
  MXFP4_E3M0 = 3,
  MXFP4_E1M2 = 4,
  MXFP4_E0M3 = 5,
  MXFP2_E0M2 = 6
} mxfp_type;

//---------------------------------------------------------
// Helper functions for quantization
//---------------------------------------------------------
int get_sign(const u_float_int input) {
  int sign = input.i >> FLOAT32_SIGN_OFFSET;
  return sign;
}

float get_fp_scale(const int exp) {
  u_float_int out;
  // Shift up to exp bits
  out.i = exp << FLOAT32_EXP_OFFSET;
  return out.f;
}

int get_biased_exponent(const u_float_int input) {
  // Mask only exponent bits
  int exp = input.i & FLOAT32_EXP_MASK;
  // Shift down to lowest bits
  exp = exp >> FLOAT32_EXP_OFFSET;
  return exp;
}

// get_unbiased_exponent supports denorms
int get_unbiased_exponent(const float input) {
  u_float_int u;
  u.f = input;
  int exp = get_biased_exponent(u);
  if (exp == 0) {
    // Denorm
    return 1 - FLOAT32_EXP_BIAS;
  } else {
    return exp - FLOAT32_EXP_BIAS;
  }
}

int get_biased_exponent(const float input) {
  u_float_int u;
  u.f = input;
  return get_biased_exponent(u);
}

int get_trailing_mantissa(const u_float_int input) {
  return input.i & FLOAT32_MANTISSA_MASK;
}

// Construct float from sign, biased exponent, and mantissa
float construct_float(int sign, int biased_exp, int trailing_mantissa) {
  u_float_int x;
  x.i = trailing_mantissa | (biased_exp << FLOAT32_EXP_OFFSET) |
      (sign << FLOAT32_SIGN_OFFSET);
  return x.f;
}

//-----------------------------------------------------------------------
// Bound the shared_exp based on ebits
//-----------------------------------------------------------------------
int clamp_shared_exp(int shared_exp, const int ebits) {
  // Set overflowing shared exps to NaN and
  // bound underflowing shared exps to -emax
  // Note that (for 8 bits) the min scale is -127, not -126
  int emax = ebits != 0 ? (1 << (ebits - 1)) - 1 : FLOAT32_EXP_MAX;
  int shared_ub = shared_exp - FLOAT32_EXP_BIAS;
  shared_exp = shared_ub > emax ? FLOAT32_EXP_MAX : shared_exp;
  shared_exp = shared_ub < -emax ? FLOAT32_EXP_BIAS - emax : shared_exp;
  return shared_exp;
}

//-----------------------------------------------------------------------
// Compute shared scale for MX
//-----------------------------------------------------------------------
float mx_get_shared_scale(
    int shared_exp,
    const int scale_bits,
    const float elem_max_norm) {
  // Offset shared exponent by elem_emax, preserve NaNs
  const int elem_emax = get_unbiased_exponent(elem_max_norm);
  shared_exp =
      (shared_exp != FLOAT32_EXP_MAX) ? shared_exp - elem_emax : shared_exp;

  // Clamp to scale_bits range
  shared_exp = clamp_shared_exp(shared_exp, scale_bits);

  // Scale mantissa is 1 on the MSB mantissa bit if
  // scale is subnormal or NaN, otherwise 0
  const int scale_mant = (shared_exp == 0 || shared_exp == FLOAT32_EXP_MAX)
      ? (FLOAT32_IMPLIED1 >> 1)
      : 0;

  // Construct scale
  return construct_float(0, shared_exp, scale_mant);
}

//---------------------------------------------------------
// Shift right and round a float32 mantissa
// Example of "allow_overflow". Say we are rounding 11111 to 4 bits
// If allow_overflow is False, it will floor the result to 1111
// If allow_overflow is True,  it will round up to 10000, overflowing 4 bits
//---------------------------------------------------------
void shift_right_round_mantissa(
    int& mantissa, // 23-bit float32 trailing mantissa
    const bool is_subnorm, // is the input a subnorm?
    const int mbits, // number to bits to round to
    const int exp_diff, // extra right shifts
    const RoundingMode rounding_mode,
    const bool allow_overflow = false) {
  // Implied 1
  mantissa = is_subnorm ? mantissa : mantissa + FLOAT32_IMPLIED1;
  int fp32_sig_bits = is_subnorm ? 23 : 24;

  // RNE logic
  bool tie = false;
  bool even = false;
  if (rounding_mode == rd_even) {
    // tbits is the no. of bits that will be removed
    int tbits = exp_diff + (fp32_sig_bits - mbits);
    // 1 at all truncation locations except the first truncation location
    int mask = (1 << (tbits - 1)) - 1;
    // We have a tie only if all the truncation bits except the first
    // one are zero. If the first truncation bit is 1, we have an
    // actual tie. For rounding, we don't care if the first bits
    // is 1 or 0. If it is 0, no rounding happens.
    tie = !(mantissa & mask);
    mask = (1 << tbits); // 1 at the first non-truncated location
    even =
        !(mantissa &
          mask); // True if the last bit before truncation location is 0
  }

  // Adjust for shared exponent
  mantissa = mantissa >> exp_diff;
  // Shift down to target bit width + 1
  mantissa = mantissa >> (fp32_sig_bits - mbits - 1);
  // Rounding using floor(x+1), with overflow check
  if ((rounding_mode == rd_away || rounding_mode == rd_even) &&
      (allow_overflow || mantissa != ((1 << (mbits + 1)) - 1))) {
    if (!(tie && even))
      mantissa = mantissa + 1;
  }
  // Shift last bit away
  mantissa = mantissa >> 1;
}

//---------------------------------------------------------
// Shift back up to restore a float32 mantissa
// Use in pair with shift_right_round_mantissa to check for
// overflows caused by rounding
//---------------------------------------------------------
bool shift_left_mantissa(
    int& mantissa, // From shift_right_round_mantissa
    const bool is_subnorm, // is the input a subnorm?
    const int mbits,
    const int exp_diff) {
  int fp32_sig_bits = is_subnorm ? 23 : 24;
  mantissa = mantissa << (fp32_sig_bits - mbits + exp_diff);
  // Handle overflow.
  // When a subnorm overflows (into a normal) we don't rshift
  const bool overflow = (mantissa >= (1 << fp32_sig_bits));
  mantissa = (overflow && !is_subnorm) ? mantissa >> 1 : mantissa;
  // Remove implied 1
  mantissa = mantissa & (FLOAT32_IMPLIED1 - 1);
  return overflow;
}

//---------------------------------------------------------
// Quantize a float32 to arbitrary exponent and mantissa
// (trailing mantissa + implied 1 + sign) bits.
// The 'bits' arg should include sign and implicit 1. For
// bfloat16 bits=9.
//---------------------------------------------------------
float quantize_elemwise(
    float input,
    int bits, // bits = mantissa bits + sign bit
    int exp_bits, // exp_bits == 0 indicates integer dtype
    float max_norm,
    const RoundingMode rounding_mode = rd_away,
    bool saturate_normals = false,
    bool allow_denorm = true) {
  u_float_int input_;
  input_.f = input;
  int biased_exp = get_biased_exponent(input_);
  int sign = get_sign(input_);
  int tmant = get_trailing_mantissa(input_);

  // Mantissa bits to quantize to (remove sign)
  const int mbits = bits - 1;
  const bool is_int = exp_bits == 0;

  // Integers can be treated has having exp bias of 1
  const int new_bias = is_int ? 1 : (1 << (exp_bits - 1)) - 1;
  const int new_biased_exp = biased_exp - FLOAT32_EXP_BIAS + new_bias;

  // Skip denorms
  if ((!is_int) && (!allow_denorm) && (new_biased_exp < 1)) {
    return 0.0;
  }

  // Use exp_diff to truncate additional bits for subnorms
  // mbits includes implicit 1, so when new_biased_exp==0
  // we want exp_diff = 1 to truncate away 1 bit
  int exp_diff = (new_biased_exp <= 0) ? 1 - new_biased_exp : 0;
  exp_diff = (exp_diff > FLOAT32_FULL_MBITS) ? FLOAT32_FULL_MBITS : exp_diff;

  // Shift down and round mantissa, allow overflow except for integers
  // This converts tmant into a full mantissa
  shift_right_round_mantissa(
      tmant, biased_exp == 0, mbits, exp_diff, rounding_mode, !is_int);

  if (tmant == 0) {
    return 0.0;
  }

  // Shift back up to restore mantissa
  // This converts back to a trailing mantissa
  const bool overflow =
      shift_left_mantissa(tmant, biased_exp == 0, mbits, exp_diff);
  biased_exp = overflow ? biased_exp + 1 : biased_exp;

  // Reconstruct float number
  float output = construct_float(sign, biased_exp, tmant);

  // Return Inf if rounded value is out of bounds,
  // unless target format is integer or saturate_normals==True
  if (abs(output) > max_norm) {
    if (is_int || saturate_normals)
      output = sign ? -max_norm : max_norm;
    else
      output = construct_float(sign, 0xFF, 0);
  }
  return output;
}

void quantize_mx_func_cpp(
    float* A, // Input tensor
    float* B, // Output tensor
    int axis_size,
    int pre_axis_size,
    int post_axis_size,
    int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    float* max_values, // Max values in each block
    const bool flush_fp32_subnorms,
    const int rmode = 0) {
  // Explicitly cast int to enum
  RoundingMode rounding_mode = static_cast<RoundingMode>(rmode);

  // Get data pointers
  auto max_value_data = max_values;
  auto A_data = A;
  auto B_data = B;

  // Loop over dimension before shared axis
  for (int i = 0; i < pre_axis_size; i++) {
    // Loop over dimension after shared axis
    for (int j = 0; j < post_axis_size; j++) {
      // Get shared exponent
      const long m_i = i * post_axis_size + j;
      int shared_exp = (int)get_biased_exponent(max_value_data[m_i]);
      bool flush_tile = (shared_exp == 0 && flush_fp32_subnorms);

      // Compute the shared scale
      const float scale =
          mx_get_shared_scale(shared_exp, scale_bits, elem_max_norm);

      // Loop over axis
      for (int k = 0; k < axis_size; k++) {
        int A_i = i * post_axis_size * axis_size + k * post_axis_size + j;

        float scaled_in = (flush_tile) ? 0 : A_data[A_i] / scale;

        float scaled_out = quantize_elemwise(
            scaled_in,
            elem_mbits,
            elem_ebits,
            elem_max_norm,
            rounding_mode,
            true,
            true);

        B_data[A_i] = scaled_out * scale;
      }
    }
  }
  return;
}

void quantize_mx_scale_func_cpp(
    float* A, // Input tensor
    unsigned char* B, // Output tensor
    unsigned char* scf,
    int axis_size,
    int pre_axis_size,
    int post_axis_size,
    int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm,
    float* max_values, // Max values in each block
    const bool flush_fp32_subnorms,
    const int rmode = 0) {
  // Explicitly cast int to enum
  RoundingMode rounding_mode = static_cast<RoundingMode>(rmode);

  // Get data pointers
  auto max_value_data = max_values;
  auto A_data = A;

  // Loop over dimension before shared axis
  for (int i = 0; i < pre_axis_size; i++) {
    // Loop over dimension after shared axis
    for (int j = 0; j < post_axis_size; j++) {
      // Get shared exponent
      const long m_i = i * post_axis_size + j;
      int shared_exp = (int)get_biased_exponent(max_value_data[m_i]);
      bool flush_tile = (shared_exp == 0 && flush_fp32_subnorms);
      unsigned int scale_int;
      unsigned char scale_uchar;
      unsigned int* scale_ptr;

      // Compute the shared scale
      const float scale =
          mx_get_shared_scale(shared_exp, scale_bits, elem_max_norm);

      scale_ptr = (unsigned int*)&scale;
      scale_int = *scale_ptr;
      scale_int = ((scale_int << 1) >> 24);
      scale_uchar = (unsigned char)scale_int;
      scf[0] = scale_uchar;

      // Loop over axis
      for (int k = 0; k < axis_size; k++) {
        int A_i = i * post_axis_size * axis_size + k * post_axis_size + j;

        float scaled_in = (flush_tile) ? 0 : A_data[A_i] / scale;

        float scaled_out = quantize_elemwise(
            scaled_in,
            elem_mbits,
            elem_ebits,
            elem_max_norm,
            rounding_mode,
            true,
            true);

        if (scaled_out == 0.0) {
          B[A_i] = 0;
        } else if (scaled_out == 0.5) {
          B[A_i] = 1;
        } else if (scaled_out == 1.0) {
          B[A_i] = 2;
        } else if (scaled_out == 1.5) {
          B[A_i] = 3;
        } else if (scaled_out == 2.0) {
          B[A_i] = 4;
        } else if (scaled_out == 3.0) {
          B[A_i] = 5;
        } else if (scaled_out == 4.0) {
          B[A_i] = 6;
        } else if (scaled_out == 6.0) {
          B[A_i] = 7;
        } else if (scaled_out == -0.0) {
          B[A_i] = 8;
        } else if (scaled_out == -0.5) {
          B[A_i] = 9;
        } else if (scaled_out == -1.0) {
          B[A_i] = 10;
        } else if (scaled_out == -1.5) {
          B[A_i] = 11;
        } else if (scaled_out == -2.0) {
          B[A_i] = 12;
        } else if (scaled_out == -3.0) {
          B[A_i] = 13;
        } else if (scaled_out == -4.0) {
          B[A_i] = 14;
        } else if (scaled_out == -6.0) {
          B[A_i] = 15;
        }
      }
    }
  }
  return;
}

float find_max(float* array, int n) {
  float result = fabs(array[0]);
  for (int i = 1; i < n; i++) {
    if (fabs(array[i]) > result) {
      result = fabs(array[i]);
    }
  }
  return result;
}

void quantize_mx_rne(
    float* A, // Input tensor
    float* B, // Output tensor
    int n_elts,
    int scale_bits,
    int elem_ebits,
    int elem_mbits,
    float elem_max_norm) {
  float max_val = find_max(A, n_elts);
  quantize_mx_func_cpp(
      A,
      B,
      n_elts,
      1,
      1,
      scale_bits,
      elem_ebits,
      elem_mbits,
      elem_max_norm,
      &max_val,
      true,
      2);
}

void quantize_mxfp_rne(
    float* in, // Input tensor
    float* out, // Output tensor
    int n_elts,
    mxfp_type type) {
  // 8-bit scaling factors
  int scale_bits = 8;
  if (type == MXFP4_E2M1) {
    // FP4 specifics
    const float fp4_e2_m1_max_norm = 6.0;
    int fp4_e2_m1_elem_ebits = 2;
    int fp4_e2_m1_elem_mbits = 3;
    quantize_mx_rne(
        in,
        out,
        n_elts,
        scale_bits,
        fp4_e2_m1_elem_ebits,
        fp4_e2_m1_elem_mbits,
        fp4_e2_m1_max_norm);
  } else if (type == MXFP4_E3M0) {
    // FP4 specifics
    const float fp4_e3_m0_max_norm = 16.0;
    int fp4_e3_m0_elem_ebits = 3;
    int fp4_e3_m0_elem_mbits = 2;
    quantize_mx_rne(
        in,
        out,
        n_elts,
        scale_bits,
        fp4_e3_m0_elem_ebits,
        fp4_e3_m0_elem_mbits,
        fp4_e3_m0_max_norm);
  } else if (type == MXFP4_E1M2) {
    // FP4 specifics
    const float fp4_e1_m2_max_norm = 3.5;
    int fp4_e1_m2_elem_ebits = 1;
    int fp4_e1_m2_elem_mbits = 4;
    quantize_mx_rne(
        in,
        out,
        n_elts,
        scale_bits,
        fp4_e1_m2_elem_ebits,
        fp4_e1_m2_elem_mbits,
        fp4_e1_m2_max_norm);
  } else if (type == MXFP6_E2M3) {
    // FP6 e2m3 specifics
    const float fp6_e2_m3_max_norm = 7.5;
    int fp6_e2_m3_elem_ebits = 2;
    int fp6_e2_m3_elem_mbits = 5;
    quantize_mx_rne(
        in,
        out,
        n_elts,
        scale_bits,
        fp6_e2_m3_elem_ebits,
        fp6_e2_m3_elem_mbits,
        fp6_e2_m3_max_norm);
  } else if (type == MXFP6_E3M2) {
    // FP6 e3m2 specifics
    const float fp6_e3_m2_max_norm = 28.0;
    int fp6_e3_m2_elem_ebits = 3;
    int fp6_e3_m2_elem_mbits = 4;
    quantize_mx_rne(
        in,
        out,
        n_elts,
        scale_bits,
        fp6_e3_m2_elem_ebits,
        fp6_e3_m2_elem_mbits,
        fp6_e3_m2_max_norm);
  } else if (type == MXFP4_E0M3) {
    // FP4 specifics
    const float fp4_e0_m3_max_norm = 1.75;
    int fp4_e0_m3_elem_ebits = 0;
    int fp4_e0_m3_elem_mbits = 4;
    quantize_mx_rne(
        in,
        out,
        n_elts,
        scale_bits,
        fp4_e0_m3_elem_ebits,
        fp4_e0_m3_elem_mbits,
        fp4_e0_m3_max_norm);
  } else if (type == MXFP2_E0M2) {
    // FP4 specifics
    const float fp2_e0_m2_max_norm = 1.0;
    int fp2_e0_m2_elem_ebits = 0;
    int fp2_e0_m2_elem_mbits = 2;
    quantize_mx_rne(
        in,
        out,
        n_elts,
        scale_bits,
        fp2_e0_m2_elem_ebits,
        fp2_e0_m2_elem_mbits,
        fp2_e0_m2_max_norm);
  } else {
    printf("MXFP type is not supported!\n");
  }
}
