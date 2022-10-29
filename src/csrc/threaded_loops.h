/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#ifndef _THREADED_LOOPS_H_
#define _THREADED_LOOPS_H_

#include <stdio.h>
#include <array>
#include <cassert>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

typedef std::function<void()> init_func;
typedef std::function<void()> fini_func;
typedef std::function<void(int*)> loop_func;

class LoopSpecs {
 public:
  LoopSpecs(long end) : LoopSpecs(0L, end, 1L) {}
  LoopSpecs(long end, bool isParallel) : LoopSpecs(0L, end, 1L, isParallel) {}
  LoopSpecs(long start, long end) : LoopSpecs(start, end, 1L) {}
  LoopSpecs(long start, long end, bool isParallel)
      : LoopSpecs(start, end, 1L, isParallel) {}
  LoopSpecs(long start, long end, long step)
      : LoopSpecs(start, end, step, true) {}
  LoopSpecs(long start, long end, long step, bool isParallel)
      : start(start), end(end), step(step), isParallel(isParallel) {}
  long start;
  long end;
  long step;
  bool isParallel;
};

typedef void (*par_loop_kernel)(
    LoopSpecs* loopSpecs,
    std::function<void(int*)>,
    std::function<void()>,
    std::function<void()>);

extern std::unordered_map<std::string, par_loop_kernel> pre_defined_loops;

class LoopingScheme {
 public:
  LoopingScheme(std::string scheme) : scheme(scheme), loop_kernel(NULL) {
    auto search = pre_defined_loops.find(scheme);
    if (search != pre_defined_loops.end()) {
      loop_kernel = search->second;
    } else {
      throw std::runtime_error(
          "The loop scheme '" + scheme + "' is not pre-defined" +
          " and dynamic loop schemes not supported yet!");
    }
  }

  void call(
      LoopSpecs* loopSpecs,
      std::function<void(int*)> body_func,
      std::function<void()> init_func,
      std::function<void()> fini_func) {
    loop_kernel(loopSpecs, body_func, init_func, fini_func);
  }

  std::string scheme;
  par_loop_kernel loop_kernel;
};

inline LoopingScheme* getLoopingScheme(std::string scheme) {
  static std::unordered_map<std::string, LoopingScheme*> kernel_cache;

  LoopingScheme* kernel = NULL;
  auto search = kernel_cache.find(scheme);
  if (search != kernel_cache.end())
    kernel = search->second;
  if (kernel == NULL) {
    kernel = new LoopingScheme(scheme);
    kernel_cache[scheme] = kernel;
  }
  return kernel;
}

template <int N>
class ThreadedLoop {
 public:
  ThreadedLoop(const LoopSpecs (&bounds)[N], std::string scheme = "")
      : bounds(bounds), scheme(scheme) {
    if (scheme == "")
      scheme = getDefaultScheme();
    loopScheme = getLoopingScheme(scheme);
  }

  template <class T>
  void operator()(T func) {
    loopScheme->call(bounds, func, NULL, NULL);
  }
  template <class T, class Ti, class Tf>
  void operator()(T func, Ti init, Tf fini) {
    loopScheme->call(bounds, func, init, fini);
  }

  std::string getDefaultScheme() {
    std::string scheme;
    for (int i = 0; i < N; i++) {
      if (bounds[i].isParallel)
        scheme.append(std::to_string('A' + i));
      else
        scheme.append(std::to_string('a' + i));
    }
    return scheme;
  }

 private:
  LoopSpecs bounds[N];
  std::string scheme;
  LoopingScheme* loopScheme;
};

#endif // _THREADED_LOOPS_H_
