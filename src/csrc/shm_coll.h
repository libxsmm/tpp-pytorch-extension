/******************************************************************************
 * Copyright (c) 2022 Intel Corporation - All rights reserved.                *
 *                                                                            *
 * For information on the license, see the LICENSE file.                      *
 * Further information: https://github.com/libxsmm/tpp-pytorch-extension/     *
 * SPDX-License-Identifier: BSD-3-Clause                                      *
 ******************************************************************************/
/* Author: Dhiraj Kalamkar (Intel Corp.)
 ******************************************************************************/

#ifndef _TPP_SHM_COLL_H_
#define _TPP_SHM_COLL_H_

#include <ATen/record_function.h>
#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <omp.h>
#include <sys/shm.h>
#include "xsmm_functors.h"

class SHMBuffer {
 public:
  static const int SHMID = 100;
  static const int BARID = 10000;
  static const int MAX_RANKS = 64;
  static const int DIRECT_THRESHOLD = 8*1024;
  c10::intrusive_ptr<c10d::ProcessGroup> pg;
  int rank;
  int size;
  size_t bufsz;
  int shmid[MAX_RANKS];
  int barid;
  void *shm_data[MAX_RANKS];
  void *scratch_data[MAX_RANKS];
  void *bar_data;
  volatile int *bar1;
  volatile int *bar2;

  SHMBuffer(size_t bufsz_, c10::intrusive_ptr<c10d::ProcessGroup> pg) : pg(pg) {
    bufsz = ((bufsz_ + 4095) / 4096) * 4096 * 2;
    rank = pg->getRank();
    size = pg->getSize();
    /* each process creates its own shared memory */
    // printf("SHM At %d r: %d  s: %d\n", __LINE__, rank, size);
    shmid[rank] = shmget(SHMID+rank, bufsz, IPC_CREAT | 0666);
    TPP_ASSERT(shmid[rank] >= 0, "shmid cannot create shared memory of size %lu", bufsz);
    // printf("SHM At %d\n", __LINE__);
    if (rank == 0) {
      barid = shmget(BARID, 4096, IPC_CREAT | 0666);
      TPP_ASSERT(barid >= 0, "barid cannot create shared memory");
    }
    // printf("SHM At %d\n", __LINE__);
    pg->barrier()->wait();
    // printf("SHM At %d\n", __LINE__);
    /* each process attaches itself with other processes */
    for(int i=0; i < size; i++) {
      if (i != rank)
        shmid[i] = shmget(SHMID+i, bufsz, 0666);
      TPP_ASSERT(shmid[i] >= 0, "shmid cannot get shared memory");
    }
    // printf("SHM At %d\n", __LINE__);
    if (rank != 0) {
      barid = shmget(BARID, 4096, IPC_CREAT | 0666);
      TPP_ASSERT(barid >= 0, "barid cannot create shared memory");
    }
    // printf("SHM At %d\n", __LINE__);
    for(int i=0; i < size; i++) {
       shm_data[i] = shmat(shmid[i], NULL, 0);
       TPP_ASSERT(shm_data[i], "shmat failed\n");
       scratch_data[i] = shm_data[i] + bufsz/2;
    }
    // printf("SHM At %d\n", __LINE__);
    bar_data = shmat(barid, NULL, 0);
    TPP_ASSERT(bar_data, "barat failed\n");
    bar1 = (int *)bar_data;
    *bar1 = 0;
    bar2 = bar1 + 128;
    *bar2 = 0;
    pg->barrier()->wait();
    printf("Shm buffer allocated with size %lu\n", bufsz);
  }

  ~SHMBuffer() {
    pg->barrier()->wait();
    for(int i=0; i < size; i++)
      shmdt(shm_data[i]);
    shmdt(bar_data);
    pg->barrier()->wait();
    shmctl(shmid[rank], IPC_RMID, NULL);
    shmctl(barid, IPC_RMID, NULL);
    pg->barrier()->wait();
    printf("SHM buffers deleted\n");
  }

  static SHMBuffer* getInst(size_t sz, c10::intrusive_ptr<c10d::ProcessGroup> pg) {
    static size_t buf_sz = 0;
    static SHMBuffer* inst = nullptr;

    if (buf_sz < sz) {
      if (inst != nullptr) delete inst;
      inst = new SHMBuffer(sz, pg);
      TPP_ASSERT(inst != nullptr, "Unable to create shm buffer\n");
      buf_sz = sz;
    }
    return inst;
  }

  void barrier() {
    static uint32_t count=0;
    // printf("SHM At %s:%d\n", __func__, __LINE__);
    if(count % 2) {
      __sync_fetch_and_add(bar1, 1); while((*bar1 % size) != 0);
    } else {
      __sync_fetch_and_add(bar2, 1); while((*bar2 % size) != 0);
    }
    count++;
    // printf("SHM At %s:%d\n", __func__, __LINE__);
  }

  at::Tensor getTensor(at::Tensor t) {
    size_t sz = t.numel() * t.element_size();
    TPP_ASSERT(sz <= bufsz, "Requested tensor size too big\n");
    auto ptr = shm_data[rank];
    auto t_new = torch::from_blob(ptr, t.sizes(), t.options());
    return t_new;
  }

  template<typename T>
  void allreduce_impl(at::Tensor t) {
    constexpr int BS = 1024;
    auto numel = t.numel();
    auto nBytes = numel * t.element_size();
    TPP_ASSERT(nBytes <= bufsz / 2, "Too large allreduce size");
    int nBlk = numel / BS;
    T*ptr = (T*)t.data_ptr();
    int rem = numel % BS;
    TPP_ASSERT(rem == 0, "Reminder not supported yet\n");
    bool need_copy = ptr != shm_data[rank];
    auto cpy_tpp = SCOPEIT(CpyTPP<T>(BS), EW_COPY);
    auto ucvt_tpp = SCOPEIT((ConvertTPP<T,float>(BS)), EW_COPY);
    auto dcvt_tpp = SCOPEIT((ConvertTPP<float,T>(BS)), EW_COPY);
    auto add_tpp = SCOPEIT((AddTPP<float,float,T>(BS)), EW_ADD);
    // printf("SHM At %s:%d\n", __func__, __LINE__);

    if (need_copy) {
    // printf("SHM At %s:%d\n", __func__, __LINE__);
      auto src = ptr;
      auto dst = (T*)shm_data[rank];
#pragma omp parallel for
      for (int i = 0; i < numel; i += BS) {
        cpy_tpp(src+i, dst+i);
      }
    }

    // printf("SHM At %s:%d\n", __func__, __LINE__);
    barrier();
    // printf("SHM At %s:%d\n", __func__, __LINE__);

    if (numel <= DIRECT_THRESHOLD) {
    // printf("SHM At %s:%d\n", __func__, __LINE__);
      auto dst = (T*)scratch_data[rank];
      auto lsrc = (T*)shm_data[rank];
#pragma omp parallel for
      for (int i = 0; i < numel; i += BS) {
        float ldst[BS];
        ucvt_tpp(lsrc+i, ldst);
        for (int r = 1; r < size; r++) {
          int r1 = (r + rank) % size;
          auto src = (T*)shm_data[r1];
          add_tpp(ldst, src+i, ldst);
        }
        dcvt_tpp(ldst, dst+i);
      }
      barrier();

      if (true) {
    // printf("SHM At %s:%d\n", __func__, __LINE__);
        auto src = (T*)scratch_data[rank];
        auto dst = ptr;
#pragma omp parallel for
        for (int i = 0; i < numel; i += BS) {
          cpy_tpp(src+i, dst+i);
        }
      }
    } else {
    // printf("SHM At %s:%d\n", __func__, __LINE__);
      int slice_start = (nBlk * rank / size) * BS;
      int slice_end = (nBlk * (rank+1) / size) * BS;
      int slice_size = slice_end - slice_start;

      auto dst = (T*)scratch_data[rank];
      auto lsrc = (T*)shm_data[rank];
#pragma omp parallel for
      for (int i = slice_start; i < slice_end; i += BS) {
        float ldst[BS];
        ucvt_tpp(lsrc+i, ldst);
        for (int r = 1; r < size; r++) {
          int r1 = (r + rank) % size;
          auto src = (T*)shm_data[r1];
          add_tpp(ldst, src+i, ldst);
        }
        dcvt_tpp(ldst, dst+i);
      }
    // printf("SHM At %s:%d\n", __func__, __LINE__);
      barrier();
    // printf("SHM At %s:%d\n", __func__, __LINE__);
      if (true) {
    // printf("SHM At %s:%d\n", __func__, __LINE__);
        for (int r = 0; r < size; r++) {
          int r1 = (r + rank) % size;
          int slice_start = (nBlk * r1 / size) * BS;
          int slice_end = (nBlk * (r1+1) / size) * BS;
          int slice_size = slice_end - slice_start;

          auto src = (T*)scratch_data[r1];
          auto dst = ptr;
#pragma omp parallel for
          for (int i = slice_start; i < slice_end; i += BS) {
            cpy_tpp(src+i, dst+i);
          }
        }
      }
    // printf("SHM At %s:%d\n", __func__, __LINE__);
    }
    // printf("SHM At %s:%d\n", __func__, __LINE__);
  }

  void allreduce(at::Tensor t) {
    auto dt = t.dtype();
    if (dt == at::kFloat) {
      allreduce_impl<float>(t);
    } else if (dt == at::kBFloat16) {
      allreduce_impl<bfloat16>(t);
    } else if (dt == at::kHalf) {
      allreduce_impl<half>(t);
    } else {
      TPP_ASSERT(0, "Unsupported dtype in allreduce\n");
    }
  }
};

#endif // _TPP_SHM_COLL_H_
