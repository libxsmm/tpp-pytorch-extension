#pragma once

#include <ATen/record_function.h>
#include <torch/csrc/distributed/c10d/comm.hpp>
#include <torch/extension.h>

void shm_allreduce(
    at::Tensor t_in,
    c10::intrusive_ptr<c10d::ProcessGroup> process_group);
