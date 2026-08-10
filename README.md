# Intel® Tensor Processing Primitives (TPP) Extension for PyTorch

© Intel Corporation

[![BSD 3-Clause License](https://img.shields.io/badge/license-BSD3-blue.svg "BSD 3-Clause License")](LICENSE.md)

The **Intel® Tensor Processing Primitives (TPP)** extension for PyTorch brings highly optimized deep learning kernels to PyTorch, delivering **accelerated performance on Intel architectures**. It is designed to efficiently execute compute-intensive operations using Intel® AVX-512 and other architectural features through JIT-compiled kernels.

## What is TPP?

[**TPP (Tensor Processing Primitives)**](https://libxsmm.readthedocs.io/en/latest/libxsmm_tpp/) is a collection of low-level building blocks designed for performance-critical deep learning workloads. It is part of the [LIBXSMM](https://github.com/libxsmm/libxsmm) ecosystem, which focuses on:

- Just-In-Time (JIT) code generation for tensor operations.
- Optimized support for small and medium-sized GEMMs (General Matrix-Matrix Multiplications), especially BRGEMMs (Batch Reduce GEMMs).
- Cache-aware loop transformations and blocking strategies.
- Vectorization and multi-threading support.
  
  TPP supports multiple operation types using a unified JIT dispatch mechanism: **Unary Operations** (e.g., ReLU, copy, negation), **Binary Operations** (e.g., add, multiply), **Ternary Operations** (e.g., fused multiply-add), and **GEMM** and **BRGEMM** Kernels.
---

## About This Extension

This PyTorch extension integrates TPP kernels to accelerate various deep learning operators. Notably:

- It does **not** use `torch.compile` or other PyTorch dynamic compiler paths.
- It allows **direct invocation** of optimized C++/x86 kernels in PyTorch workflows.
- Ideal for research and experimentation on performance-aware PyTorch models.
  
## Prerequisites

- **Operating System:** Linux-based system (e.g., Ubuntu)
- **Compiler:** GCC 8.3.0 or higher
- **Environment:** Anaconda/Miniconda (recommended for Python and package management)
- **PyTorch:** 1.4.0 or higher
- **Python:**  3.6 or higher

---

## Installation Guide

### 1. Set Up Conda Environment

Use the provided script to create and configure a Conda environment.

```bash
# Optionally specify the conda installation path
bash utils/setup_conda.sh [-p <conda_install_path>]
```

This generates an `env.sh` script for activating the environment.

## 2. Install the TPP Extension

Activate the environment and install the extension:

```bash
# Activate Conda environment
source env.sh

# Initialize Git submodules
git submodule update --init

# Install the extension
python setup.py install
```

setup.py install is deprecated. For modern packaging, consider using:

```bash
pip install .
```

## Multi-Node Support (Optional)

To enable distributed training across multiple nodes, install the `torch_ccl` communication library:

```bash
bash utils/install_torch_ccl.sh
```
## Related Work & Repositories

### Core Libraries
- [libxsmm/libxsmm-dnn](https://github.com/libxsmm/libxsmm-dnn): LIBXSMM-based DNN kernels.
- [libxsmm/parlooper](https://github.com/libxsmm/parlooper): Loop parallelization library.

📄 [Tensor Processing Primitives(TPP)](https://arxiv.org/pdf/2104.05755)

📄 [Parlooper](https://arxiv.org/pdf/2304.12576)


### Kernel Testing
- [BRGEMM sample kernel](https://github.com/libxsmm/libxsmm/blob/main/samples/xgemm/gemm_kernel.c)
- [GEMM test script](https://github.com/libxsmm/libxsmm/blob/main/samples/xgemm/kernel_test/generate_gemm_test_scripts.sh#L8)

### MLIR Compiler Integration
- [tpp-mlir](https://github.com/libxsmm/tpp-mlir): MLIR-based TPP backend.
  
📄 [TPP-MLIR Compiler Paper](https://arxiv.org/abs/2404.15204v1)

### Triton CPU Upstreaming Efforts
- [triton-cpu (xsmm-main)](https://github.com/libxsmm/triton-cpu/tree/xsmm-main): Extending Triton with TPP support.
  
## Examples
### [Intel-alphafold](https://github.com/IntelLabs/open-omics-alphafold)
- [TPP optimization of AlphaFold2](examples/alphafold/README.md)
### BERT
- [BERT SQuAD Fine-tuning](examples/bert/squad/README.txt)
- [BERT MLPerf pre-training](examples/bert/pretrain_mlperf/README.txt)

### [GNN](examples/gnn/README.md)
- [GraphSage](examples/gnn/graphsage/README.md)
- [Graph Attention Network (GAT)](examples/gnn/gat/README.md)

### LLM
- [GPT-J](examples/llm/README.txt)


