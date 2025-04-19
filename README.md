# Intel® Tensor Processing Primitives (TPP) Extension for PyTorch

© Intel Corporation

[![BSD 3-Clause License](https://img.shields.io/badge/license-BSD3-blue.svg "BSD 3-Clause License")](LICENSE.md)

The Intel® Tensor Processing Primitives (TPP) extension brings highly optimized kernels to PyTorch, delivering accelerated deep learning performance on Intel architectures.

---

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
## Examples
### BERT
- [BERT SQuAD Fine-tuning](examples/bert/squad/README.txt)
- [BERT MLPerf pre-training](examples/bert/pretrain_mlperf/README.txt)

### [GNN](examples/gnn/README.md)
- [GraphSage](examples/gnn/graphsage/README.md)
- [Graph Attention Network (GAT)](examples/gnn/gat/README.md)


