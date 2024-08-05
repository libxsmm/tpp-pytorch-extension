Extension to enable distributed MPI backend for PyTorch installed from binaries
===============================================================================

`torch-mpi` module loads the MPI backend dynamically.

```python
>>> import torch
>>> torch.distributed.is_mpi_available()
False
>>> import torch_mpi
>>> torch.distributed.is_mpi_available()
True
```

# Pytorch API Align
We recommend Anaconda as Python package management system. The following is the corresponding branchs of torch-mpi and supported Pytorch.

   | ``torch`` | ``torch-mpi`` |  
   | :-----:| :---: |  
   |  ``master`` |  ``main``  |
   | [v2.3.0](https://github.com/pytorch/pytorch/tree/v2.3.0) |  v2.3  |
   | [v2.2.0](https://github.com/pytorch/pytorch/tree/v2.2.0) |  v2.2  |
   | [v2.1.0](https://github.com/pytorch/pytorch/tree/v2.1.0) |  v2.1  |
   | [v2.0.0](https://github.com/pytorch/pytorch/tree/v2.0.0) |  v2.0  |
   | [v1.13.0](https://github.com/pytorch/pytorch/tree/v1.13.0) |  v1.13  |
   | [v1.12.0](https://github.com/pytorch/pytorch/tree/v1.12.0) |  v1.12  |
   | [v1.11.0](https://github.com/pytorch/pytorch/tree/v1.11.0) |  v1.11  |
   | [v1.10.0](https://github.com/pytorch/pytorch/tree/v1.10.0) |  v1.10  |
   | [v1.9.0](https://github.com/pytorch/pytorch/tree/v1.9.0) |  v1.9  |
   | [v1.8.0](https://github.com/pytorch/pytorch/tree/v1.8.0) |  v1.8  |

# Requirements

MPI compiler `mpicxx` needs to be in PATH

Python 3.8 or later and a C++17 compiler

pytorch 1.8+

# Installation

To install `torch-mpi`:

1. clone the `torch-mpi`.

```bash
   git clone https://github.com/intel-sandbox/torch-mpi.git && cd torch-mpi 
```
2. Install torch-mpi

```bash
   python setup.py install
```

