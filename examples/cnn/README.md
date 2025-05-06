# Optimized 1D U-Net with TPP Convolution Layer

An optimized 1D convolutional layer (`Conv1dOpti`) is developed using LIBXSMM Tensor Processing Primitives (TPPs) to enable high-performance execution. The implementation is integrated into the `cnn/` directory of the `tpp_pytorch_extension` package, allowing seamless access upon compilation. The optimized layer is subsequently integrated into a full 1D U-Net architecture to serve as a complete and practical example.
---

## 🚀 Features

- ✅ Drop-in replacement of `nn.Conv1d` with `Conv1dOpti`

---

## 📁 File Overview

- `examples/cnn/unet_example.py`: U-Net implementation
- `tpp_pytorch_extension/cnn/Conv1dOpti_ext.py`: Optimized convolution pytorch extension
- `examples/cnn/Efficiency_test.py`: Original Conv1d vs Conv1dOpti comparison

---

## ⚙️ Environment Setup

Set up your environment using conda:

```bash
conda create -n tpp-unet python=3.10 -y (python: 3.6 or higher)
conda activate tpp-unet

# Install PyTorch (1.4.0 or higher)
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch

# Install required Python packages
pip install numpy==1.24
```

Install build tools and compile the TPP extension:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git

# Clone and initialize submodules
git submodule update --init 

# Build the TPP PyTorch extension
python setup.py install
```

---

## 🛠️ Development Guide

To use the optimized convolution in your own model, import:

```python
from tpp_pytorch_extension.cnn.Conv1dOpti_ext import Conv1dOpti, ReLU_bf16
```

### Example Usage:
```python
self.conv = Conv1dOpti(in_ch, out_ch, kernel_size=kernel_size, stride=1,
                       padding=0, dilation=dilation, bias=False)

x = torch.randn(8, 16, 1024)
out = conv(x)
```

---

## 📈 Run the U-Net


```bash
cd examples/cnn
python unet_example.py
```

---