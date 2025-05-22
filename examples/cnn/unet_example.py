import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the optimized Conv1d layer and optional BF16 ReLU
from tpp_pytorch_extension.cnn.Conv1dOpti_ext import Conv1dOpti, ReLU_bf16

# ------------------------ Config ------------------------ #
Batch_size  = 64
Input_width = 600
Channels    = 15
Filters     = 15
Kernel_size = 20
Dilation    = 8
enable_BF16 = False  # Set True to enable BF16 compute where allowed

def weight_init(m):
    if isinstance(m, (nn.Conv1d, Conv1dOpti)) and hasattr(m, 'weight'):
        nn.init.normal_(m.weight)

class ZeroSamePad1d(nn.Module):
    """Apply SAME zero padding to input."""
    def __init__(self, interval_size, kernel_size, stride, dilation):
        super().__init__()
        total = ZeroSamePad1d._get_total_same_padding(interval_size, kernel_size, stride, dilation)
        left = total // 2
        right = total - left
        self.pad = nn.ConstantPad1d((left, right), 0)

    @staticmethod
    def _get_total_same_padding(interval_size, kernel_size, stride, dilation):
        eff_kernel = (kernel_size - 1) * dilation + 1
        return (interval_size - 1) * stride + eff_kernel - interval_size

    def forward(self, x):
        return self.pad(x)

class DoubleConv1d(nn.Module):
    """Two Conv1d layers with BatchNorm and ReLU, SAME-padded."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, use_opt=False, bf16=False):
        super().__init__()
        self.use_opt = use_opt
        self.bf16 = bf16
        # first conv + padding
        self.pad1 = ZeroSamePad1d(Input_width, kernel_size, 1, dilation)
        if use_opt:
            self.conv1 = Conv1dOpti(in_ch, out_ch,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=0,
                                     dilation=dilation,
                                     bias=False,
                                     enable_BF16=bf16)
        else:
            self.conv1 = nn.Conv1d(in_ch, out_ch,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=0,
                                   dilation=dilation,
                                   bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        # second conv + padding
        self.pad2 = ZeroSamePad1d(Input_width, kernel_size, 1, dilation)
        if use_opt:
            self.conv2 = Conv1dOpti(out_ch, out_ch,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=0,
                                     dilation=dilation,
                                     bias=False,
                                     enable_BF16=bf16)
        else:
            self.conv2 = nn.Conv1d(out_ch, out_ch,
                                   kernel_size=kernel_size,
                                   stride=1,
                                   padding=0,
                                   dilation=dilation,
                                   bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x.float()).to(x.dtype)
        x = ReLU_bf16.apply(x) if (self.use_opt and self.bf16) else F.relu(x, inplace=True)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.bn2(x.float()).to(x.dtype)
        x = ReLU_bf16.apply(x) if (self.use_opt and self.bf16) else F.relu(x, inplace=True)
        return x

class Down1d(nn.Module):
    """Downscaling block with optional pooling."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, use_opt, bf16, do_pool):
        super().__init__()
        self.do_pool = do_pool
        self.pool    = nn.MaxPool1d(2)
        self.conv    = DoubleConv1d(in_ch, out_ch, kernel_size, dilation, use_opt, bf16)

    def forward(self, x):
        if self.do_pool:
            x = self.pool(x)
        return self.conv(x)

class Up1d(nn.Module):
    """Upscaling block with optional upsample."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, use_opt, bf16, do_pool, bilinear=True):
        super().__init__()
        self.do_pool = do_pool
        if do_pool:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True) if bilinear else \
                       nn.ConvTranspose1d(in_ch//2, in_ch//2, kernel_size=2, stride=2)
        else:
            self.up = nn.Identity()
        self.conv = DoubleConv1d(in_ch, out_ch, kernel_size, dilation, use_opt, bf16)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size(2) - x1.size(2)
        x1 = F.pad(x1, [diff//2, diff-diff//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv1d(nn.Module):
    """Final 1x1 convolution."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet1d(nn.Module):
    """1D U-Net without down-sampling for apple-to-apple timing comparison."""
    def __init__(self, n_channels, n_classes, base, kernel_size,
                 dilation, use_opt=False, bf16=False, do_pool=False, bilinear=True):
        super().__init__()
        self.inc   = DoubleConv1d(n_channels, base, kernel_size, dilation, use_opt, bf16)
        self.down1 = Down1d(base, base*2, kernel_size, dilation, use_opt, bf16, do_pool)
        self.down2 = Down1d(base*2, base*4, kernel_size, dilation, use_opt, bf16, do_pool)
        self.down3 = Down1d(base*4, base*8, kernel_size, dilation, use_opt, bf16, do_pool)
        factor    = 2 if bilinear else 1
        self.down4 = Down1d(base*8, base*16//factor, kernel_size, dilation, use_opt, bf16, do_pool)
        self.up1 = Up1d(base*16, base*8//factor, kernel_size, dilation, use_opt, bf16, do_pool, bilinear)
        self.up2 = Up1d(base*8, base*4//factor, kernel_size, dilation, use_opt, bf16, do_pool, bilinear)
        self.up3 = Up1d(base*4, base*2//factor, kernel_size, dilation, use_opt, bf16, do_pool, bilinear)
        self.up4 = Up1d(base*2, base,          kernel_size, dilation, use_opt, bf16, do_pool, bilinear)
        self.outc = OutConv1d(base, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x, x3)
        x  = self.up3(x, x2)
        x  = self.up4(x, x1)
        return self.outc(x)

if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(Batch_size, Channels, Input_width, requires_grad=True)

    net_std = UNet1d(Channels, Channels, Filters, Kernel_size, Dilation,
                      use_opt=False, bf16=enable_BF16, do_pool=False)
    net_opt = UNet1d(Channels, Channels, Filters, Kernel_size, Dilation,
                      use_opt=True, bf16=enable_BF16,  do_pool=False)

    net_opt.load_state_dict(net_std.state_dict())

    # Accuracy check
    net_std.zero_grad(); y1 = net_std(X); y1.sum().backward(); gw1 = next(net_std.parameters()).grad.clone()
    net_opt.zero_grad(); y2 = net_opt(X); y2.sum().backward(); gw2 = next(net_opt.parameters()).grad.clone()
    print("Weight grad match:", torch.allclose(gw1, gw2, atol=1e-5))

    # Timing loops
    forward1 = backward1 = 0.0
    forward2 = backward2 = 0.0
    N = 20
    for _ in range(N):
        start = time.time(); out1 = net_std(X); forward1 += time.time() - start
        start = time.time(); out1.sum().backward(); backward1 += time.time() - start; net_std.zero_grad()
    if enable_BF16: X = X.to(torch.bfloat16)
    for _ in range(N):
        start = time.time(); out2 = net_opt(X); forward2 += time.time() - start
        start = time.time(); out2.sum().backward(); backward2 += time.time() - start; net_opt.zero_grad()

    print(f"Forward time (Std): {forward1/N*1e3:.3f} ms | (Opt): {forward2/N*1e3:.3f} ms")
    print(f"Backward time(Std): {backward1/N*1e3:.3f} ms | (Opt): {backward2/N*1e3:.3f} ms")

