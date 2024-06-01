import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils import LayerNorm

# ref: https://github.com/DequanWang/weightnet.pytorch/blob/master/mmcls/models/backbones/weightnet.py#L22
def conv2d_sample_by_sample(x: torch.Tensor, weight: torch.Tensor, oup: int, inp: int, ksize: int, stride: int, padding: int, groups: int):
    batch_size = x.shape[0]
    if batch_size == 1:
        out = F.conv2d(x, weight=weight.view(oup, inp, ksize, ksize), stride=stride, padding=padding, groups=groups)
    else:
        out = F.conv2d(
            x.view(1, -1, x.shape[2], x.shape[3]), weight.view(batch_size * oup, inp, ksize, ksize), stride=stride, padding=padding, groups=groups * batch_size)
        out = out.view(batch_size, oup, out.shape[2], out.shape[3])
    return out

class FGRN(nn.Module):
    """ Filter-wise Global Response Normalization (FGRN)
    """
    def __init__(self, dim, kernel_size=3, **kwargs):
        super().__init__()

        self.dim = dim
        self.kernel_size = kernel_size
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x):
        batch, c, _, _ = x.size()
        x = x.reshape(batch, self.dim, self.kernel_size, self.kernel_size)
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-3, keepdim=True) + 1e-6)
        return (self.gamma * (x * Nx) + self.beta).reshape(batch, c, 1, 1)

class DPAC(nn.Module):
    """ Dual-Pooling Adaptive Convolution (DPAC)
    """
    def __init__(self, dim, kernel_size=3, stride=1, **kwargs):
        super().__init__()

        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_avg = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=True)
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.conv_max = nn.Conv2d(dim, dim, 1, 1, 0, groups=1, bias=True)
        self.act = nn.GELU()
        self.conv_mix = nn.Conv2d(dim, dim * kernel_size * kernel_size, 1, 1, 0, groups=dim, bias=False)
        self.norm = FGRN(dim=dim, kernel_size=kernel_size)

    def forward(self, x):
        x_w =  self.act(self.conv_avg(self.avg_pool(x)) + self.conv_max(self.max_pool(x)))
        x_w = self.conv_mix(x_w)
        x_w = self.norm(x_w)
        return conv2d_sample_by_sample(x, x_w, self.dim, 1, self.kernel_size, self.stride, self.padding, self.dim)

class MiddleBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, hidden_dim=1, **kwargs):
        super().__init__()

        self.wnet = DPAC(dim=dim, ksize=kernel_size, stride=1)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Conv2d(in_channels=dim, out_channels=hidden_dim * dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(in_channels=hidden_dim * dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def forward(self, x):
        input = x
        x = self.wnet(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + x
        return x

class MiddleNetwork(nn.Module):
    def __init__(self, embed_dims=[64, 128, 256, 512], kernel_size=3, blk_nums=3):
        super().__init__()

        self.block = nn.ModuleList([
            MiddleBlock(dim=embed_dims[3], kernel_size=kernel_size)
            for _ in range(blk_nums)])

    def forward(self, x):
        x = x[3]
        outs = []
        for blk in self.block:
            x = blk(x)
        outs.append(x)
        return outs
