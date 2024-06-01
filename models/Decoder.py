import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from .utils import LayerNorm

class DecoderBlock(nn.Module):
    def __init__(self, dim, drop_path=0., kernel_size=3):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class Decoder(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512], blk_nums=[1, 1, 1, 1], **kwargs):
        super().__init__()

        self.convd16x = nn.ConvTranspose2d(in_channels=channels[3], out_channels=channels[2], kernel_size=4, stride=2, padding=1)
        self.dense_4 = nn.Sequential(*[DecoderBlock(dim=channels[2]) for _ in range(blk_nums[0])])

        self.convd8x = nn.ConvTranspose2d(in_channels=channels[2], out_channels=channels[1], kernel_size=4, stride=2, padding=1)
        self.dense_3 = nn.Sequential(*[DecoderBlock(dim=channels[1]) for _ in range(blk_nums[1])])

        self.convd4x = nn.ConvTranspose2d(in_channels=channels[1], out_channels=channels[0], kernel_size=4, stride=2, padding=1)
        self.dense_2 = nn.Sequential(*[DecoderBlock(dim=channels[0]) for _ in range(blk_nums[2])])

        self.convd2x = nn.ConvTranspose2d(in_channels=channels[0], out_channels=16, kernel_size=4, stride=2, padding=1)
        self.dense_1 = nn.Sequential(*[DecoderBlock(dim=16) for _ in range(blk_nums[3])])

        self.convd1x = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride=2, padding=1)

        self.clean = nn.Conv2d(in_channels=8, out_channels=3, kernel_size=3, stride=1, padding=1)

        self.active = nn.Tanh()


    def forward(self,x1,x2):
        res16x = x2[0] + x1[3]
        res16x = self.convd16x(res16x)
        res8x = self.dense_4(res16x) + x1[2]
        res8x = self.convd8x(res8x)
        res4x = self.dense_3(res8x) + x1[1]
        res4x = self.convd4x(res4x)
        res2x = self.dense_2(res4x) + x1[0]
        res2x = self.convd2x(res2x)
        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)
        x = self.active(self.clean(x))

        return x
