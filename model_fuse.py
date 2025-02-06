import torch
import torch.nn as nn
from torchvision import models as ML
import math
import copy
import numpy as np
import torch.nn.functional as F
# from KFBNet import KFB_VGG16
from torch.autograd import Variable
import torchvision.models as models
# from MSI_Model import MSINet
# from hrps_model import HpNet
# import hrnet
import pretrainedmodels
# from block import fusions
import argparse
from torchvision.models import resnet50, resnext50_32x4d, densenet121
import pretrainedmodels
from pretrainedmodels.models import *
# from models.segformer import SegFormer
import torch
import torch.nn as nn
import os
import torchvision
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
import time
from torch import nn, Tensor
from torch.nn import functional as F
from tabulate import tabulate
from typing import Tuple
import torch
from torch import nn, Tensor
from torch.nn import functional as F
# from resnet import ResNet
import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F
from backbones import MiT, ResNet, PVTv2
from backbones.layers import trunc_normal_
# from heads import MTSBAHead
# from heads import SFHead
# from Deformable_ConvNet import DeformConv2D
# from baseline_models import FCN, deeplabv3
from Unet import UNet
# from backbones.uniformer import UniFormer
from backbones.mobilenetv2 import MobileNetV2
from backbones.mit import MiT, Global_Local_block
from backbones.ResNet_backbone import ResNet
from backbones.mobileViT import mobilevit_xxs, MobileViTBlock

from torch.nn import BatchNorm2d
from ContextBlock2d import ContextBlock2d
# from CPAM import CPAM_Attention

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(True)
        )


class DetailBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.S1 = nn.Sequential(
            ConvModule(3, 64, 3, 2, 1),
            ConvModule(64, 64, 3, 1, 1)
        )
        self.S2 = nn.Sequential(
            ConvModule(64, 64, 3, 2, 1),
            # MobileViTBlock(32, 2, 64, 3, (2, 2), int(32*2)),
            ConvModule(64, 64, 3, 1, 1),
            ConvModule(64, 128, 3, 1, 1),
            ConvModule(128, 128, 3, 1, 1),
            # MobileViTBlock(64, 2, 128, 3, (2, 2), int(64*2))
            ConvModule(128, 128, 3, 1, 1)
        )

    def forward(self, x):
        return self.S2(self.S1(x))

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextEdgeFeatureEnhanceModule(nn.Module):
    def __init__(self, in_d):
        super(ContextEdgeFeatureEnhanceModule, self).__init__()
        self.in_d = in_d
        self.ContextBlock2d = ContextBlock2d(self.in_d, self.in_d, "att", ["channel_add"])
        self.edge_enhance = Sobel(self.in_d)

    def forward(self, x):

        print('GC before: ', x.shape)
        x_branchContextBlock2d = self.ContextBlock2d(x)
        print('GC block: ', x_branchContextBlock2d.shape)
        x_branchEdge = self.edge_enhance(x)
        print('sobel block: ', x_branchEdge.shape)
        print('GC and Sobel: ', torch.cat([x_branchContextBlock2d, x_branchEdge], 1).shape)

        return torch.cat([x_branchContextBlock2d, x_branchEdge], 1)
# MultiscaleContextFeatureEnhanceModule_Block

class ConvBlock(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )

class AttentionRefinmentModule(nn.Module):
    def __init__(self, c1, c2) -> None:
        super().__init__()
        self.conv_3x3 = ConvBlock(c1, c2, 3, 1, 1)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.Sigmoid()
        )

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.attention(fm)
        return fm * fm_se + x

class CEFEB(nn.Module):
    def __init__(self, channel):
        super(CEFEB, self).__init__()
        self.in_channels = channel
        # self.hide_channels = hide_dim

        self.ContextEdgeFeatureEnhanceModule = ContextEdgeFeatureEnhanceModule(self.in_channels)

        self.AttentionRefinmentModule = AttentionRefinmentModule(self.in_channels+2, self.in_channels+2)

    def forward(self, feature):

        feature_context = self.ContextEdgeFeatureEnhanceModule(feature)
        # feature_context = self.AggregationLayer(feature_context)
        print('attention before: ', feature_context.shape)
        feature_context = self.AttentionRefinmentModule(feature_context)
        print('attention after: ', feature_context.shape)
        # out = self.augmented_representation(feature_context)
        # out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return feature_context


class Sobel(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=inchannels, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)
        # self.filter_edge = nn.Conv2d(in_channels=inchannels, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        sobel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).astype(dtype='float32')
        sobel_x = np.reshape(sobel_x, (1, 3, 3))
        sobel_y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).astype(dtype='float32')
        sobel_y = np.reshape(sobel_y, (1, 3, 3))

        sobel_x = np.repeat(sobel_x, inchannels, axis=0).reshape((1, inchannels, 3, 3))
        sobel_y = np.repeat(sobel_y, inchannels, axis=0).reshape((1, inchannels, 3, 3))
        # sobel_x_2 = np.repeat(sobel_x_2, inchannels, axis=0).reshape((1, inchannels, 3, 3))
        # sobel_y_2 = np.repeat(sobel_y_2, inchannels, axis=0).reshape((1, inchannels, 3, 3))

        # print(sobel_x.shape, sobel_y.shape)
        sobel = torch.cat([torch.from_numpy(sobel_x), torch.from_numpy(sobel_y)], 0)
        # grad_edge = torch.cat([torch.from_numpy(sobel_x_2), torch.from_numpy(sobel_y_2)], 0)

        self.filter.weight = nn.Parameter(sobel, requires_grad=False)
        # self.filter_edge.weight = nn.Parameter(grad_edge, requires_grad=False)

    def forward(self, img):

        x = self.filter(img)

        return x # torch.cat([x, x_edge], 1)


class Head(nn.Module):
    def __init__(self, c1, n_classes, upscale_factor, is_aux=False) -> None:
        super().__init__()
        ch = 256 if is_aux else 64
        c2 = n_classes * upscale_factor * upscale_factor
        self.conv_3x3 = ConvModule(c1, ch, 3, 1, 1)
        self.conv_1x1 = nn.Conv2d(ch, c2, 1, 1, 0)
        self.upscale = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv_1x1(self.conv_3x3(x))
        return self.upscale(x)

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        # self.convblk = ConvModule(in_chan+2, in_chan+2 // 2, 1, 1, 0)
        self.Head = Head(in_chan+2, out_chan, 4)
        self.CEFEB = CEFEB(in_chan)


    def forward(self, fsp, fcp):

        fcat = torch.cat([fsp, fcp], dim=1)
        fcat = self.CEFEB(fcat) #  + fcat

        return self.Head(fcat)


# 传入nn.Module表示传入类别数，此时为：9
class LightSDNet(nn.Module):
    def __init__(self, n_class):
        super(LightSDNet, self).__init__()
        self.n_class = n_class
        # self.out_channels = 150
        self.dim = 128

        self.backbone = MobileNetV2()
        self.backbone.load_state_dict(torch.load('.\modules\\mobilenet_v2-b0353104.pth', map_location='cpu'), strict=False)
        # self.backbone.load_state_dict(torch.load('K:\\ShadowDetectionRS\\modules\\uniformer_small_in1k.pth', map_location='cpu')['model'], strict=False)

        self.decode_path = LightHead([24, 32, 96, 320], self.dim*2, self.dim)
        self.SpatialPath = DetailBranch()
        # self.SpatialPath  = mobilevit_xxs()
        # self.SpatialPath = Global_Local_block(self.dim // 2)
        self.FeatureFusionModule = FeatureFusionModule(self.dim + self.dim, self.n_class)
        self.conv_fc =  Head(self.dim, self.n_class, 4)
        self.conv_sp_fc = Head(self.dim, self.n_class, 4)

    def forward(self, h_rs):
        # print(h_rs.shape, x_floor.shape)
        out = self.backbone(h_rs)
        spatial_out = self.SpatialPath(h_rs)
        out_0 = self.conv_sp_fc(spatial_out)

        semantic_out = self.decode_path(out)
        out_1 = self.conv_fc(semantic_out)

        # for feature in out:
        # print(spatial_out.shape, semantic_out.shape)
        out = self.FeatureFusionModule(spatial_out, semantic_out)
        # print(out_0.shape, out_1.shape, out.shape)
        #     print(feature.shape)

        # out_0 = F.interpolate(out_0, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        # out_1 = F.interpolate(out_1, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        # out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        print('out_0: ', out_0.shape)
        print('out_1: ', out_1.shape)
        print('out: ', out.shape)
        return out_0, out_1, out

# class ConvModule(nn.Sequential):
#     def __init__(self, c1, c2, k, s=1, p=0):
#         super().__init__(
#             nn.Conv2d(c1, c2, k, s, p, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.ReLU(True)
#         )

class ConvModuleCBA(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)        # use SyncBN in original  # nn.GroupNorm(32, c2)
        self.activate = nn.LeakyReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activate(self.bn(self.conv(x)))

class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class LightHead(nn.Module):
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 19):
        super().__init__()
        for i, dim in enumerate(dims):
            print('MLP_i: ', i)
            print('MLP_dim: ', dim)
            self.add_module(f"linear_c{i+1}", MLP(dim, embed_dim))
            # self.add_module(f"MCFEB_c{i+1}", MCFEB(ch, channel, self.ch2dim[ch], self.ch2dim[ch]))  # (ch, channel, self.ch2dim[ch], self.ch2dim[ch])

        self.linear_fuse = ConvModuleCBA(embed_dim*4, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        B, _, H, W = features[0].shape
        outs = [self.linear_c1(features[0]).permute(0, 2, 1).reshape(B, -1, *features[0].shape[-2:])]

        for i, feature in enumerate(features[1:]):
            cf = eval(f"self.linear_c{i+2}")(feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            # cf = MCFEB()
            outs.append(F.interpolate(cf, size=(H, W), mode='bilinear', align_corners=False))

        seg = self.linear_fuse(torch.cat(outs[::-1], dim=1))
        seg = self.linear_pred(self.dropout(seg))
        print('243296320: ', seg.shape)
        return seg

class FPN(nn.Module):
    def __init__(self, n_class):
        super(FPN,self).__init__()
        self.n_class=n_class
        # self.out_channels = 150
        self.backbone = ResNet('50')
        # self.backbone.load_state_dict(torch.load('resnet50.pth', map_location='cpu'))
        self.head = FPNHead([256, 512, 1024, 2048], 128, self.n_class)
        # self.semantic_img_model = SegFormer('B3', self.out_channels)
        # self.semantic_img_model.load_state_dict(torch.load('.\\segformer.b3.ade.pth', map_location='cpu'))
        # # print(self.semantic_img_model)

    def forward(self, h_rs):
        # x = torch.randn(2, 3, 96, 96)
        features = self.backbone(h_rs)
        out = self.head(features)
        out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        # print(out.shape)
        return out

class fcn_resnet50(nn.Module):
    def __init__(self, n_class):
        super(fcn_resnet50,self).__init__()
        self.n_class=n_class
        # self.out_channels = 150
        self.FCN = FCN(self.n_class)

    def forward(self, h_rs):
        out = self.FCN(h_rs)
        # print(out.shape)
        return out

class deeplabv3_resnet50(nn.Module):
    def __init__(self, n_class):
        super(deeplabv3_resnet50,self).__init__()
        self.n_class=n_class
        # self.out_channels = 150
        self.deeplabv3 = deeplabv3(self.n_class)

    def forward(self, h_rs):
        out = self.deeplabv3(h_rs)
        # print(out.shape)
        return out

class UNet_model(nn.Module):
    def __init__(self, n_class):
        super(UNet_model,self).__init__()
        self.n_class=n_class
        # self.out_channels = 150
        self.UNet = UNet(3, self.n_class)

    def forward(self, h_rs):
        out = self.UNet(h_rs)
        # print(out.shape)
        return out

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )

class FPNHead(nn.Module):
    """Panoptic Feature Pyramid Networks
    https://arxiv.org/abs/1901.02446
    """
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        self.lateral_convs = nn.ModuleList([])
        self.output_convs = nn.ModuleList([])

        for ch in in_channels[::-1]:
            self.lateral_convs.append(ConvModule(ch, channel, 1))
            self.output_convs.append(ConvModule(channel, channel, 3, 1, 1))

        self.conv_seg = nn.Conv2d(channel, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.lateral_convs[0](features[0])

        for i in range(1, len(features)):
            out = F.interpolate(out, scale_factor=2.0, mode='nearest')
            out = out + self.lateral_convs[i](features[i])
            out = self.output_convs[i](out)
        out = self.conv_seg(self.dropout(out))
        return out

# if __name__ == "__main__":
if __name__ == '__main__':
    img = torch.randn(16, 3, 512, 512).to('cuda')
    model = LightSDNet(6).to('cuda')
    out = model(img)
    print(out[2].shape)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # LightSDNet()
