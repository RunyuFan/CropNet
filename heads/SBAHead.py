import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import DeformConv2d
# from cbam import CBAM
# from .fpn import ResNet
# from .cbam import CBAM
import torch
from torch import nn
from torch.nn.parameter import Parameter
import math
import torch
import numpy as np
from axial_attention import AxialAttention, AxialPositionalEmbedding
from .ContextBlock2d import ContextBlock2d
from torch.nn import BatchNorm2d

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


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

class DCNv2(nn.Module):
    def __init__(self, c1, c2, k, s, p, g=1):
        super().__init__()
        self.dcn = DeformConv2d(c1, c2, k, s, p, groups=g)
        self.offset_mask = nn.Conv2d(c2,  g* 3 * k * k, k, s, p)
        self._init_offset()

    def _init_offset(self):
        self.offset_mask.weight.data.zero_()
        self.offset_mask.bias.data.zero_()

    def forward(self, x, offset):
        out = self.offset_mask(offset)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        # print('o1, o2, mask', o1.shape, o2.shape, mask.shape)
        offset = torch.cat([o1, o2], dim=1)
        mask = mask.sigmoid()
        # print('x, offset, mask', x.shape, offset.shape, mask.shape)
        return self.dcn(x, offset, mask)



class FSM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv_atten = nn.Conv2d(c1, c1, 1, bias=False)
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        atten = self.conv_atten(F.avg_pool2d(x, x.shape[2:])).sigmoid()
        feat = torch.mul(x, atten)
        x = x + feat
        return self.conv(x)


class DCFAM(nn.Module):
    def __init__(self, c1, c2, W, H):
        super().__init__()
        self.lateral_conv = FSM(c1, c2) #FSM(c1, c2, W, H)
        self.offset = nn.Conv2d(c2*2, c2, 1, bias=False)
        self.dcpack_l2 = DCNv2(c2, c2, 3, 1, 1, 8)

    def forward(self, feat_l, feat_s):
        feat_up = feat_s
        if feat_l.shape[2:] != feat_s.shape[2:]:
            feat_up = F.interpolate(feat_s, size=feat_l.shape[2:], mode='bilinear', align_corners=False)

        feat_arm = self.lateral_conv(feat_l)
        offset = self.offset(torch.cat([feat_arm, feat_up*2], dim=1))

        feat_align = F.relu(self.dcpack_l2(feat_up, offset))
        # feat_align = self.dcpack_l2(feat_up, offset)
        return feat_align + feat_arm


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MultiscaleContextFeatureEnhanceModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(MultiscaleContextFeatureEnhanceModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.relu = nn.ReLU(inplace=True)
        # branch 1
        self.conv_branch1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 2
        self.conv_branch2 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch2_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 3
        self.conv_branch3 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch3_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 4
        self.conv_branch4 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch4_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_branch5 = nn.Conv2d(self.in_d, self.out_d, kernel_size=1)
        self.conv_branchContextBlock2d = nn.Conv2d(self.in_d, self.out_d, kernel_size=1)
        self.ContextBlock2d = ContextBlock2d(self.in_d, self.in_d // 4, "att", ["channel_add"])
        self.edge_enhance = Sobel(self.in_d)

    def forward(self, x):

        # branch 1
        x_branch1 = self.conv_branch1(x)
        x_branchContextBlock2d = self.relu(self.conv_branchContextBlock2d(self.ContextBlock2d(x)))
        x_branchEdge = self.edge_enhance(x)

        # branch 2
        x_branch2 = self.relu(self.conv_branch2(x) + x_branch1)
        x_branch2 = self.conv_branch2_f(x_branch2)
        # branch 3
        x_branch3 = self.relu(self.conv_branch3(x) + x_branch2)
        x_branch3 = self.conv_branch3_f(x_branch3)
        # branch 4
        x_branch4 = self.relu(self.conv_branch4(x) + x_branch3)
        x_branch4 = self.conv_branch4_f(x_branch4)
        x_out = self.relu(self.conv_branch5(x) + x_branch4)

        return torch.cat([x_out + x_branchContextBlock2d, x_branchEdge], 1)

class MSCB(nn.Module):
    def __init__(self, channel, W, H):
        super(MSCB, self).__init__()
        self.in_channels = channel
        self.hide_channels = 128

        self.MultiscaleContextFeatureEnhanceModule = MultiscaleContextFeatureEnhanceModule(self.in_channels, self.hide_channels)
        self.augmented_representation =  nn.Conv2d(self.hide_channels+2, self.in_channels, kernel_size=1,stride=1,padding=0)


    def forward(self, feature):

        feature_context =self.MultiscaleContextFeatureEnhanceModule(feature)
        out = self.augmented_representation(feature_context)
        # out = F.interpolate(out, size=h_rs.shape[-2:], mode='bilinear', align_corners=False)
        return out


class Sobel(nn.Module):
    def __init__(self, inchannels):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=inchannels, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        sobel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).astype(dtype='float32')
        sobel_x = np.reshape(sobel_x, (1, 3, 3))
        sobel_y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).astype(dtype='float32')
        sobel_y = np.reshape(sobel_y, (1, 3, 3))

        sobel_x = np.repeat(sobel_x, inchannels, axis=0).reshape((1, inchannels, 3, 3))
        sobel_y = np.repeat(sobel_y, inchannels, axis=0).reshape((1, inchannels, 3, 3))

        # print(sobel_x.shape, sobel_y.shape)
        sobel = torch.cat([torch.from_numpy(sobel_x),torch.from_numpy(sobel_y)], 0)
        # G = G.unsqueeze(1)
        # G = G.expand(inchannels, 3, 3)
        self.filter.weight = nn.Parameter(sobel, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        # x = torch.mul(x, x)
        # x = torch.sum(x, dim=1, keepdim=True)
        # x = torch.sqrt(x)
        return x


class HierarchicalFeatureAggregation(nn.Module):
    """Hierarchical Feature Aggregation module"""

    def __init__(self, channel_in, channel_out, k_size=3):
        super(HierarchicalFeatureAggregation, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # # self.max_pool = nn.AdaptiveMaxPool2d(1)
        # # self.avg_pool_1d = nn.AdaptiveAvgPool1d(1)
        # self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2)  # padding=(k_size - 1) // 2
        # self.sigmoid = nn.Sigmoid()
        # self.conv_c1 = nn.Sequential(
        #     # nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(channel_in),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv_c2 = nn.Sequential(
        #     # nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(channel_in),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv_c3 = nn.Sequential(
        #     # nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(channel_in, channel_in // 2, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(channel_in // 2),
        #     nn.ReLU(inplace=True)
        # )
        # self.conv_c4 = nn.Sequential(
        #     # nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(channel_in, channel_in // 2, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(channel_in // 2),
        #     nn.ReLU(inplace=True)
        # )
        self.conv_fc3 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1),
            # nn.BatchNorm2d(channel_in),
            # nn.ReLU(inplace=True)
        )
        self.conv_fc4 = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1),
            # nn.BatchNorm2d(channel_in),
            # nn.ReLU(inplace=True)
        )


    def forward(self, x3, x4):

        # x1 = F.interpolate(self.conv_c1(x1), size=x4.shape[-2:], mode='bilinear', align_corners=False)
        # x2 = F.interpolate(self.conv_c2(x2), size=x4.shape[-2:], mode='bilinear', align_corners=False)
        # x3 = F.interpolate(self.conv_c3(x3), size=x4.shape[-2:], mode='bilinear', align_corners=False)
        #
        # x4_down = self.conv_c4(x4)
        # features_dove3 = self.upsample3(features_dove[3], output_size=features[3].size())

        # x4 = self.conv_c2(x4)
        # x4 = self.conv_c1(x4)

        #
        # x2 = F.interpolate(x2, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        # x3 = F.interpolate(x3, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        # print(x1.shape, x2.shape, x3.shape, x4.shape)
        # x = torch.cat([x3, x4_down], 1)
        # print(x.shape)
        # x = self.conv_c4(x)
        # print(self.avg_pool(x).squeeze(-1).transpose(-1, -2).shape, self.max_pool(x).squeeze(-1).transpose(-1, -2).shape)
        # y = torch.cat([self.avg_pool(x).squeeze(-1).transpose(-1, -2), self.max_pool(x).squeeze(-1).transpose(-1, -2)], 1)
        # y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        # y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        # y = self.sigmoid(y)
        return  self.conv_fc3(x3), self.conv_fc4(x4)# self.conv_fc(x * y.expand_as(x))

class MTSBAHead(nn.Module):
    def __init__(self, in_channels, channel=128, num_classes=19):
        super().__init__()
        in_channels = in_channels[::-1]
        self.align_modules = nn.ModuleList([ConvModule(in_channels[0], channel, 1)])
        self.output_convs = nn.ModuleList([])
        # self.edge_enhance_layers = nn.ModuleList([])
        # self.csa_layer_last = csa_layer(channel, 1)
        self.ch2dim = {96:16, 32:32, 24:64} # {256:16, 128:32, 64:64} # {320:16, 128:32, 64:64}
        for ch in in_channels[1:]:
            # print(ch)
            self.align_modules.append(DCFAM(ch, channel, self.ch2dim[ch], self.ch2dim[ch]))
            self.output_convs.append(MSCB(channel, self.ch2dim[ch], self.ch2dim[ch]))  # MultiscaleContextFeatureEnhanceModule
            # self.edge_enhance_layers.append(Sobel(channel))
        # self.HierarchicalFeatureAggregation = HierarchicalFeatureAggregation(channel, num_classes)
        # self.out_list = []
        # self.conv_seg = nn.Sequential(
        #     nn.Conv2d(in_channels=channel, out_channels=channel,kernel_size=1,stride=1,padding=0),
        #     # nn.BatchNorm2d(channel),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channel, num_classes, 1)
        # )
        # self.conv_seg_edge = nn.Sequential(
        #     ChannelAttentionModule(channel, 1),
        #     # nn.BatchNorm2d(channel),
        #     # nn.ReLU(inplace=True),
        #     # nn.Conv2d(channel, 1, 1),
        #     nn.Sigmoid()
        # )
        # self.conv_seg_edge = ChannelAttentionModule(channel, num_classes)

        # self.Sobel = Sobel()
        # self.csa_layer_last = csa_layer(channel, 1)
        """
        torch.Size([32, 320, 16, 16])
        torch.Size([32, 128, 32, 32])
        torch.Size([32, 64, 64, 64])
        """
    def forward(self, features) -> Tensor:
        features = features[::-1]
        out = self.align_modules[0](features[0])
        # self.out_list.append(out)
        for feat, align_module, output_conv in zip(features[1:], self.align_modules[1:], self.output_convs):
            # print(feat.shape)
            out = align_module(feat, out)
            out = output_conv(out)
        #     self.out_list.append(out)

        # for feat, align_module, output_conv in zip(features[1:], self.align_modules[1:], self.output_convs):
            # print(feat.shape)
        # out1 = self.align_modules[1](features[1], out)
        # out1 = self.output_convs[0](out1)
        #
        # out2 = self.align_modules[2](features[2], out1)
        # out2 = self.output_convs[1](out2)
        #
        # out3 = self.align_modules[3](features[3], out2)
        # out3 = self.output_convs[2](out3)
        # self.out_list.append(out)
            # out = csa_layer(out)

        #
        # for feat, align_module in zip(features[1:], self.align_modules[1:]):
        #     out = align_module(feat, out)
        #     # out = output_conv(out)
        # out_edge = self.Sobel(self.conv_seg_edge(out))
        # out_fuse = torch.cat([out_0, out_1, out_2, out], 1)
        # out_aux, out = self.HierarchicalFeatureAggregation(out2, out3)
        return out


class ConvReLU(nn.Module):
    """docstring for ConvReLU"""

    def __init__(self, channels: int = 2048, kernel_size: int = 3, stride: int = 1, padding: int = 1, dilation: int = 1,
                 out_channels=None):
        super(ConvReLU, self).__init__()
        if out_channels is None:
            self.out_channels = channels
        else:
            self.out_channels = out_channels

        self.conv = nn.Conv2d(channels, self.out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                              bias=False)
        # self.bn = nn.GroupNorm(self.out_channels, self.out_channels)
        self.relu = nn.CELU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))



if __name__ == '__main__':
    import sys
    # sys.path.insert(0, '.')
    # from models.backbones.resnet import ResNet
    # backbone = ResNet('50')
    # for y in backbone:
    #     print(y.shape)
    # head = FaPNHead([256, 512, 1024, 2048], 128, 4)
    x = torch.randn(8, 16, 512, 512)
    # for slice in range(x.shape[1]):
    features = Sobel(16)(x)
    print(features.shape)
    # out = head(features)
    # out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
    # out = FaPN([256, 512, 1024, 2048], 128, 4)(x)
    # print(out.shape)
