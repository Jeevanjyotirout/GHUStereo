from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
import timm

from .submodule import *
from typing import List, Optional, Tuple

class SubModule(nn.Module):
    def __init__(self) -> None:
        super(SubModule, self).__init__()

    def weight_init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Feature(SubModule):
    def __init__(self, backbone: str) -> None:
        super(Feature, self).__init__()
        self.backbone = backbone
        pretrained =  True
        if backbone == "efficientnet_b2a":
            model = timm.create_model('efficientnet_b2a', pretrained=pretrained, features_only=True)
            layers = [1, 2, 3, 5, 6]
            self.chans = [16, 24, 48, 120, 208]
            self.conv_stem = model.conv_stem
            self.bn1 = model.bn1
            self.act1 = nn.ReLU6()

        elif backbone == "mobilenetv2_100":
            model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
            layers = [1, 2, 3, 5, 6]
            self.chans = [16, 24, 32, 96, 160]
            self.conv_stem = model.conv_stem
            self.bn1 = model.bn1
            self.act1 = nn.ReLU6()

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

    def forward(self, x:     torch.Tensor) -> List[    torch.Tensor]:

        x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)

        return [x4, x8, x16, x32]

class FeatUp(SubModule):
    def __init__(self, chans: List[int], vol_size: int) -> None:
        super(FeatUp, self).__init__()
        self.v = vol_size
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)

        if self.v == 16:
            self.conv16 = BasicConv(chans[3]*2, chans[2]*2, kernel_size=3, stride=1, padding=1)

        if self.v in [8, 4]:
            self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)

        if self.v == 8:
            self.conv8 = BasicConv(chans[2]*2, chans[2]*2, kernel_size=3, stride=1, padding=1)

        if self.v == 4:
            self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
            self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, featL: List[    torch.Tensor], featR: Optional[List[    torch.Tensor]]=None) -> Tuple[List[    torch.Tensor], List[    torch.Tensor]]:
        x4, x8, x16, x32 = featL
        y4, y8, y16, y32 = featR
        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)


        if self.v == 16:
            x16 = self.conv16(x16)
            y16 = self.conv16(y16)

        if self.v in [8, 4]:
            x8 = self.deconv16_8(x16, x8)
            y8 = self.deconv16_8(y16, y8)

        if self.v == 8:
            x8 = self.conv8(x8)
            y8 = self.conv8(y8)

        if self.v == 4:
            x4 = self.deconv8_4(x8, x4)
            y4 = self.deconv8_4(y8, y4)
            x4 = self.conv4(x4)
            y4 = self.conv4(y4)

        return [x4, x8, x16, x32], [y4, y8, y16, y32]

class aggregation(nn.Module):
    def __init__(self, in_channels: int, add_channel: int) -> None:
        super(aggregation, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels+add_channel, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels+add_channel, in_channels+add_channel, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=(1, 1, 1), stride=1, dilation=1))

        self.conv2 = nn.Sequential(BasicConv(in_channels+add_channel, in_channels+add_channel*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=(1, 1, 1), stride=2, dilation=1),
                                   BasicConv(in_channels+add_channel*2, in_channels+add_channel*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=(1, 1, 1), stride=1, dilation=1))

        self.conv3 = nn.Sequential(BasicConv(in_channels+add_channel*2, in_channels+add_channel*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels+add_channel*4, in_channels+add_channel*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv3_up = BasicConv(in_channels+add_channel*4, in_channels+add_channel*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels+add_channel*2, in_channels+add_channel, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels+add_channel, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))


        self.agg_0 = nn.Sequential(BasicConv(2*in_channels+add_channel*4, in_channels+add_channel*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels+add_channel*2, in_channels+add_channel*2, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(2*in_channels+add_channel*2, in_channels+add_channel, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels+add_channel, in_channels+add_channel, is_3d=True, kernel_size=3, padding=1, stride=1))


    def forward(self, x:     torch.Tensor) ->     torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up[:, :, 0:conv2.shape[2], 0:conv2.shape[3], 0:conv2.shape[4]], conv2), dim=1)
        conv2 = self.agg_0(conv2)

        conv2_up = self.conv2_up(conv2)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)

        conv = self.conv1_up(conv1)

        return conv


class upsample(nn.Module):
    def __init__(self, C: int, cf1: int, cf2: int) -> None:
        super(upsample, self).__init__()


        self.conv1 = nn.Sequential(BasicConv(1, C, is_3d=False, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(C, C, is_3d=False, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv2 = nn.Sequential(BasicConv(C, C, is_3d=False, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(C, C, is_3d=False, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv3 = nn.Sequential(BasicConv(C, C, is_3d=False, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(C, C, is_3d=False, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))

        self.conv3_up = BasicConv(C, C, deconv=True, is_3d=False, bn=True,
                                  relu=True, kernel_size=4, padding=1, stride=2)

        self.conv2_up = BasicConv(C, C, deconv=True, is_3d=False, bn=True,
                                  relu=True, kernel_size=4, padding=1, stride=2)

        self.conv1_up = BasicConv(C, 1, deconv=True, is_3d=False, bn=False,
                                  relu=False, kernel_size=4, padding=1, stride=2)


        self.agg_0 = nn.Sequential(BasicConv(2*C+cf1, C, is_3d=False, kernel_size=1, padding=0, stride=1),
                                   BasicConv(C, C, is_3d=False, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(2*C+cf2, C, is_3d=False, kernel_size=1, padding=0, stride=1),
                                   BasicConv(C, C, is_3d=False, kernel_size=3, padding=1, stride=1))

        self.dm = nn.Sequential(BasicConv(1, C, is_3d=False, kernel_size=5, padding=1, stride=1),
                                BasicConv(C, C, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C, C, is_3d=False, kernel_size=3, padding=1, stride=1),
                                BasicConv(C, C, is_3d=False, kernel_size=1, padding=1, stride=1))

        self.spx_4 = nn.Sequential(BasicConv(C+cf2, C, kernel_size=3, stride=1, padding=1),
                                   nn.Conv2d(C, C, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(C), nn.ReLU()
                                   )


        self.spx = nn.Sequential(nn.ConvTranspose2d(C, 9, kernel_size=4, stride=2, padding=1),)

    def forward(self, left_f1x:     torch.Tensor, left_f2x:     torch.Tensor, init_disp:     torch.Tensor) ->     torch.Tensor:

        disp_features = self.dm(init_disp)
        cat_features = self.spx_4(torch.cat((disp_features, left_f2x), dim=1))
        cat_features = self.spx(cat_features)

        sfm = F.softmax(cat_features, 1)

        b, c, h, w = init_disp.shape
        disp_unfold = F.unfold(init_disp, 3, 1, 1).reshape(b, -1, h, w)
        disp_unfold = F.interpolate(disp_unfold, (h * 2, w * 2), mode='nearest').reshape(b, 9, h * 2, w * 2)

        disp = (disp_unfold * sfm).sum(1).unsqueeze(1)
        conv1 = self.conv1(disp)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up[:, 0:conv2.shape[1], 0:conv2.shape[2], 0:conv2.shape[3]], conv2, left_f1x), dim=1)
        conv2 = self.agg_0(conv2)

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1, left_f2x), dim=1)

        conv1 = self.agg_1(conv1)
        conv = self.conv1_up(conv1)

        return conv + disp

class GHUStereo(nn.Module):
    def __init__(self, maxdisp: int,  gwc: bool=False, norm_correlation: bool=True, backbone: str="efficientnet_b2a", cv_scale: int=8) -> None:
        super(GHUStereo, self).__init__()
        self.maxdisp = maxdisp
        self.vol_size = cv_scale
        self.backbone = backbone
        self.feature = Feature(backbone)
        self.feature_up = FeatUp(self.feature.chans, self.vol_size)

        if self.vol_size == 4:
            self.stem_2 = nn.Sequential(
                BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU()
                )

            self.stem_4 = nn.Sequential(
                BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(48, 48, 3, 1, 1, bias=False),
                nn.BatchNorm2d(48), nn.ReLU()
                )

        if self.vol_size == 8:
            self.stem_2 = nn.Sequential(
                BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.BatchNorm2d(32), nn.ReLU()
                )

            self.stem_4 = nn.Sequential(
                BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(48, 48, 3, 1, 1, bias=False),
                nn.BatchNorm2d(48), nn.ReLU()
                )

            self.stem_8 = nn.Sequential(
                BasicConv(48, 64, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU()
                )

        if backbone == "efficientnet_b2a":
            if self.vol_size == 4:
                self.conv = BasicConv(96, 64, kernel_size=3, padding=1, stride=1)
                self.desc = nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)

            elif self.vol_size == 8:
                self.conv = BasicConv(160, 64, kernel_size=3, padding=1, stride=1)
                self.desc = nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)

            elif self.vol_size == 16:
                self.conv = BasicConv(176, 64, kernel_size=3, padding=1, stride=1)
                self.desc = nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)

        elif backbone == "mobilenetv2_100":
            self.conv = BasicConv(128, 64, kernel_size=3, padding=1, stride=1)
            self.desc = nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1)

        self.gwc = gwc
        self.norm_correlation = norm_correlation

        reduction_multiplier = 8

        if self.norm_correlation:
            print("Cost volumes: norm correlation")
            self.corr_stem = BasicConv(1, reduction_multiplier, deconv=False, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1)

        if self.gwc:
            print("Cost volumes: gwc ")
            self.num_groups = 32
            self.group_stem = BasicConv(self.num_groups, reduction_multiplier, deconv=False, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1)

        self.agg = BasicConv(reduction_multiplier, reduction_multiplier, deconv=False, is_3d=True, bn=True, relu=True, kernel_size=3, padding=1, stride=1)

        if self.vol_size == 4:
            add_channel = 16

        elif self.vol_size == 8:
            add_channel = 8

        self.aggregation_out = aggregation(reduction_multiplier, add_channel)

        if self.vol_size == 4:
               if backbone == "efficientnet_b2a":
                    self.upsample_module_4_2 = upsample(64, 96, 48)
                    self.upsample_module_2_1 = upsample(64, 48, 32)
               elif backbone == "mobilenetv2_100":
                    raise NotImplementedError

        elif self.vol_size == 8:
            if backbone == "efficientnet_b2a":
                self.upsample_module_8_4 = upsample(32, 240, 96)
                self.upsample_module_4_2 = upsample(32, 96, 24)
                self.upsample_module_2_1 = upsample(32, 24, 32)

            elif backbone == "mobilenetv2_100":
                self.upsample_module_8_4 = upsample(32, 192, 64)
                self.upsample_module_4_2 = upsample(32, 64, 24)
                self.upsample_module_2_1 = upsample(32, 24, 32)

    def forward(self, left:     torch.Tensor, right:     torch.Tensor, train_status: bool) -> List[    torch.Tensor]:

        features_left = self.feature(left)
        features_right = self.feature(right)
        features_left, features_right = self.feature_up(features_left, features_right)
        if self.vol_size == 4:

            stem_2x = self.stem_2(left)
            stem_2y = self.stem_2(right)

            stem_4x = self.stem_4(stem_2x)
            stem_4y = self.stem_4(stem_2y)

            match_left = torch.cat((features_left[0], stem_4x), 1)
            match_right = torch.cat((features_right[0], stem_4y), 1)

            match_left = self.desc(self.conv(match_left))
            match_right = self.desc(self.conv(match_right))

        if self.vol_size == 8:
            stem_2x = self.stem_2(left)
            stem_2y = self.stem_2(right)

            stem_4x = self.stem_4(stem_2x)
            stem_4y = self.stem_4(stem_2y)

            stem_8x = self.stem_8(stem_4x)
            stem_8y = self.stem_8(stem_4y)

            match_left = torch.cat((features_left[1], stem_8x), 1)
            match_right = torch.cat((features_right[1], stem_8y), 1)

            match_left = self.desc(self.conv(match_left))
            match_right = self.desc(self.conv(match_right))

        if self.norm_correlation:
            volume = build_norm_correlation_volume(match_left, match_right, self.maxdisp // self.vol_size)
            volume = self.corr_stem(volume)

        if self.gwc:
            volume = build_gwc_volume(match_left, match_right, self.maxdisp // self.vol_size, self.num_groups)
            volume = self.group_stem(volume)

        volume = self.agg(volume)
        cost = self.aggregation_out(volume)

        if self.vol_size == 4:
            disp_samples = torch.arange(0, self.maxdisp // self.vol_size, dtype=cost.dtype, device=cost.device)
            disp_samples = disp_samples.view(1, self.maxdisp // self.vol_size, 1, 1).repeat(cost.shape[0], 1, cost.shape[3], cost.shape[4])
            init_pred = regression_topk(cost.squeeze(1), disp_samples, 2)
            disp_2 = self.upsample_module_4_2(features_left[1], features_left[0], init_pred)
            disp_1 = self.upsample_module_2_1(features_left[0], stem_2x, disp_2)

        if self.vol_size == 8:
            init_pred = disparity_regression(cost.squeeze(1), self.maxdisp // self.vol_size).unsqueeze(1)
            disp_4 = self.upsample_module_8_4(features_left[2], features_left[1], init_pred)
            disp_2 = self.upsample_module_4_2(features_left[1], features_left[0], disp_4)
            disp_1 = self.upsample_module_2_1(features_left[0], stem_2x, disp_2)

        if train_status:

            if self.vol_size == 4:
                return [disp_1.squeeze(1)*4,
                        disp_2.squeeze(1)*4]

            if self.vol_size == 8:
                return [disp_1.squeeze(1)*4,
                        disp_2.squeeze(1)*4,
                        disp_4.squeeze(1)*4]
        else:
            return [disp_1.squeeze(1)*4]


