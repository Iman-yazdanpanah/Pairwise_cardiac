# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 10:19:26 2019

@author: sigma
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.aspp import build_aspp
from models.decoder import build_decoder
from models.backbone import build_backbone

class ConsistentDeepLab(nn.Module):
    def __init__(self, backbone='resnet18', in_channels=3, output_stride=8, num_classes=1,
                 sync_bn=True, freeze_bn=False, pretrained=False, **kwargs):
        super(ConsistentDeepLab, self).__init__()
        if backbone in ['drn', 'resnet18', 'resnet34']:
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, in_channels, output_stride, BatchNorm, pretrained)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, x1, x2, is_val=False):
        br1_x, low_level_feat1 = self.backbone(x1)
        br1_x = self.aspp(br1_x)
        
        if not is_val:
            br2_x, low_level_feat2 = self.backbone(x2)
            br2_x = self.aspp(br2_x)
        else:
            low_level_feat2 = low_level_feat1
            br2_x = br1_x
        
        fusion_x = br1_x + br2_x
        fusion_low_feat = low_level_feat1 + low_level_feat2
        
        x = self.decoder(fusion_x, fusion_low_feat)
        x = F.interpolate(x, size=x1.size()[2:], mode='bilinear', align_corners=True)
        # x = x.permute(0, 2, 3, 1).contiguous()
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    model = ConsistentDeepLab(backbone='resnet18', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input, input)
    print(output.size())
    print("Total paramerters: {}".format(sum(x.numel() for x in model.parameters())))   


