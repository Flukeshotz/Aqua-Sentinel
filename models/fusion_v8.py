import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPModule, self).__init__()
        # 1x1 conv
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        # 3x3 dilated convs
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = F.interpolate(self.global_avg_pool(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        return self.bottleneck(x)

class GatedFusion(nn.Module):
    def __init__(self, channels):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, rgb, sar):
        cat = torch.cat([rgb, sar], dim=1)
        alpha = self.gate(cat)
        return alpha * rgb + (1 - alpha) * sar

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class DualEncoderFusionUNetV8(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        # Encoders
        rgb_backbone = models.resnet18(pretrained=True)
        sar_backbone = models.resnet18(pretrained=True)
        
        # RGB Encoder
        self.rgb_init = nn.Sequential(rgb_backbone.conv1, rgb_backbone.bn1, rgb_backbone.relu)
        self.rgb_pool = rgb_backbone.maxpool
        self.rgb_l1 = rgb_backbone.layer1 # 64
        self.rgb_l2 = rgb_backbone.layer2 # 128
        self.rgb_l3 = rgb_backbone.layer3 # 256
        self.rgb_l4 = rgb_backbone.layer4 # 512
        
        # SAR Encoder
        self.sar_init = nn.Sequential(sar_backbone.conv1, sar_backbone.bn1, sar_backbone.relu)
        self.sar_pool = sar_backbone.maxpool
        self.sar_l1 = sar_backbone.layer1
        self.sar_l2 = sar_backbone.layer2
        self.sar_l3 = sar_backbone.layer3
        self.sar_l4 = sar_backbone.layer4
        
        # ASPP Blocks
        self.aspp_rgb = ASPPModule(512, 512)
        self.aspp_sar = ASPPModule(512, 512)
        
        # Gated Fusion at Bottleneck
        self.fusion_bottleneck = GatedFusion(512)
        
        # Fusion Skip Connections
        self.fusion_l3 = GatedFusion(256)
        self.fusion_l2 = GatedFusion(128)
        self.fusion_l1 = GatedFusion(64)
        self.fusion_init = GatedFusion(64)
        
        # Decoder
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, rgb, sar):
        # RGB Stream
        r0 = self.rgb_init(rgb)   # 64, 128, 128
        r1 = self.rgb_l1(self.rgb_pool(r0)) # 64, 64, 64
        r2 = self.rgb_l2(r1) # 128, 32, 32
        r3 = self.rgb_l3(r2) # 256, 16, 16
        r4 = self.rgb_l4(r3) # 512, 8, 8
        r4 = self.aspp_rgb(r4)
        
        # SAR Stream
        s0 = self.sar_init(sar)
        s1 = self.sar_l1(self.sar_pool(s0))
        s2 = self.sar_l2(s1)
        s3 = self.sar_l3(s2)
        s4 = self.sar_l4(s3)
        s4 = self.aspp_sar(s4)
        
        # Gated Fusion
        fused_bottleneck = self.fusion_bottleneck(r4, s4)
        fused_3 = self.fusion_l3(r3, s3)
        fused_2 = self.fusion_l2(r2, s2)
        fused_1 = self.fusion_l1(r1, s1)
        fused_0 = self.fusion_init(r0, s0)
        
        # Decode
        x = self.dec4(fused_bottleneck, fused_3)
        x = self.dec3(x, fused_2)
        x = self.dec2(x, fused_1)
        x = self.dec1(x, fused_0)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.final_conv(x)
