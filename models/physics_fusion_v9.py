import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
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

class PhysicsGuidedDualEncoder(nn.Module):
    """
    Physics-Guided Architecture V9 (Late Injection Variant)
    Maintains 3-channel input to permit loading of V8 pre-trained weights.
    Injects 2-channel Physics Modality tensors AFTER `layer1` dynamically.
    """
    def __init__(self, n_classes=1, use_physics=True, use_fp_head=True):
        super().__init__()
        self.use_physics = use_physics
        self.use_fp_head = use_fp_head
        
        # Encoders (Standard 3 Channel to allow V8 Weight Loading)
        rgb_backbone = models.resnet18(pretrained=True)
        sar_backbone = models.resnet18(pretrained=True)
        
        # RGB Encoder
        self.rgb_init = nn.Sequential(rgb_backbone.conv1, rgb_backbone.bn1, rgb_backbone.relu)
        self.rgb_pool = rgb_backbone.maxpool
        self.rgb_l1 = rgb_backbone.layer1 
        self.rgb_l2 = rgb_backbone.layer2 
        self.rgb_l3 = rgb_backbone.layer3 
        self.rgb_l4 = rgb_backbone.layer4 
        
        # SAR Encoder
        self.sar_init = nn.Sequential(sar_backbone.conv1, sar_backbone.bn1, sar_backbone.relu)
        self.sar_pool = sar_backbone.maxpool
        self.sar_l1 = sar_backbone.layer1
        self.sar_l2 = sar_backbone.layer2
        self.sar_l3 = sar_backbone.layer3
        self.sar_l4 = sar_backbone.layer4
        
        # --- TASK 2: PHYSICS INJECTION COMPRESSION ---
        if self.use_physics:
            # Reconstructs dimensionality back to standard expected limits (64 channels)
            # after concatenating the 2 physics features with the 64 native SAR features.
            self.physics_proj_sar = nn.Conv2d(64 + 2, 64, kernel_size=1, bias=False)
            self.physics_bn_sar = nn.BatchNorm2d(64)
            self.physics_proj_rgb = nn.Conv2d(64 + 2, 64, kernel_size=1, bias=False)
            self.physics_bn_rgb = nn.BatchNorm2d(64)
            
            # Initialize exactly to identity logic for stability when loading V8 weights
            nn.init.dirac_(self.physics_proj_sar.weight[:, :64, :, :])
            nn.init.zeros_(self.physics_proj_sar.weight[:, 64:, :, :])
            nn.init.dirac_(self.physics_proj_rgb.weight[:, :64, :, :])
            nn.init.zeros_(self.physics_proj_rgb.weight[:, 64:, :, :])

        # ASPP Blocks
        self.aspp_rgb = ASPPModule(512, 512)
        self.aspp_sar = ASPPModule(512, 512)
        
        # Gated Fusion at Bottleneck
        self.fusion_bottleneck = GatedFusion(512)
        
        # --- TASK 3: FALSE POSITIVE SUPPRESSION HEAD ---
        if self.use_fp_head:
            self.fp_pooling = nn.AdaptiveAvgPool2d((1, 1))
            self.fp_classifier = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 1) # Binary output (Lookalike vs True Spill)
            )
        
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

    def forward(self, rgb, sar, physics_tensor=None):
        # RGB Stream
        r0 = self.rgb_init(rgb)   
        r1 = self.rgb_l1(self.rgb_pool(r0)) 
        
        # SAR Stream
        s0 = self.sar_init(sar)
        s1 = self.sar_l1(self.sar_pool(s0))
        
        # --- INJECT PHYSICS (TASK 2) ---
        if self.use_physics and physics_tensor is not None:
            phys_scaled = F.interpolate(physics_tensor, size=s1.shape[2:], mode='bilinear', align_corners=True)
            
            # Inject into SAR encoder
            s1_cat = torch.cat([s1, phys_scaled], dim=1)
            s1 = F.relu(self.physics_bn_sar(self.physics_proj_sar(s1_cat)))
            
            # Inject into RGB Encoder natively (symmetric matching)
            r1_cat = torch.cat([r1, phys_scaled], dim=1)
            r1 = F.relu(self.physics_bn_rgb(self.physics_proj_rgb(r1_cat)))

        r2 = self.rgb_l2(r1) 
        r3 = self.rgb_l3(r2) 
        r4 = self.rgb_l4(r3) 
        r4 = self.aspp_rgb(r4)
        
        s2 = self.sar_l2(s1)
        s3 = self.sar_l3(s2)
        s4 = self.sar_l4(s3)
        s4 = self.aspp_sar(s4)
        
        # Gated Fusion
        fused_bottleneck = self.fusion_bottleneck(r4, s4)
        
        # Task 3: False Positive Classification
        lookalike_logit = None
        if self.use_fp_head:
            fp_features = self.fp_pooling(fused_bottleneck)
            fp_features = fp_features.view(fp_features.size(0), -1)
            lookalike_logit = self.fp_classifier(fp_features)
        
        # Decode Fusions
        fused_3 = self.fusion_l3(r3, s3)
        fused_2 = self.fusion_l2(r2, s2)
        fused_1 = self.fusion_l1(r1, s1)
        fused_0 = self.fusion_init(r0, s0)
        
        x = self.dec4(fused_bottleneck, fused_3)
        x = self.dec3(x, fused_2)
        x = self.dec2(x, fused_1)
        x = self.dec1(x, fused_0)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        seg_mask = self.final_conv(x)
        
        if self.use_fp_head:
            return seg_mask, lookalike_logit
        return seg_mask
