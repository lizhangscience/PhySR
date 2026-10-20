# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, filters, dropout_rate=0.1):
        super(EnhancedResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(filters, filters // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(filters, filters // 2, kernel_size=5, padding=2)

        self.dropout = nn.Dropout(dropout_rate)

        self.bn2 = nn.BatchNorm2d(filters)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv4 = nn.Conv2d(filters, filters, kernel_size=3, padding=1)

        self.relu3 = nn.LeakyReLU(0.1)

        self.adjust = None
        if in_channels != filters:
            self.adjust = nn.Conv2d(in_channels, filters, kernel_size=1)

    def forward(self, x):
        residual = x

        if self.adjust:
            residual = self.adjust(residual)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.conv2(x)
        x2 = self.conv3(x)

        x = torch.cat([x1, x2], dim=1)
        x = self.dropout(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv4(x)

        x += residual

        return self.relu3(x)

class DetailEnhancedUpsampler(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(DetailEnhancedUpsampler, self).__init__()

        self.scale = scale

        self.main_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.edge_conv = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.edge_conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        main_branch = nn.functional.interpolate(
            x,
            scale_factor=self.scale,
            mode='bilinear',
            align_corners=False
        )
        main_branch = self.main_conv(main_branch)

        edge_x = self.edge_conv(x)
        edge_x = nn.functional.leaky_relu(edge_x, 0.1)

        edge_upsampled = nn.functional.interpolate(
            edge_x,
            scale_factor=self.scale,
            mode='bilinear',
            align_corners=False
        )
        edge_upsampled = self.edge_conv2(edge_upsampled)

        merged = main_branch + edge_upsampled
        merged = self.bn(merged)
        merged = self.relu(merged)

        return merged

class DynamicScaleUNet(nn.Module):
    def __init__(self, im_dim_input, im_dim_gt, super_resolution_scale, dropout_rate=0.1):
        super(DynamicScaleUNet, self).__init__()

        self.super_resolution_scale = super_resolution_scale
        self.im_dim_input = im_dim_input
        self.im_dim_gt = im_dim_gt

        assert im_dim_gt == im_dim_input * super_resolution_scale, \
            f"Output size must be input size times scale (got {im_dim_gt} != {im_dim_input} x {super_resolution_scale})"

        self.upsample_steps = int(np.log2(super_resolution_scale))
        assert 2 ** self.upsample_steps == super_resolution_scale, \
            "Only power-of-two scales are supported (e.g., 2x/4x/8x/16x)"

        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.enc_res1 = EnhancedResidualBlock(32, 32, dropout_rate)
        self.pool1 = nn.MaxPool2d(2)

        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.enc_res2 = EnhancedResidualBlock(64, 64, dropout_rate)
        self.pool2 = nn.MaxPool2d(2)

        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_res3 = EnhancedResidualBlock(128, 128, dropout_rate)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(128, 192, kernel_size=3, padding=1)
        self.bottleneck_res1 = EnhancedResidualBlock(192, 192, dropout_rate)
        self.bottleneck_res2 = EnhancedResidualBlock(192, 192, dropout_rate)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(192, 128, kernel_size=2, stride=2)
        self.dec_att1 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.dec_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_res1 = EnhancedResidualBlock(128, 128, dropout_rate)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_att2 = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.dec_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_res2 = EnhancedResidualBlock(64, 64, dropout_rate)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_att3 = nn.Conv2d(32, 32, kernel_size=1, padding=0)
        self.dec_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec_res3 = EnhancedResidualBlock(32, 32, dropout_rate)

        # Retain the cascaded super-resolution upsampling module.
        self.dynamic_upsamplers = nn.ModuleList()

        in_channels = 32
        out_channels = 16

        for _ in range(self.upsample_steps):
            self.dynamic_upsamplers.append(
                DetailEnhancedUpsampler(
                    in_channels,
                    out_channels,
                    scale=2
                )
            )

            in_channels = out_channels
            out_channels = max(out_channels // 2, 4)

        self.output_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='leaky_relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        enc1 = nn.functional.leaky_relu(self.enc_conv1(x), 0.1)
        enc1 = self.enc_res1(enc1)
        pool1 = self.pool1(enc1)

        enc2 = nn.functional.leaky_relu(self.enc_conv2(pool1), 0.1)
        enc2 = self.enc_res2(enc2)
        pool2 = self.pool2(enc2)

        enc3 = nn.functional.leaky_relu(self.enc_conv3(pool2), 0.1)
        enc3 = self.enc_res3(enc3)
        pool3 = self.pool3(enc3)

        bottleneck = nn.functional.leaky_relu(self.bottleneck_conv(pool3), 0.1)
        bottleneck = self.bottleneck_res1(bottleneck)
        bottleneck = self.bottleneck_res2(bottleneck)

        up1 = self.upconv1(bottleneck)
        up1 = nn.functional.leaky_relu(up1, 0.1)

        att1 = torch.sigmoid(self.dec_att1(enc3))
        enc3_filtered = enc3 * att1
        concat1 = torch.cat([up1, enc3_filtered], dim=1)

        dec1 = nn.functional.leaky_relu(self.dec_conv1(concat1), 0.1)
        dec1 = self.dec_res1(dec1)

        up2 = self.upconv2(dec1)
        up2 = nn.functional.leaky_relu(up2, 0.1)

        att2 = torch.sigmoid(self.dec_att2(enc2))
        enc2_filtered = enc2 * att2
        concat2 = torch.cat([up2, enc2_filtered], dim=1)

        dec2 = nn.functional.leaky_relu(self.dec_conv2(concat2), 0.1)
        dec2 = self.dec_res2(dec2)

        up3 = self.upconv3(dec2)
        up3 = nn.functional.leaky_relu(up3, 0.1)

        att3 = torch.sigmoid(self.dec_att3(enc1))
        enc1_filtered = enc1 * att3
        concat3 = torch.cat([up3, enc1_filtered], dim=1)

        dec3 = nn.functional.leaky_relu(self.dec_conv3(concat3), 0.1)
        dec3 = self.dec_res3(dec3)

        x_upsampled = dec3

        for upsampler in self.dynamic_upsamplers:
            x_upsampled = upsampler(x_upsampled)

        output = self.output_conv(x_upsampled)

        max_val = torch.max(output)
        output = torch.clamp(output, 0.0, max_val.item())

        return output
