# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class LossFunctions:
    """
    Preserve the original computation of each PhySR loss term.
    This module computes raw losses only; it does not assign fixed physics/SSIM/MSE weights.
    Adaptive balancing of the three main losses is handled by adaptive_weighting.py.
    """

    def __init__(self, config, psf, beam):
        self.config = config
        self.psf = psf
        self.beam = beam
        self.super_resolution_scale = config.super_resolution_scale

        assert self.psf.shape[2:] == self.beam.shape == (config.im_dim_input, config.im_dim_input), \
            f"PSF/beam dimensions must match input size {config.im_dim_input}x{config.im_dim_input}!"

        self.beam_norm = self.beam / (torch.max(self.beam) + 1e-8)

        threshold = 0.2
        mask = (self.beam_norm > threshold).float()
        self.mask = mask.unsqueeze(0).unsqueeze(0)

        self.tv_weight = float(getattr(config, 'tv_weight', 1e-4))
        self.artifact_weight = float(getattr(config, 'artifact_weight', 5e-4))

    def _dynamic_downsample(self, x):
        batch_size, channels, h, w = x.shape
        s = self.super_resolution_scale

        assert h % s == 0, f"SR output height {h} must be divisible by scale {s}"
        assert w % s == 0, f"SR output width {w} must be divisible by scale {s}"
        assert channels == 1, f"Downsampling supports only single-channel input; got {channels} channels"

        x_reshaped = x.reshape(batch_size, channels, h // s, s, w // s, s)
        return x_reshaped.mean(dim=(3, 5))

    def masked_mse(self, x, y):
        mask = self.mask
        diff2 = (x - y) ** 2 * mask
        return diff2.sum() / (mask.sum() + 1e-8)

    def radio_physics_loss(self, y_true, y_pred):
        y_pred_downsampled = self._dynamic_downsample(y_pred)

        beam_expanded = self.beam.unsqueeze(0).unsqueeze(0)
        y_beam = y_pred_downsampled * beam_expanded

        _, _, kh, kw = self.psf.shape
        pad_h_total = kh - 1
        pad_w_total = kw - 1

        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left

        y_pad = nn.functional.pad(
            y_beam,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='reflect'
        )

        simulated_dirty = nn.functional.conv2d(
            y_pad,
            self.psf,
            stride=1,
            padding=0
        )

        simulated_dirty = torch.where(
            torch.isnan(simulated_dirty),
            torch.zeros_like(simulated_dirty),
            simulated_dirty
        )
        simulated_dirty = torch.where(
            torch.isinf(simulated_dirty),
            torch.zeros_like(simulated_dirty),
            simulated_dirty
        )

        return self.masked_mse(y_true, simulated_dirty)

    def ssim_loss(self, x, y):
        C1 = (0.01 * torch.max(torch.cat([x, y]))) ** 2
        C2 = (0.03 * torch.max(torch.cat([x, y]))) ** 2

        mu_x = nn.functional.avg_pool2d(x, 3, 1, 1)
        mu_y = nn.functional.avg_pool2d(y, 3, 1, 1)

        sigma_x = nn.functional.avg_pool2d(x ** 2, 3, 1, 1) - mu_x ** 2
        sigma_y = nn.functional.avg_pool2d(y ** 2, 3, 1, 1) - mu_y ** 2
        sigma_xy = nn.functional.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        ssim_map = ssim_n / ssim_d

        return 1 - torch.mean(ssim_map)

    def tv_loss(self, x):
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        return dh.mean() + dw.mean()

    def artifact_suppression_loss(self, y_pred_downsampled):
        beam_norm = self.beam_norm.unsqueeze(0).unsqueeze(0)
        weight = 1.0 - beam_norm
        return (torch.abs(y_pred_downsampled) * weight).mean()

    def compute_components(self, y_true, y_pred, clean_label):
        physics_loss = self.radio_physics_loss(y_true, y_pred)
        y_pred_downsampled = self._dynamic_downsample(y_pred)

        ssim_loss = self.ssim_loss(clean_label, y_pred_downsampled)
        mse_loss = self.masked_mse(clean_label, y_pred_downsampled)
        tv = self.tv_loss(y_pred_downsampled)
        artifact = self.artifact_suppression_loss(y_pred_downsampled)

        return {
            'physics': physics_loss,
            'ssim': ssim_loss,
            'mse': mse_loss,
            'tv': tv,
            'artifact': artifact,
        }

    def regularization_loss(self, components):
        return (
            self.tv_weight * components['tv'] +
            self.artifact_weight * components['artifact']
        )
