# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import resize

from device import device


class PhysicsFieldProcessor:
    """
    Shared PSF/Beam processor.

    Supported PSF modes:
      - 'l1': exact in-model PyTorch L1 normalization used by the original
              SKA-Mid training logic: F.normalize(..., p=1, dim=(2,3)).
      - 'auto_calibrate': keep raw PSF structure and estimate one global
              amplitude scale from TRAIN data only (recommended for SKA-Low).
      - 'raw': use raw PSF as-is.
    """

    def __init__(self, config):
        self.config = config
        self.psf_path = config.psf_path
        self.beam_path = config.beam_path

    def load_radio_psf(self):
        try:
            if not os.path.exists(self.psf_path):
                raise FileNotFoundError(f"PSF file not found: {self.psf_path}")

            psf = np.load(self.psf_path).squeeze().astype(np.float32)

            if psf.shape != (self.config.im_dim_input, self.config.im_dim_input):
                psf = resize(
                    psf,
                    (self.config.im_dim_input, self.config.im_dim_input),
                    order=1,
                    mode='edge',
                    anti_aliasing=True,
                    preserve_range=True
                ).astype(np.float32)

            return torch.from_numpy(psf).unsqueeze(0).unsqueeze(0).to(device)

        except Exception as e:
            raise ValueError(f"Failed to load PSF: {e}")

    def load_primary_beam(self):
        try:
            if not os.path.exists(self.beam_path):
                raise FileNotFoundError(f"Primary-beam file not found: {self.beam_path}")

            beam = np.load(self.beam_path).squeeze().astype(np.float32)

            if beam.shape != (self.config.im_dim_input, self.config.im_dim_input):
                beam = resize(
                    beam,
                    (self.config.im_dim_input, self.config.im_dim_input),
                    order=1,
                    mode='edge',
                    anti_aliasing=True,
                    preserve_range=True
                ).astype(np.float32)

            return torch.from_numpy(beam).to(device)

        except Exception as e:
            raise ValueError(f"Failed to load primary beam: {e}")

    @staticmethod
    def _physical_forward_lr(clean_lr, psf, beam):
        beam_expanded = beam.unsqueeze(0).unsqueeze(0)
        y_beam = clean_lr * beam_expanded

        _, _, kh, kw = psf.shape
        pad_h_total = kh - 1
        pad_w_total = kw - 1

        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left

        y_pad = F.pad(
            y_beam,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='reflect'
        )

        simulated_dirty = F.conv2d(
            y_pad,
            psf,
            stride=1,
            padding=0
        )

        simulated_dirty = torch.nan_to_num(
            simulated_dirty,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
        return simulated_dirty

    def estimate_psf_scale(self, calibration_loader, raw_psf, beam):
        """
        g = argmin_g ||D - gF||_2^2 = <D,F>/<F,F>,
        where F=(clean*Beam)*raw_PSF.
        """
        max_batches = int(getattr(self.config, 'psf_calibration_batches', 0))
        eps = float(getattr(self.config, 'psf_calibration_eps', 1e-12))
        positive_only = bool(
            getattr(self.config, 'psf_calibration_positive_only', True)
        )

        beam_norm = beam / (torch.max(beam) + 1e-8)
        mask = (beam_norm > 0.2).float().unsqueeze(0).unsqueeze(0)

        numerator = torch.zeros((), device=device, dtype=torch.float64)
        denominator = torch.zeros((), device=device, dtype=torch.float64)
        used_batches = 0
        used_samples = 0

        with torch.no_grad():
            for batch_idx, (dirty_lr, clean_lr) in enumerate(calibration_loader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break

                dirty_lr = dirty_lr.to(device=device, dtype=torch.float32)
                clean_lr = clean_lr.to(device=device, dtype=torch.float32)

                sim_raw = self._physical_forward_lr(clean_lr, raw_psf, beam)
                mask_expanded = mask.expand_as(sim_raw)

                d = (dirty_lr * mask_expanded).to(torch.float64)
                f = (sim_raw * mask_expanded).to(torch.float64)

                numerator += torch.sum(d * f)
                denominator += torch.sum(f * f)
                used_batches += 1
                used_samples += dirty_lr.shape[0]

        if used_batches == 0 or denominator.item() <= eps:
            raise RuntimeError('PSF auto-calibration failed: no valid batches or denominator too small.')

        scale = (numerator / (denominator + eps)).item()

        if positive_only and scale <= 0:
            raise RuntimeError(
                f"PSF auto-calibration yielded a non-positive scale g={scale:.6e}; "
                "check the sign and amplitude conventions for dirty/clean/PSF data."
            )

        info = {
            'mode': 'auto_calibrate',
            'scale': float(scale),
            'numerator': float(numerator.item()),
            'denominator': float(denominator.item()),
            'used_batches': int(used_batches),
            'used_samples': int(used_samples),
            'raw_psf_l1': float(raw_psf.abs().sum().item()),
            'raw_psf_max': float(raw_psf.max().item()),
            'raw_psf_min': float(raw_psf.min().item()),
        }
        return float(scale), info

    def prepare_psf(self, raw_psf, beam, calibration_loader=None,
                    checkpoint_info=None):
        """Prepare PSF according to config.psf_mode or checkpoint metadata."""
        mode = str(getattr(self.config, 'psf_mode', 'auto_calibrate')).lower()

        if checkpoint_info is not None:
            ckpt_mode = str(checkpoint_info.get('mode', mode)).lower()
            if ckpt_mode != mode:
                raise RuntimeError(
                    f"PSF mode mismatch: config={mode}, checkpoint={ckpt_mode}."
                )

        if mode == 'l1':
            # Preserve the original SKA-Mid operation exactly.
            psf = F.normalize(raw_psf, p=1, dim=(2, 3))
            info = {
                'mode': 'l1',
                'scale': None,
                'raw_psf_l1': float(raw_psf.abs().sum().item()),
                'prepared_psf_l1': float(psf.abs().sum().item()),
            }
            return psf, info

        if mode == 'raw':
            info = {
                'mode': 'raw',
                'scale': 1.0,
                'raw_psf_l1': float(raw_psf.abs().sum().item()),
                'prepared_psf_l1': float(raw_psf.abs().sum().item()),
            }
            return raw_psf, info

        if mode == 'auto_calibrate':
            if checkpoint_info is not None and checkpoint_info.get('scale') is not None:
                scale = float(checkpoint_info['scale'])
                info = dict(checkpoint_info)
            else:
                if calibration_loader is None:
                    raise RuntimeError('The auto_calibrate mode requires a calibration_loader.')
                scale, info = self.estimate_psf_scale(
                    calibration_loader, raw_psf, beam
                )

            psf = raw_psf * float(scale)
            info['prepared_psf_l1'] = float(psf.abs().sum().item())
            return psf, info

        raise ValueError(
            f"Unknown psf_mode={mode}. Options: 'l1' / 'auto_calibrate' / 'raw'"
        )
