# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

for folder in [PROJECT_ROOT, PROJECT_ROOT / "code"]:
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from models.adaptive_weighting import AdaptiveLossBalancer
from models.dataset import DataLoaderPyTorch
from models.model import DynamicScaleUNet
from models.physics import PhysicsFieldProcessor

from device import device
from evaluator import Evaluator

from configs.SKA_x2_test_config import Config


EXPECTED_VERSION = AdaptiveLossBalancer.VERSION


def main():
    config = Config()
    os.makedirs(config.result_dir, exist_ok=True)

    print(f"Checkpoint file: {config.weight_path}")
    print(f"Results directory: {config.result_dir}")
    print(
        f"Super-resolution setup: {config.im_dim_input}x{config.im_dim_input} -> "
        f"{config.im_dim_gt}x{config.im_dim_gt} ({config.super_resolution_scale}x)"
    )

    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        torch.backends.cudnn.benchmark = True

    data_loader = DataLoaderPyTorch(config)
    test_loader, test_size = data_loader.create_dataloader_from_split(
        config.test_dir, 'test', shuffle=False
    )
    print(f"Number of test samples: {test_size}")

    model = DynamicScaleUNet(
        im_dim_input=config.im_dim_input,
        im_dim_gt=config.im_dim_gt,
        super_resolution_scale=config.super_resolution_scale
    ).to(device)

    if not os.path.exists(config.weight_path):
        raise FileNotFoundError(f"Model checkpoint not found: {config.weight_path}")

    checkpoint = torch.load(config.weight_path, map_location=device)
    if checkpoint.get('training_version') != EXPECTED_VERSION:
        raise RuntimeError(
            f"Checkpoint is not version V2: {checkpoint.get('training_version')}"
        )

    state_dict = checkpoint['model_state_dict']
    if not isinstance(model, torch.nn.DataParallel) and \
            next(iter(state_dict.keys())).startswith('module.'):
        state_dict = {
            k.replace('module.', ''): v for k, v in state_dict.items()
        }
    model.load_state_dict(state_dict)
    print(f"Loaded model checkpoint: {config.weight_path}")

    if 'adaptive_weights' in checkpoint:
        print("checkpoint adaptive weights:")
        for name, value in checkpoint['adaptive_weights'].items():
            print(f"  - {name}: {float(value):.6f}")

    physics_processor = PhysicsFieldProcessor(config)
    raw_psf = physics_processor.load_radio_psf()
    beam = physics_processor.load_primary_beam()
    psf_info = checkpoint.get('psf_info')
    if psf_info is None:
        raise KeyError('V2 checkpoint does not contain psf_info.')

    psf, prepared_info = physics_processor.prepare_psf(
        raw_psf,
        beam,
        calibration_loader=None,
        checkpoint_info=psf_info
    )
    print(f"PSF mode: {prepared_info['mode']}")
    print(f"raw PSF L1: {raw_psf.abs().sum().item():.8e}")
    print(f"prepared PSF L1: {psf.abs().sum().item():.8e}")

    evaluator = Evaluator(config, psf, beam)
    evaluator.evaluate_and_save_all(model, test_loader)
    print(f"\nSuper-resolution results saved to: {evaluator.results_dir}")


if __name__ == '__main__':
    main()
