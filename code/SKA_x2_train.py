# -*- coding: utf-8 -*-
import json
import os
import pickle
import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

for folder in [PROJECT_ROOT, PROJECT_ROOT / "code"]:
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.adaptive_weighting import AdaptiveLossBalancer
from models.dataset import DataLoaderPyTorch
from models.losses import LossFunctions
from models.model import DynamicScaleUNet
from models.physics import PhysicsFieldProcessor
from device import device

# Training configuration
from configs.SKA_x2_train_config import Config


TRAINING_VERSION = AdaptiveLossBalancer.VERSION


def build_calibration_loader(train_loader, config):
    g = torch.Generator()
    g.manual_seed(config.random_seed + 12345)
    return DataLoader(
        train_loader.dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        generator=g
    )


def find_latest_checkpoint(checkpoint_dir):
    files = [
        f for f in os.listdir(checkpoint_dir)
        if f.endswith('.pth') and f.startswith('checkpoint_epoch_')
    ]
    if not files:
        return None
    files.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        reverse=True
    )
    return os.path.join(checkpoint_dir, files[0])


def get_shared_balance_parameter(model):
    m = model.module if isinstance(model, torch.nn.DataParallel) else model
    if not hasattr(m, 'output_conv'):
        raise AttributeError('The model has no output_conv for gradient-based loss balancing.')
    return m.output_conv.weight


def estimate_initial_loss_means(model, loss_functions, calibration_loader,
                                max_batches):
    was_training = model.training
    model.eval()
    sums = {'physics': 0.0, 'ssim': 0.0, 'mse': 0.0}
    count = 0

    with torch.no_grad():
        for batch_idx, (dirty_lr, clean_lr) in enumerate(calibration_loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            dirty_lr = dirty_lr.to(device=device, dtype=torch.float32)
            clean_lr = clean_lr.to(device=device, dtype=torch.float32)
            sr_pred = model(dirty_lr)
            comp = loss_functions.compute_components(
                dirty_lr, sr_pred, clean_lr
            )
            for name in sums:
                sums[name] += float(comp[name].item())
            count += 1

    if was_training:
        model.train()

    if count == 0:
        raise RuntimeError('Adaptive loss initialization failed: no usable batch.')

    return {name: sums[name] / count for name in sums}


def save_calibration_info(output_dir, psf_info, loss_init_means=None):
    path = os.path.join(output_dir, 'adaptive_calibration.json')
    payload = {
        'training_version': TRAINING_VERSION,
        'psf_preparation': psf_info,
        'initial_raw_losses': loss_init_means,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Created output directory: {config.output_dir}")
    print(
        f"Super-resolution setup: {config.im_dim_input}x{config.im_dim_input} -> "
        f"{config.im_dim_gt}x{config.im_dim_gt} ({config.super_resolution_scale}x)"
    )
    print(f"PSF mode: {config.psf_mode}")
    print(f"Adaptive loss version: {TRAINING_VERSION}")

    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)

    print("Random seeds set")
    print("\n=== Loading NPY data ===")

    data_loader = DataLoaderPyTorch(config)
    train_loader, train_size = data_loader.create_dataloader_from_split(
        config.train_dir, 'train', shuffle=True
    )
    val_loader, val_size = data_loader.create_dataloader_from_split(
        config.val_dir, 'val', shuffle=False
    )
    calibration_loader = build_calibration_loader(train_loader, config)

    print("Data loading completed:")
    print(f"  - Training samples: {train_size}")
    print(f"  - Validation samples: {val_size}")

    print("\n=== Loading raw physical-response data ===")
    physics_processor = PhysicsFieldProcessor(config)
    raw_psf = physics_processor.load_radio_psf()
    beam = physics_processor.load_primary_beam()

    print(f"  - Raw PSF shape: {tuple(raw_psf.shape)}")
    print(f"  - raw PSF L1: {raw_psf.abs().sum().item():.8e}")
    print(f"  - raw PSF min/max: {raw_psf.min().item():.8e} / {raw_psf.max().item():.8e}")
    print(f"  - Primary-beam shape: {tuple(beam.shape)}")

    print("\n=== Building model ===")
    model = DynamicScaleUNet(
        im_dim_input=config.im_dim_input,
        im_dim_gt=config.im_dim_gt,
        super_resolution_scale=config.super_resolution_scale
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Number of upsampling stages: {model.upsample_steps}")

    latest_checkpoint = find_latest_checkpoint(config.checkpoint_dir)
    checkpoint = None
    if latest_checkpoint is not None:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        ckpt_version = checkpoint.get('training_version')
        if ckpt_version != TRAINING_VERSION:
            raise RuntimeError(
                "Detected an outdated or incompatible checkpoint.\n"
                f"checkpoint version={ckpt_version}, current={TRAINING_VERSION}\n"
                "Start training with a new output_dir; do not resume incompatible earlier runs."
            )

        state_dict = checkpoint['model_state_dict']
        if isinstance(model, torch.nn.DataParallel):
            if not next(iter(state_dict.keys())).startswith('module.'):
                state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        else:
            if next(iter(state_dict.keys())).startswith('module.'):
                state_dict = {
                    k.replace('module.', ''): v for k, v in state_dict.items()
                }
        model.load_state_dict(state_dict)

    print("\n=== Preparing PSF ===")
    ckpt_psf_info = None if checkpoint is None else checkpoint.get('psf_info')
    psf, psf_info = physics_processor.prepare_psf(
        raw_psf,
        beam,
        calibration_loader=calibration_loader,
        checkpoint_info=ckpt_psf_info
    )
    print(f"  - mode: {psf_info['mode']}")
    if psf_info.get('scale') is not None:
        print(f"  - scale: {float(psf_info['scale']):.10e}")
    print(f"  - prepared PSF L1: {psf.abs().sum().item():.8e}")
    print(f"  - prepared PSF min/max: {psf.min().item():.8e} / {psf.max().item():.8e}")

    print("\n=== Configuring losses / bounded gradient balancing / optimizer ===")
    loss_functions = LossFunctions(config, psf, beam)
    loss_balancer = AdaptiveLossBalancer(config).to(device)

    loss_init_means = None
    if checkpoint is not None and 'loss_balancer_state_dict' in checkpoint:
        loss_balancer.load_state_dict(checkpoint['loss_balancer_state_dict'])
        print("Restored V2 loss balancer from checkpoint")
    else:
        loss_init_means = estimate_initial_loss_means(
            model,
            loss_functions,
            calibration_loader,
            int(config.loss_balance_init_batches)
        )
        loss_balancer.initialize_from_means(loss_init_means)
        print("Initial mean raw losses (relative training-rate baseline only; not 1/loss weights):")
        for name, value in loss_init_means.items():
            print(f"  - {name}: {value:.8e}")
        print("Initial balancing weights: physics=1, ssim=1, mse=1")

    calibration_json = save_calibration_info(
        config.output_dir, psf_info, loss_init_means
    )
    print(f"Calibration information saved to: {calibration_json}")

    # Only model parameters are optimized.  Balancer contains buffers only.
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * config.epochs
    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=5e-7
    )

    history_path = os.path.join(config.output_dir, 'history.pkl')
    start_epoch = 0
    best_selection_score = float('inf')

    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
        print(f"Loaded training history containing {len(history.get('loss', []))} epochs")
    else:
        history = {
            'loss': [],
            'val_loss': [],
            'selection_score': [],
            'downsampled_mse': [],
            'val_downsampled_mse': [],
            'downsampled_ssim': [],
            'val_downsampled_ssim': [],
            'w_physics': [],
            'w_ssim': [],
            'w_mse': [],
        }

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = int(checkpoint['epoch']) + 1
        best_selection_score = float(
            checkpoint.get('best_selection_score', float('inf'))
        )
        print(f"Resumed from checkpoint: {latest_checkpoint}")
        print(f"Continuing training from epoch {start_epoch + 1}")
    else:
        print("No compatible checkpoint found; starting training from scratch")

    shared_parameter = get_shared_balance_parameter(model)

    print("\n=== Starting training ===")
    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss = train_mse = train_ssim = 0.0
        train_count = 0

        for lr_imgs, clean_imgs in train_loader:
            lr_imgs = lr_imgs.to(device=device, dtype=torch.float32)
            clean_imgs = clean_imgs.to(device=device, dtype=torch.float32)
            optimizer.zero_grad()

            sr_pred = model(lr_imgs)
            components = loss_functions.compute_components(
                lr_imgs, sr_pred, clean_imgs
            )

            # Update weights from gradient information BEFORE the main backward.
            loss_balancer.maybe_update(components, shared_parameter)
            adaptive_main, _ = loss_balancer(components)
            regularization = loss_functions.regularization_loss(components)
            loss = adaptive_main + regularization

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clipnorm)
            optimizer.step()
            scheduler.step()

            y_pred_downsampled = loss_functions._dynamic_downsample(sr_pred)
            mse_val = loss_functions.masked_mse(clean_imgs, y_pred_downsampled)
            ssim_val = 1 - loss_functions.ssim_loss(clean_imgs, y_pred_downsampled)

            train_loss += float(loss.item())
            train_mse += float(mse_val.item())
            train_ssim += float(ssim_val.item())
            train_count += 1

        avg_train_loss = train_loss / train_count
        avg_train_mse = train_mse / train_count
        avg_train_ssim = train_ssim / train_count

        model.eval()
        val_loss = val_mse = val_ssim = 0.0
        val_count = 0

        with torch.no_grad():
            for lr_imgs, clean_imgs in val_loader:
                lr_imgs = lr_imgs.to(device=device, dtype=torch.float32)
                clean_imgs = clean_imgs.to(device=device, dtype=torch.float32)
                sr_pred = model(lr_imgs)
                components = loss_functions.compute_components(
                    lr_imgs, sr_pred, clean_imgs
                )

                adaptive_main, _ = loss_balancer(components)
                regularization = loss_functions.regularization_loss(components)
                loss = adaptive_main + regularization

                y_pred_downsampled = loss_functions._dynamic_downsample(sr_pred)
                mse_val = loss_functions.masked_mse(clean_imgs, y_pred_downsampled)
                ssim_val = 1 - loss_functions.ssim_loss(clean_imgs, y_pred_downsampled)

                val_loss += float(loss.item())
                val_mse += float(mse_val.item())
                val_ssim += float(ssim_val.item())
                val_count += 1

        avg_val_loss = val_loss / val_count
        avg_val_mse = val_mse / val_count
        avg_val_ssim = val_ssim / val_count

        # Fixed checkpoint criterion: independent of changing training weights.
        selection_score = avg_val_mse + (1.0 - avg_val_ssim)
        current_weights = loss_balancer.current_weights()
        balance_info = loss_balancer.current_info()

        history['loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['selection_score'].append(selection_score)
        history['downsampled_mse'].append(avg_train_mse)
        history['val_downsampled_mse'].append(avg_val_mse)
        history['downsampled_ssim'].append(avg_train_ssim)
        history['val_downsampled_ssim'].append(avg_val_ssim)
        history['w_physics'].append(current_weights['physics'])
        history['w_ssim'].append(current_weights['ssim'])
        history['w_mse'].append(current_weights['mse'])

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch + 1:4d}/{config.epochs} | "
            f"Train loss: {avg_train_loss:.6f} | Val loss: {avg_val_loss:.6f} | "
            f"Train MSE: {avg_train_mse:.6f} | Val MSE: {avg_val_mse:.6f} | "
            f"Train SSIM: {avg_train_ssim:.4f} | Val SSIM: {avg_val_ssim:.4f} | "
            f"score: {selection_score:.6f} | "
            f"w_phy: {current_weights['physics']:.4f} | "
            f"w_ssim: {current_weights['ssim']:.4f} | "
            f"w_mse: {current_weights['mse']:.4f} | lr: {current_lr:.3e}"
        )

        checkpoint_payload = {
            'training_version': TRAINING_VERSION,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss_balancer_state_dict': loss_balancer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_selection_score': best_selection_score,
            'selection_score': selection_score,
            'val_mse': avg_val_mse,
            'val_ssim': avg_val_ssim,
            'psf_info': psf_info,
            'adaptive_weights': current_weights,
            'balance_info': balance_info,
        }

        if selection_score < best_selection_score:
            best_selection_score = selection_score
            checkpoint_payload['best_selection_score'] = best_selection_score
            best_model_path = os.path.join(
                config.checkpoint_dir, 'best_model.pth'
            )
            torch.save(checkpoint_payload, best_model_path)
            print(
                f"  -> Saved best model to: {best_model_path} "
                f"(fixed score={best_selection_score:.6f}, "
                f"valMSE={avg_val_mse:.6f}, valSSIM={avg_val_ssim:.4f})"
            )

        if (epoch + 1) % 5 == 0:
            checkpoint_payload['best_selection_score'] = best_selection_score
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f'checkpoint_epoch_{epoch + 1}.pth'
            )
            torch.save(checkpoint_payload, checkpoint_path)
            with open(history_path, 'wb') as f:
                pickle.dump(history, f)
            print(f"  -> Saved checkpoint to: {checkpoint_path}")
            print(f"  -> Saved training history to: {history_path}")

    print("\n=== Saving training results ===")
    final_model_path = os.path.join(config.checkpoint_dir, 'final_model.pth')
    torch.save({
        'training_version': TRAINING_VERSION,
        'epoch': config.epochs - 1,
        'model_state_dict': model.state_dict(),
        'loss_balancer_state_dict': loss_balancer.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_selection_score': best_selection_score,
        'psf_info': psf_info,
        'adaptive_weights': loss_balancer.current_weights(),
    }, final_model_path)
    print(f"Final model saved to: {final_model_path}")

    with open(history_path, 'wb') as f:
        pickle.dump(history, f)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='train loss')
    plt.plot(history['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['downsampled_mse'], label='train MSE')
    plt.plot(history['val_downsampled_mse'], label='val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(alpha=0.3)

    loss_curve_path = os.path.join(config.output_dir, 'loss_mse_curve.png')
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(history['w_physics'], label='physics')
    plt.plot(history['w_ssim'], label='ssim')
    plt.plot(history['w_mse'], label='mse')
    plt.xlabel('Epoch')
    plt.ylabel('Bounded adaptive weight')
    plt.title('Bounded gradient-balanced loss weights')
    plt.legend()
    plt.grid(alpha=0.3)
    weight_curve_path = os.path.join(
        config.output_dir, 'adaptive_loss_weights.png'
    )
    plt.tight_layout()
    plt.savefig(weight_curve_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Loss/MSE curves saved to: {loss_curve_path}")
    print(f"Adaptive weight curves saved to: {weight_curve_path}")
    print("\n=== V2 adaptive PhySR training completed ===")
    print(f"All outputs saved under: {config.output_dir}")


if __name__ == '__main__':
    main()
