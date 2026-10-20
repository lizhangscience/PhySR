# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from device import device

class Evaluator:
    def __init__(self, config, psf, beam):
        self.config = config
        self.super_resolution_scale = config.super_resolution_scale

        self.psf = psf.to(device=device, dtype=torch.float32)
        self.beam = beam.to(device=device, dtype=torch.float32)

        # PSF tensor in the training code: [1, 1, 64, 64]
        if self.psf.ndim != 4 or self.psf.shape[0] != 1 or self.psf.shape[1] != 1:
            raise ValueError(
                f"Expected PSF shape [1,1,H,W]; got {tuple(self.psf.shape)}"
            )

        expected_lr_hw = (config.im_dim_input, config.im_dim_input)
        if tuple(self.psf.shape[-2:]) != expected_lr_hw:
            raise ValueError(
                f"Expected LR PSF dimensions {expected_lr_hw}; got {tuple(self.psf.shape[-2:])}"
            )

        # Primary-beam tensor in the training code: [64, 64]
        if self.beam.ndim != 2:
            raise ValueError(
                f"Expected primary-beam shape [H,W]; got {tuple(self.beam.shape)}"
            )

        if tuple(self.beam.shape) != expected_lr_hw:
            raise ValueError(
                f"Expected LR primary-beam dimensions {expected_lr_hw}; got {tuple(self.beam.shape)}"
            )

        self.results_dir = config.result_dir

        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'sharp'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'blur'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'sr'), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, 'sim_dirty'), exist_ok=True)

    def _controlled_downsample(self, img, target_shape):
        """
        CPU/NumPy version, used only for computing the final image metrics.
        For integer scale factors, block averaging matches the scale mapping used in training.
        """
        img = np.asarray(img).squeeze().astype(np.float32)

        if img.shape == target_shape:
            return img.astype(np.float32)

        h, w = img.shape
        th, tw = target_shape

        if h % th == 0 and w % tw == 0:
            fh = h // th
            fw = w // tw
            img_reshaped = img.reshape(th, fh, tw, fw)
            return img_reshaped.mean(axis=(1, 3)).astype(np.float32)

        return resize(
            img,
            target_shape,
            order=1,
            mode='edge',
            anti_aliasing=True,
            preserve_range=True
        ).astype(np.float32)

    def calculate_metrics(self, gt, pred):
        gt = np.asarray(gt).squeeze().astype(np.float32)
        pred = np.asarray(pred).squeeze().astype(np.float32)

        if gt.shape != pred.shape:
            pred = self._controlled_downsample(pred, gt.shape)

        data_range = gt.max() - gt.min() if gt.max() != gt.min() else 1.0

        psnr_val = psnr(gt, pred, data_range=data_range)
        ssim_val = ssim(gt, pred, data_range=data_range)
        rmse_val = np.sqrt(np.mean((gt - pred) ** 2))

        return psnr_val, ssim_val, rmse_val

    def physical_forward_gpu(self, sr_batch):
        """
        Use the same physical forward-operator order as radio_physics_loss in training:

            SR(HR, 256x256)
            -> block-mean Downsample (64x64)
            -> Primary Beam (64x64)
            -> PSF correlation (64x64)
            -> simulated Dirty (64x64)

        Input:
            sr_batch: [B, 1, 256, 256]

        Output:
            simulated_dirty: [B, 1, 64, 64]
        """
        if sr_batch.ndim != 4:
            raise ValueError(
                f"sr_batch must be a 4D tensor [B,C,H,W]; got shape={tuple(sr_batch.shape)}"
            )

        expected_hr_hw = (self.config.im_dim_gt, self.config.im_dim_gt)
        if tuple(sr_batch.shape[-2:]) != expected_hr_hw:
            raise ValueError(
                f"Expected SR spatial dimensions {expected_hr_hw}; got {tuple(sr_batch.shape[-2:])}"
            )

        # ---------------------------------------------------------
        # 1) Match _dynamic_downsample in the training code:
        #    SR 256x256 -> block mean -> 64x64
        # ---------------------------------------------------------
        batch_size, channels, h, w = sr_batch.shape
        s = self.super_resolution_scale

        if channels != 1:
            raise ValueError(f"Only single-channel input is supported; got {channels} channels")

        if h % s != 0 or w % s != 0:
            raise ValueError(
                f"SR spatial dimensions {(h, w)} are not divisible by scale={s}"
            )

        sr_downsampled = sr_batch.reshape(
            batch_size,
            channels,
            h // s,
            s,
            w // s,
            s
        ).mean(dim=(3, 5))

        # ---------------------------------------------------------
        # 2) Apply the primary beam on the LR observation grid (as in training).
        # ---------------------------------------------------------
        beam_expanded = self.beam.unsqueeze(0).unsqueeze(0)
        y_beam = sr_downsampled * beam_expanded

        # ---------------------------------------------------------
        # 3) Apply PSF correlation on the LR grid (as in training).
        # ---------------------------------------------------------
        _, _, kh, kw = self.psf.shape

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
            self.psf,
            stride=1,
            padding=0
        )

        # Same numerical handling as in training.
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

        return simulated_dirty

    def _save_as_npy(self, img, path):
        if isinstance(img, torch.Tensor):
            img_npy = img.detach().squeeze().cpu().numpy()
        else:
            img_npy = np.asarray(img).squeeze()

        np.save(
            path,
            img_npy.astype(np.float32),
            allow_pickle=False
        )

    def evaluate_and_save_all(self, model, test_loader):
        """
        Evaluate the test set in a single pass:
        1. GPU model inference
        2. Batched physical forward operation on GPU
        3. Computation of image and physical-consistency metrics
        4. Saving sharp/blur/sr/sim_dirty arrays
        5. Saving per-image and average metrics
        """
        model.eval()

        metrics_txt = os.path.join(self.results_dir, 'metrics.txt')
        summary_path = os.path.join(self.results_dir, 'evaluation_summary.txt')

        with open(metrics_txt, 'w') as f:
            f.write("idx,PSNR,SSIM,RMSE,Phys_PSNR,Phys_SSIM\n")

        total_psnr = 0.0
        total_ssim = 0.0
        total_rmse = 0.0
        total_phys_psnr = 0.0
        total_phys_ssim = 0.0
        count = 0

        total_num = len(test_loader.dataset)

        with torch.no_grad():
            for batch_idx, (lr_batch, clean_batch) in enumerate(test_loader):
                lr_batch = lr_batch.to(device, non_blocking=True)
                clean_batch = clean_batch.to(device, non_blocking=True)

                # ---------------------------------------------------------
                # 1. Run inference on GPU.
                # ---------------------------------------------------------
                sr_pred = model(lr_batch)

                # ---------------------------------------------------------
                # 2. Apply the physical forward operator on GPU (batched).
                # ---------------------------------------------------------
                sim_dirty_batch = self.physical_forward_gpu(sr_pred)

                if sim_dirty_batch.shape != lr_batch.shape:
                    raise ValueError(
                        f"Simulated dirty shape {tuple(sim_dirty_batch.shape)} "
                        f"does not match observed dirty shape {tuple(lr_batch.shape)}"
                    )

                # ---------------------------------------------------------
                # 3. Transfer the batch to CPU once for scikit-image metrics and saving.
                # ---------------------------------------------------------
                lr_np_batch = lr_batch.detach().cpu().numpy()
                clean_np_batch = clean_batch.detach().cpu().numpy()
                sr_np_batch = sr_pred.detach().cpu().numpy()
                sim_dirty_np_batch = sim_dirty_batch.detach().cpu().numpy()

                for b in range(sr_np_batch.shape[0]):
                    global_idx = batch_idx * self.config.batch_size + b

                    lr_np = np.asarray(lr_np_batch[b]).squeeze().astype(np.float32)
                    clean_np = np.asarray(clean_np_batch[b]).squeeze().astype(np.float32)
                    sr_np = np.asarray(sr_np_batch[b]).squeeze().astype(np.float32)
                    sim_dirty_np = np.asarray(sim_dirty_np_batch[b]).squeeze().astype(np.float32)

                    # -----------------------------------------------------
                    # Image-domain metrics:
                    # Block-average SR 256 -> 64 and compare with clean 64.
                    # -----------------------------------------------------
                    sr_downsampled = self._controlled_downsample(
                        sr_np,
                        clean_np.shape
                    )

                    psnr_val, ssim_val, rmse_val = self.calculate_metrics(
                        clean_np,
                        sr_downsampled
                    )

                    # -----------------------------------------------------
                    # Observation-domain physical consistency:
                    # Compare simulated dirty 64 from the physical operator with observed dirty 64.
                    # -----------------------------------------------------
                    dirty_range = (
                        lr_np.max() - lr_np.min()
                        if lr_np.max() != lr_np.min()
                        else 1.0
                    )

                    phys_psnr = psnr(
                        lr_np,
                        sim_dirty_np,
                        data_range=dirty_range
                    )
                    phys_ssim = ssim(
                        lr_np,
                        sim_dirty_np,
                        data_range=dirty_range
                    )

                    # -----------------------------------------------------
                    # Save reconstruction outputs.
                    # -----------------------------------------------------
                    self._save_as_npy(
                        clean_np,
                        os.path.join(
                            self.results_dir,
                            'sharp',
                            f'sharp_{global_idx}.npy'
                        )
                    )

                    self._save_as_npy(
                        lr_np,
                        os.path.join(
                            self.results_dir,
                            'blur',
                            f'blur_{global_idx}.npy'
                        )
                    )

                    self._save_as_npy(
                        sr_np,
                        os.path.join(
                            self.results_dir,
                            'sr',
                            f'sr_{global_idx}.npy'
                        )
                    )

                    self._save_as_npy(
                        sim_dirty_np,
                        os.path.join(
                            self.results_dir,
                            'sim_dirty',
                            f'sim_dirty_{global_idx}.npy'
                        )
                    )

                    with open(metrics_txt, 'a') as f:
                        f.write(
                            f"{global_idx},"
                            f"{psnr_val:.4f},"
                            f"{ssim_val:.6f},"
                            f"{rmse_val:.8e},"
                            f"{phys_psnr:.4f},"
                            f"{phys_ssim:.6f}\n"
                        )

                    total_psnr += psnr_val
                    total_ssim += ssim_val
                    total_rmse += rmse_val
                    total_phys_psnr += phys_psnr
                    total_phys_ssim += phys_ssim
                    count += 1

                # Report evaluation progress.
                print(
                    f"[{min(count, total_num)}/{total_num}] "
                    f"batch {batch_idx + 1}/{len(test_loader)} completed",
                    flush=True
                )

        if count == 0:
            raise RuntimeError("The test set is empty; there are no samples to evaluate.")

        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_rmse = total_rmse / count
        avg_phys_psnr = total_phys_psnr / count
        avg_phys_ssim = total_phys_ssim / count

        with open(summary_path, 'w') as f:
            f.write(
                f"PhySR test-set average metrics"
                f" (super-resolution scale={self.super_resolution_scale}x):\n"
            )
            f.write(f"PSNR: {avg_psnr:.4f} dB\n")
            f.write(f"SSIM: {avg_ssim:.6f}\n")
            f.write(f"RMSE: {avg_rmse:.8e}\n")
            f.write(f"Physical-consistency PSNR: {avg_phys_psnr:.4f} dB\n")
            f.write(f"Physical-consistency SSIM: {avg_phys_ssim:.6f}\n")

        print("\n" + "=" * 70)
        print("Evaluation completed")
        print("=" * 70)
        print(f"Number of test samples: {count}")
        print(f"Test-set average PSNR: {avg_psnr:.4f} dB")
        print(f"Test-set average SSIM: {avg_ssim:.6f}")
        print(f"Test-set average RMSE: {avg_rmse:.8e}")
        print(f"Physical consistency - average PSNR: {avg_phys_psnr:.4f} dB")
        print(f"Physical consistency - average SSIM: {avg_phys_ssim:.6f}")
        print(f"Per-image metrics: {metrics_txt}")
        print(f"Evaluation summary: {summary_path}")
