# -*- coding: utf-8 -*-
import os


class Config:
    def __init__(self):

        self.super_resolution_scale = 4
        self.im_dim_input = 64
        self.im_dim_gt = self.im_dim_input * self.super_resolution_scale

        self.num_im = 2400
        self.bg_lvl = 0
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1
        self.random_seed = 42

        # Dataset paths
        self.data_name = 'SKA_x4'
        self.dir_name = 'D:\pyCham\PhySR\Data'
        self.data_root = os.path.join(self.dir_name, self.data_name)

        # V2 requires a new output directory to avoid resuming old 1/raw_loss experiments.
        self.output_dir = (
            f'D:\pyCham\PhySR\KAIYUAN\PhySR_Test\results\SKA_x4'

        )

        self.psf_path = os.path.join(self.data_root, 'psf.npy')
        self.beam_path = os.path.join(self.data_root, 'beam.npy')
        self.train_dir = os.path.join(self.data_root, 'train')
        self.val_dir = os.path.join(self.data_root, 'val')
        self.test_dir = os.path.join(self.data_root, 'test')

        # Preserve the original training hyperparameters.
        self.batch_size = 8
        self.epochs = 1500
        self.learning_rate = 3e-6
        self.weight_decay = 1e-5
        self.clipnorm = 1.0

        # ============================================================
        # PSF handling mode
        # ------------------------------------------------------------
        # 'auto_calibrate' -> raw PSF with automatic amplitude calibration
        #  'l1'             -> PyTorch L1 normalization used in the original paper
        # Also supported: 'raw'
        # ============================================================
        self.psf_mode = 'auto_calibrate'
        self.psf_calibration_batches = 0
        self.psf_calibration_positive_only = True
        self.psf_calibration_eps = 1e-12


        self.adaptive_loss_weighting = True
        self.loss_balance_init_batches = 10
        self.loss_balance_update_interval = 10
        self.loss_balance_ema = 0.90
        self.loss_balance_alpha = 0.50
        self.loss_balance_min_weight = 1e-3
        self.loss_balance_max_weight = 3.0
        self.loss_balance_eps = 1e-12

        # Keep the original lightweight regularization terms unchanged.
        self.tv_weight = 1e-4
        self.artifact_weight = 5e-4

        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
