# -*- coding: utf-8 -*-
import os


class Config:
    def __init__(self):
        # Must match the corresponding training configuration.
        self.super_resolution_scale = 8
        self.im_dim_input = 32
        self.im_dim_gt = self.im_dim_input * self.super_resolution_scale

        self.num_im = 300
        self.random_seed = 42
        self.batch_size = 8

        self.data_name = 'SKA_x8'
        self.dir_name = 'D:\pyCham\PhySR\Data'
        self.data_root = os.path.join(self.dir_name, self.data_name)

        self.output_dir = (
            f'D:\pyCham\PhySR\KAIYUAN\PhySR_Test\results\SKA_x8'
        )
        self.weight_path = os.path.join(
            self.output_dir, 'checkpoints', 'best_model.pth'
        )
        self.result_dir = os.path.join(self.output_dir, 'result')

        self.psf_path = os.path.join(self.data_root, 'psf.npy')
        self.beam_path = os.path.join(self.data_root, 'beam.npy')
        self.test_dir = os.path.join(self.data_root, 'test')

        # Must match the training configuration; the checkpoint is also validated.
        self.psf_mode = 'auto_calibrate'
