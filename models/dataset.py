# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from skimage.transform import resize

class SuperResolutionDataset(Dataset):
    def __init__(self, input_frames, lr_clean_frames):
        self.input_frames = input_frames
        self.lr_clean_frames = lr_clean_frames

    def __len__(self):
        return len(self.input_frames)

    def __getitem__(self, idx):
        input_img = torch.from_numpy(self.input_frames[idx]).float()
        clean_img = torch.from_numpy(self.lr_clean_frames[idx]).float()
        return input_img, clean_img

class DataLoaderPyTorch:
    def __init__(self, config):
        self.config = config

    def load_npy_image(self, path):
        try:
            img_array = np.load(path).squeeze().astype(np.float32)

            if img_array.ndim != 2:
                raise ValueError(f"Expected a 2D NPY array; got {img_array.ndim} dimensions: {path}")

            if img_array.shape != (self.config.im_dim_input, self.config.im_dim_input):
                img_array = resize(
                    img_array,
                    (self.config.im_dim_input, self.config.im_dim_input),
                    order=1,
                    mode='edge',
                    anti_aliasing=True,
                    preserve_range=True
                ).astype(np.float32)

            return img_array

        except Exception as e:
            print(f"Failed to load NPY file {path}: {e}")
            return None

    def _numeric_sort_key(self, filename):
        stem = os.path.splitext(filename)[0]
        try:
            return int(stem)
        except ValueError:
            return stem

    def load_images_from_split(self, split_dir, split_name):
        blur_dir = os.path.join(split_dir, 'blur')
        sharp_dir = os.path.join(split_dir, 'sharp')

        if not os.path.exists(blur_dir):
            raise FileNotFoundError(f"Blur directory for {split_name} does not exist: {blur_dir}")

        if not os.path.exists(sharp_dir):
            raise FileNotFoundError(f"Sharp directory for {split_name} does not exist: {sharp_dir}")

        blur_files = sorted(
            [f for f in os.listdir(blur_dir) if f.lower().endswith('.npy')],
            key=self._numeric_sort_key
        )

        sharp_files = sorted(
            [f for f in os.listdir(sharp_dir) if f.lower().endswith('.npy')],
            key=self._numeric_sort_key
        )

        if len(blur_files) != len(sharp_files):
            raise ValueError(
                f"Mismatch in the number of blur and sharp files for {split_name}: "
                f"{len(blur_files)} vs {len(sharp_files)}"
            )

        if blur_files != sharp_files:
            raise ValueError(f"Blur and sharp filenames do not match for {split_name}; files must be paired.")

        actual_num_im = min(self.config.num_im, len(blur_files))

        print(
            f"Loaded NPY files for {split_name}: {actual_num_im}"
            f" (each resized to {self.config.im_dim_input}x{self.config.im_dim_input})"
        )

        input_frames = np.zeros(
            [actual_num_im, 1, self.config.im_dim_input, self.config.im_dim_input],
            dtype=np.float32
        )

        lr_clean_frames = np.zeros(
            [actual_num_im, 1, self.config.im_dim_input, self.config.im_dim_input],
            dtype=np.float32
        )

        for idx in range(actual_num_im):
            blur_path = os.path.join(blur_dir, blur_files[idx])
            sharp_path = os.path.join(sharp_dir, sharp_files[idx])

            input_temp = self.load_npy_image(blur_path)
            clean_temp = self.load_npy_image(sharp_path)

            if input_temp is not None:
                input_frames[idx, 0, :, :] = input_temp

            if clean_temp is not None:
                lr_clean_frames[idx, 0, :, :] = clean_temp

        return input_frames, lr_clean_frames

    def create_dataloader_from_split(self, split_dir, split_name, shuffle):
        input_frames, lr_clean_frames = self.load_images_from_split(split_dir, split_name)
        dataset = SuperResolutionDataset(input_frames, lr_clean_frames)

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=4
        )

        return loader, len(dataset)
