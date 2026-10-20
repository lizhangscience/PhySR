# PhySR: Physics-Informed Neural Network for Super-Resolution Reconstruction in Radio Synthesis Imaging

PhySR uses a U-Net encoder–decoder architecture with a **Multiscale Feature Residual Module (MFRM)** and a **cascaded upsampling module** for **2×, 4×, and 8×** super-resolution reconstruction of radio astronomical images. During training, an image-domain physical-consistency constraint incorporates the primary beam (PB) and point spread function (PSF).

This repository provides the model implementation, **public test datasets, one pretrained checkpoint for each scale, and the corresponding evaluation results**. Training and validation datasets are not included. The pretrained models can be evaluated without retraining.

## 1. Repository Structure

```text
PhySR_Test/
├── code/                # Training, testing, and evaluation scripts
├── configs/             # Scale-specific configuration files
├── models/              # Network and physics-related modules
├── Data/
│   ├── SKA_x2/
│   ├── SKA_x4/
│   └── SKA_x8/
└── results/
    ├── SKA_x2/
    ├── SKA_x4/
    └── SKA_x8/
```

Each `Data/SKA_x*/` directory contains:

- `test/blur/`: low-resolution dirty images (`.npy`)
- `test/sharp/`: corresponding low-resolution clean reference images (`.npy`)
- `psf.npy` and `beam.npy`: PSF and primary-beam data

Files in `test/blur/` and `test/sharp/` must have matching filenames. For each scale, `results/SKA_x*/checkpoints/` contains the pretrained checkpoint, and `results/SKA_x*/result/` contains the provided evaluation results.

| Super-resolution scale | Input image size | Output image size |
|:----------------------:|:----------------:|:-----------------:|
| **2×** | 128 × 128 | 256 × 256 |
| **4×** | 64 × 64 | 256 × 256 |
| **8×** | 32 × 32 | 256 × 256 |

## 2. Requirements

The code uses **Python 3.8**, PyTorch, NumPy, and scikit-image. Install the main dependencies with:

```bash
pip install torch numpy scikit-image
```

Install a PyTorch build compatible with your hardware and CUDA environment, if applicable.

## 3. Testing with Pretrained Checkpoints

Run the following commands **from the repository root**. Before testing, check the corresponding `configs/SKA_x*_test_config.py` file and set `data_root`, `weight_path`, and `result_dir` to the correct dataset, checkpoint, and output locations.

**2× super-resolution**

```bash
python code/SKA_x2_test.py
```

**4× super-resolution**

```bash
python code/SKA_x4_test.py
```

**8× super-resolution**

```bash
python code/SKA_x8_test.py
```

Each script loads the test dataset and pretrained checkpoint for its scale, performs image reconstruction, and calculates evaluation metrics. **No model training is required.** Keep the PSF processing mode consistent with the checkpoint used.

## 4. Evaluation Outputs

Results are saved to the directory specified by `result_dir` in the corresponding test configuration.

| File or directory | Description |
|:------------------|:------------|
| `evaluation_summary.txt` | Test-set mean PSNR, SSIM, RMSE, and physical-consistency metrics |
| `metrics.txt` | Evaluation metrics for individual test images |
| `sr/` | Super-resolved reconstructed images |
| `sim_dirty/` | Dirty images simulated using the image-domain forward operator |

**Image-domain metrics** (PSNR, SSIM, and RMSE) are computed by downsampling each reconstructed image to the input resolution and comparing it with the corresponding clean reference image. **Physical-consistency metrics** compare the observed dirty image with the simulated dirty image.
