# PhySR: Physics-Informed Neural Network for Super-Resolution Reconstruction in Radio Synthesis Imaging

PhySR uses a U-Net encoder–decoder architecture with a **Multiscale Feature Residual Module (MFRM)** and a **cascaded upsampling module** for **2×, 4×, and 8×** super-resolution reconstruction of radio astronomical images. During training, an image-domain physical-consistency constraint incorporates the primary beam (PB) and point spread function (PSF).

This repository provides the **PhySR implementation and public test datasets distributed as ZIP archives**. Training and validation datasets are not included. Testing requires a compatible pretrained checkpoint; no retraining is needed when such a checkpoint is available.

## 1. Repository Structure

```text
PhySR/
├── README.md
├── code/                 # Training, testing, and evaluation scripts
├── configs/              # Scale-specific configuration files
├── models/               # Network and physics-related modules
├── Data/                 # Created after extracting the test archives
│   ├── SKA_x2/
│   ├── SKA_x4/
│   └── SKA_x8/
└── results/              # Checkpoints (if provided) and evaluation outputs
```

## 2. Test Datasets

The public test datasets are distributed as **compressed ZIP files** rather than individual `.npy` files. Download the available archives from the repository or its **Releases** page, then extract them into the `Data/` directory.

| Archive | Extracted directory | Input size | Output size |
|:---|:---|:---:|:---:|
| `SKA_x2.zip` | `Data/SKA_x2/` | 128 × 128 | 256 × 256 |
| `SKA_x4.zip` | `Data/SKA_x4/` | 64 × 64 | 256 × 256 |
| `SKA_x8.zip` | `Data/SKA_x8/` | 32 × 32 | 256 × 256 |

*The archive names above are examples; use the actual filenames shown on the download page.*

After extraction, each dataset should have this structure:

```text
Data/SKA_x*/
├── test/
│   ├── blur/             # Low-resolution dirty images (.npy)
│   └── sharp/            # Matching clean reference images (.npy)
├── psf.npy
└── beam.npy
```

The files in `test/blur/` and `test/sharp/` must have matching filenames. Ensure that extraction does not create an extra nested folder, such as `Data/SKA_x2/SKA_x2/`.

## 3. Requirements

The code uses **Python 3.8**, PyTorch, NumPy, and scikit-image. Install the main dependencies with:

```bash
pip install torch numpy scikit-image
```

Use a PyTorch build compatible with your hardware and CUDA environment.

## 4. Testing

Run the commands below **from the repository root**. Before testing, update the corresponding `configs/SKA_x*_test_config.py` file to point `data_root`, `weight_path`, and `result_dir` to the extracted test dataset, a compatible pretrained checkpoint, and the desired output directory.

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

Each script performs image reconstruction and calculates evaluation metrics. **No retraining is required** when using a compatible pretrained checkpoint. Keep the PSF processing mode consistent with the checkpoint.

## 5. Evaluation Outputs

Evaluation results are generated in the directory specified by `result_dir` in the test configuration.

| File or directory | Description |
|:---|:---|
| `evaluation_summary.txt` | Mean PSNR, SSIM, RMSE, and physical-consistency metrics |
| `metrics.txt` | Per-image evaluation metrics |
| `sr/` | Super-resolved reconstructed images |
| `sim_dirty/` | Dirty images simulated by the image-domain forward operator |

**Image-domain metrics** (PSNR, SSIM, and RMSE) compare each reconstructed image, downsampled to the input resolution, with its corresponding clean reference image. **Physical-consistency metrics** compare the observed and simulated dirty images.
