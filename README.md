# PhySR: Physics-Informed Neural Network for Super-Resolution Reconstruction in Radio Synthesis Imaging

PhySR uses a U-Net encoder–decoder architecture with a **Multiscale Feature Residual Module (MFRM)** and a **cascaded upsampling module** for **2×, 4×, and 8×** super-resolution reconstruction of radio astronomical images. During training, an image-domain physical-consistency constraint incorporates the primary beam (PB) and point spread function (PSF).

This repository provides the **PhySR implementation and public test datasets distributed as ZIP archives**. Training and validation datasets are not included. Testing requires a compatible pretrained checkpoint, and no retraining is needed when such a checkpoint is available.

## 1. Repository Structure

```text
PhySR/
├── README.md
├── code/                 # Training, testing, and evaluation scripts
├── configs/              # Scale-specific configuration files
├── models/               # Network and physics-related modules
├── Data/                 # Extracted test datasets
│   ├── SKA_x2/
│   ├── SKA_x4/
│   └── SKA_x8/
└── results/              # Checkpoints and evaluation outputs
```

## 2. Test Datasets

The public test datasets are provided as **ZIP archives**. Download the archives from the repository or its Releases page and extract them into the `Data/` directory.

| Dataset | Extracted Directory | Input Resolution | Output Resolution |
|:---:|:---|:---:|:---:|
| **2×** | `Data/SKA_x2/` | 128 × 128 | 256 × 256 |
| **4×** | `Data/SKA_x4/` | 64 × 64 | 256 × 256 |
| **8×** | `Data/SKA_x8/` | 32 × 32 | 256 × 256 |

Each extracted dataset should have the following structure:

```text
Data/SKA_x*/
├── test/
│   ├── blur/             # Low-resolution dirty images (.npy)
│   └── sharp/            # Corresponding clean reference images (.npy)
├── psf.npy
└── beam.npy
```

Files in `test/blur/` and `test/sharp/` must have matching filenames. The `psf.npy` and `beam.npy` files provide the PSF and primary-beam data required for physical-consistency evaluation.

## 3. Requirements

The code uses **Python 3.8**, PyTorch, NumPy, and scikit-image. Install the main dependencies with:

```bash
pip install torch numpy scikit-image
```

Use a PyTorch version compatible with your hardware and CUDA environment.

## 4. Testing

Run the following commands **from the repository root**. Before testing, configure `data_root`, `weight_path`, and `result_dir` in the corresponding `configs/SKA_x*_test_config.py` file to specify the extracted test dataset, pretrained checkpoint, and output directory.

### 2× Super-Resolution

```bash
python code/SKA_x2_test.py
```

### 4× Super-Resolution

```bash
python code/SKA_x4_test.py
```

### 8× Super-Resolution

```bash
python code/SKA_x8_test.py
```

Each script loads the corresponding test dataset and pretrained checkpoint, performs image reconstruction, and calculates evaluation metrics. **No retraining is required.** The PSF processing mode must remain consistent with the checkpoint configuration.

## 5. Evaluation Outputs

Evaluation results are saved to the directory specified by `result_dir` in the corresponding test configuration.

| File or Directory | Description |
|:---|:---|
| `evaluation_summary.txt` | Mean PSNR, SSIM, RMSE, and physical-consistency metrics |
| `metrics.txt` | Evaluation metrics for individual test images |
| `sr/` | Super-resolved reconstructed images |
| `sim_dirty/` | Dirty images generated using the image-domain forward operator |

**Image-domain metrics:** PSNR, SSIM, and RMSE are calculated by downsampling each reconstructed image to the input resolution and comparing it with the corresponding clean reference image.

**Physical-consistency metrics:** Phys_PSNR and Phys_SSIM measure the similarity between the observed dirty image and the simulated dirty image obtained through the image-domain forward operator.

**Note on observation-domain metrics:** The physical-consistency PSNR and SSIM calculated by the test scripts are **not equivalent to the observation-domain metrics reported in the paper**. The paper uses observation-domain RMSE, mean absolute error (MAE), relative residual error (RRE), and residual standard deviation (σ_R) to evaluate differences between simulated observations and the original dirty images. These metrics have different definitions and should not be used interchangeably with Phys_PSNR and Phys_SSIM.
