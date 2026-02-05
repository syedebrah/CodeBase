# Protein Image Analysis: Compression & Super-Resolution

This project implements a deep learning pipeline for protein image analysis, featuring two core components:
1.  **Image Compression**: A Balanced Autoencoder (AE) to efficiently represent images in a lower-dimensional latent space.
2.  **Super-Resolution**: An ESRGAN-based model to upscale low-resolution images by 8x, restoring fine structural details.

## Project Structure

-   `AE_Model.ipynb`: Jupyter notebook containing the **Balanced Autoencoder** implementation.
-   `ESRGAN_MODEL_8X.ipynb`: Jupyter notebook containing the **8x Super-Resolution GAN** implementation.
-   `checkpoints/`: Directory where trained model weights are saved.
-   `results_balanced_ae/`: Directory for AE reconstruction visualizations and logs.
-   `README.md`: Project documentation.

## Environment Setup

The code is designed to run in a Python environment (e.g., Kaggle, Colab, or local GPU setup).

**Required Packages:**
-   `torch` (PyTorch)
-   `torchvision`
-   `torchmetrics` (for SSIM, PSNR, LPIPS metrics)
-   `PIL` (Pillow)
-   `numpy`
-   `pandas`
-   `matplotlib`

**Note:** GPU acceleration (CUDA) is highly recommended for training these models.

## 1. Autoencoder Model (`AE_Model.ipynb`)

This notebook trains a **Balanced Autoencoder** designed to compress 512x512 grayscale protein images into a compact representation while preserving structural integrity.

### Model Architecture
-   **Encoder**: A 3-layer Convolutional Neural Network that downsamples the input image. It reduces the channel dimensions to a bottleneck of 16 channels at 64x64 resolution.
-   **Decoder**: A symmetric 3-layer Transposed Convolutional Network that reconstructs the 512x512 image from the latent representation.
-   **Latent Space**: The model learns to compress the input into a 64x64x16 representation.

### Training Details
-   **Input**: 512x512 Grayscale Images.
-   **Loss Function**: `HybridLoss` combining:
    -   **Huber Loss (Pixel)**: 20% weight. Robust to outliers.
    -   **Structural Similarity Loss (1 - SSIM)**: 80% weight. Prioritizes structural fidelity over perfect pixel matching.
-   **Optimization**: Adam optimizer with Weight Decay to prevent overfitting.
-   **Features**:
    -   Early Stopping to prevent overtraining.
    -   visualisation of original vs. reconstructed images with PSNR/SSIM metrics.

## 2. Super-Resolution Model (`ESRGAN_MODEL_8X.ipynb`)

This notebook implements an **8x Super-Resolution Generative Adversarial Network (ESRGAN)** to upscale low-resolution (64x64) images back to high-resolution (512x512) with high perceptual quality.

### Model Architecture
-   **Generator (RRDBNet)**: Uses Residual-in-Residual Dense Blocks (RRDB) for deep feature extraction. Upsamples features using nearest-neighbor interpolation followed by convolution layers.
-   **Discriminator**: A VGG-style discriminator trained to distinguish between real high-res images and generated super-res images.

### Training Details
-   **Input**: 64x64 Low-Resolution Images (generated via Lanczos downsampling).
-   **Ground Truth**: 512x512 High-Resolution Images.
-   **Loss Function**: A comprehensive weighted sum of:
    -   **L1 Loss (Pixel)**: 1% weight. Low frequency correctness.
    -   **VGG Perceptual Loss**: 100% weight. High-level feature consistency.
    -   **Adversarial Loss (RaGAN)**: 0.5% weight. Realistic texturing.
    -   **Structural Loss (1 - SSIM)**: 50% weight. Structural integrity.
-   **Optimization**: AdamW optimizer with Automatic Mixed Precision (AMP) for faster training.
-   **Features**:
    -   Tracks PSNR, SSIM, and LPIPS (Learned Perceptual Image Patch Similarity) metrics during training.
    -   Saves checkpoints periodically.

## Usage

1.  **Data Preparation**: Ensure your dataset is accessible. Update the `DATA_PATH` in the `Config` class of both notebooks to point to your image directory.
    ```python
    class Config:
        DATA_PATH = "/path/to/your/dataset"
        ...
    ```
2.  **Training**: Run the cells in each notebook sequentially.
    -   Run `AE_Model.ipynb` to train the compression model.
    -   Run `ESRGAN_MODEL_8X.ipynb` to train the upscaling model.
3.  **Visualization**: Both notebooks automatically generate and save visualization figures comparing inputs and outputs in their respective results folders.
