# Image Generation Using Denoising Diffusion Probabilistic Models

A deep learning and computer vision project that generates
**32 × 32 colour images** from random Gaussian noise using
Denoising Diffusion Probabilistic Models (DDPMs).

The project implements the diffusion process using PyTorch and
compares DDPM variants with **VAE, DCGAN, and autoencoder baselines**.

## 🔄 Project Pipeline

```text
CIFAR-10 Images
      ↓
Image Preprocessing and Normalization
      ↓
Training and Validation Split
      ↓
Forward Diffusion: Add Gaussian Noise
      ↓
Timestep-Conditioned U-Net
      ↓
Noise Prediction and MSE Training
      ↓
Reverse Diffusion: Gradually Remove Noise
      ↓
Generated Images
      ↓
FID, Inception Score, and Visual Evaluation
      ↓
Comparison with AE, VAE, and DCGAN
```

## 📊 Dataset

This project uses **CIFAR-10**, which contains 60,000 colour images
across 10 classes. Each image is **32 × 32 pixels**.

| Split | Images | Purpose |
|---|---:|---|
| Training | 45,000 | Model training |
| Validation | 5,000 | Evaluation and comparison |
| Original test split | 10,000 | Reserved for separate final evaluation |

The models generate images **unconditionally**. A user cannot select
a specific class such as “cat” or “airplane.”

## 🧹 Data Preparation

- Converted images to PyTorch tensors.
- Normalized pixel values to `[-1, 1]`.
- Applied random horizontal flips to training images.
- Created shuffled batches using a DataLoader.
- Saved training and validation indices for consistent comparisons.

## 🧠 DDPM Architecture

The denoising network is a U-Net containing:

- Sinusoidal timestep embeddings.
- Time-conditioned residual blocks.
- Group normalization and SiLU activation.
- Encoder-decoder layers with skip connections.
- Self-attention at selected spatial resolutions.
- Convolution-based downsampling and interpolation-based upsampling.

The model predicts the added noise and is trained using
**mean squared error**.

## 🧪 Experiments

| Experiment | Schedule | Attention | Purpose |
|---|---|---|---|
| Cosine DDPM | Cosine | Yes | Main diffusion experiment |
| Linear DDPM | Linear | Yes | Noise-schedule comparison |
| Linear DDPM baseline | Linear | No | Attention ablation |
| VAE | — | — | Generative baseline |
| DCGAN | — | — | Adversarial generative baseline |
| AE | — | — | Reconstruction baseline |

## 📈 Results

Generative models were evaluated using **5,000 real validation images**
and **5,000 generated images**.

| Model | Epoch | FID ↓ | Inception Score ↑ |
|---|---:|---:|---:|
| **Linear DDPM + attention** | 200 | **39.0745** | 6.5577 |
| DCGAN | 200 | 39.6168 | 6.6133 |
| Cosine DDPM + attention | 200 | 39.9546 | **6.8209** |
| Linear DDPM without attention | 200 | 52.3218 | 5.6041 |
| VAE | 200 | 132.6878 | 3.4810 |

- **Linear DDPM + attention** achieved the lowest FID.
- **Cosine DDPM + attention** achieved the highest Inception Score.
- DCGAN was competitive with both attention-enabled DDPMs.
- No model was best on both metrics.

The results are from single runs. Equal epochs do not imply equal
computational cost.

### Autoencoder Result

The AE achieved a **validation reconstruction MSE of 0.01754 at
epoch 50**, measured on pixels normalized to `[-1, 1]`.

AE reconstruction is reported separately because it is a different
task from unconditional image generation.

## 🔍 Attention Ablation

With the linear schedule:

| Model | FID ↓ | Inception Score ↑ |
|---|---:|---:|
| With attention | 39.07 | 6.56 |
| Without attention | 52.32 | 5.60 |

Attention reduced FID by approximately **25.3%** in this experiment.
Repeated runs would be needed to establish whether the improvement is
consistent.

## 🖼️ Generated Images

Add your saved qualitative comparison here:

```markdown
![Generated image comparison](figures/final_qualitative_comparison.png)
```

Add your saved training-loss plot here:

```markdown
![Training loss comparison](figures/final_training_loss_comparison.png)
```

The inspected samples showed varied colours and object-like structures.
Some images remained blurry or contained distorted shapes.

## 📥 Trained Model Files

Trained checkpoints and evaluation files can be stored on Google Drive.

**Replace the placeholder below with your actual shareable link:**

[Download Model Files – Google Drive](YOUR_GOOGLE_DRIVE_LINK)

Example checkpoint files:

```text
ddpm_cosine_epoch_200.pt
ddpm_linear_epoch_200.pt
linear_no_attention/
    ddpm_linear_no_attention_epoch_200.pt
```

The checkpoints contain model weights and configuration. They may also
contain the optimizer state, noise schedule, split indices, and training
history.

## 📁 Project Structure

Update the filenames below to match your actual repository:

```text
DDPM-Image-Generation/
├── notebooks/
│   ├── data_preparation_eda.ipynb
│   ├── ddpm_cosine.ipynb
│   ├── ddpm_linear.ipynb
│   ├── ddpm_no_attention.ipynb
│   ├── baseline_models.ipynb
│   └── evaluation.ipynb
├── figures/
│   ├── final_qualitative_comparison.png
│   └── final_training_loss_comparison.png
├── requirements.txt
└── README.md
```

## 🚀 Run the Project

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd DDPM-Image-Generation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install the main packages:

```bash
pip install torch torchvision numpy pandas matplotlib tqdm torch-fidelity
```

### 3. Prepare Data and Checkpoints

- Download CIFAR-10 through the notebook.
- Download the trained checkpoints if you want to generate images
  without retraining.
- Update data and checkpoint paths for your environment.

### 4. Run the Notebooks

Run the model-definition cells before loading checkpoints. Then run
the sampling or evaluation cells.

**A checkpoint contains saved weights; it does not automatically
provide the Python model-class definitions.**

## 🧰 Technologies Used

- Python
- PyTorch and torchvision
- NumPy and pandas
- Matplotlib
- torch-fidelity
- Google Colab

## ⚠️ Limitations

- Images are limited to the 32 × 32 CIFAR-10 resolution.
- Generation is unconditional.
- Results are from individual training runs.
- Equal epochs do not necessarily mean equal computational cost.
- FID depends on the evaluation protocol and sample count.
- The validation split was used for model comparisons; the original
  test split should be used for a separate final evaluation.
- The current FID is above the aspirational range in the project brief.
  Published benchmark results used a different research setup.

## 🔮 Future Work

- Evaluate EMA weights if the EMA experiment is completed.
- Explore larger U-Net architectures and longer training.
- Compare repeated training seeds.
- Evaluate the selected model on the untouched test split.
- Explore class conditioning and DDIM sampling.
- Measure computational cost alongside image quality.

