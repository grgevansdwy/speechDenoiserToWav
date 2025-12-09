## Dataset Preparation

We use the **Denoising Audio Collection** Kaggle dataset, which provides parallel
clean and noisy speech recordings. The `dataset_prep.ipynb` notebook:

- Downloads the dataset via `kagglehub`
- Recursively finds matching clean/noisy `.wav` files in:
  - `clean_trainset_28spk_wav` / `noisy_trainset_28spk_wav` (train speakers)
  - `clean_testset_wav` / `noisy_testset_wav` (test set)
- Converts all audio to **mono, 16 kHz**, and normalizes to `[-1, 1]`
- Splits the data into **train / val / test**
- Segments audio into fixed-length chunks (default: **2 seconds** → 32,000 samples)
- Saves processed files under:

- data_16k/
- train/clean/.wav
- train/noisy/.wav
- val/clean/.wav
- val/noisy/.wav
- test/clean/.wav
- test/noisy/.wav
- metadata.csv

### PyTorch Dataset Interface

To make training and evaluation easier, we provide a `DenoisingDataset` class
that returns aligned `(noisy, clean)` waveform pairs as PyTorch tensors:

```python
from dataset import DenoisingDataset

train_ds = DenoisingDataset("data_16k/metadata.csv", split="train")
noisy, clean = train_ds[0]  # shapes: [1, T], 16 kHz, normalized to [-1, 1]
