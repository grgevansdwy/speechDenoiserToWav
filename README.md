## Dataset Preparation (Colab)

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
