# 🎧 Neural Audio Denoising with a 1D Conv U-Net for Whisper ASR

This repository implements a **machine learning–based audio denoiser** using a  
**1D Convolutional U-Net** to remove background noise from speech before feeding
the audio into the **Whisper-small Automatic Speech Recognition (ASR)** model.

The goal is to compare three pipelines:
1. **Pipeline A – Baseline (No Denoiser)**
2. **Pipeline B – Classical DSP Denoiser (noisereduce)**
3. **Pipeline C – Neural Denoiser (1D Conv U-Net → Whisper ASR)** ← *main contribution*

This project demonstrates how learned denoising can improve speech recognition
on noisy recordings, especially under challenging noise conditions.

---

# 📌 Table of Contents

- [Overview](#overview)
- [What is ASR?](#what-is-asr)
- [System Architecture](#system-architecture)
- [Pipeline A — Baseline](#pipeline-a--baseline-no-denoiser)
- [Pipeline B — DSP Denoiser](#pipeline-b--dsp-based-denoiser)
- [Pipeline C — Neural Denoiser (Main Contribution)](#pipeline-c--neural-denoiser-1d-conv-u-net--whisper-asr)
  - [1. Input Preparation](#1-input-preparation)
  - [2. 1D Conv U-Net Overview](#2-1d-conv-u-net-overview)
  - [3. Encoder (Downsampling Path)](#3-encoder-downsampling-path)
  - [4. Bottleneck](#4-bottleneck)
  - [5. Decoder (Upsampling Path)](#5-decoder-upsampling-path)
  - [6. Skip Connections](#6-skip-connections)
  - [7. Model Output](#7-model-output)
  - [8. Training the Denoiser](#8-training-the-denoiser)
- [Evaluation with Whisper-small](#evaluation-with-whisper-small)
- [Repository Structure](#repository-structure)
- [Results](#results)
- [Installation](#installation)
- [Credits](#credits)

---

# 🧠 Overview

Real-world speech often contains unwanted noise:
- cafeteria chatter  
- traffic  
- HVAC hum  
- keyboard clicking  
- street noise  
- microphone hiss  

While **Whisper-small** is robust, giving it *cleaner audio* improves
transcription accuracy.

This repo builds a **learned audio preprocessing module** (a 1D Conv U-Net)
trained to:

noisy_audio → denoised_audio → Whisper ASR → transcript


The 1D Conv U-Net operates directly on raw waveforms and learns to remove noise
in a data-driven way.

---

# 🔊 What is ASR?

**ASR (Automatic Speech Recognition)** is the process of converting **audio → text**.

In this project, the ASR backend is:

> **Whisper-small**, a pretrained Transformer-based speech recognition model by OpenAI.

Internally, Whisper performs:

1. **Audio preprocessing**  
   - Convert to mono  
   - Resample to 16 kHz  
   - Normalize amplitude  

2. **Feature extraction**  
   - Compute STFT  
   - Apply mel filterbank  
   - Compute log-mel spectrogram  

3. **Transformer encoder**  
   Produces embeddings representing the audio content.

4. **Transformer decoder**  
   Autoregressively generates text tokens.

Whisper is **not trained or modified** in this project.  
We only feed it different versions of audio and observe transcription accuracy.

---

# 🏗 System Architecture

noisy.wav →  Denoiser  → clean.wav → Whisper-small → text

Three versions of the denoiser are tested:


---

# 🅰 Pipeline A — Baseline (No Denoiser)


This provides the baseline transcription accuracy under noisy conditions.

---

# 🅱 Pipeline B — DSP Denoiser

Uses classical signal processing:

- spectral subtraction
- Wiener filtering
- noise profiling

Implemented using **noisereduce**:

noisy audio → classical DSP → denoised audio → Whisper-small


Pros:
- fast  
- interpretable  
- simple  

Cons:
- often removes speech detail  
- introduces musical noise  
- not adaptive to complex environments  

---

# 🅾 Pipeline C — Neural Denoiser (1D Conv U-Net → Whisper ASR)

This is the **main contribution** of the project.

The neural denoiser is a **learned model** trained on pairs of:
- **noisy speech (input)**
- **clean speech (target)**

It learns to produce a cleaned waveform that improves Whisper transcription.

---

## 1. Input Preparation

Before denoising, audio is standardized:

- Convert to **mono**
- Resample to **16,000 Hz**
- Normalize amplitude to `[-1, 1]`
- Pad/crop to fixed duration (optional)
- Convert to a PyTorch tensor of shape:

[batch_size, 1, time_samples]


This gives the U-Net a consistent waveform representation.

---

## 2. 1D Conv U-Net Overview

A **1D Conv U-Net** is a special type of encoder–decoder network:

Encoder: reduces time resolution, increases channels (features)
Bottleneck: compressed representation
Decoder: increases time resolution, reconstructs waveform
Skip connections: copy encoder features directly into decoder


Why U-Net?

- Encoder learns **what is speech vs noise**  
- Decoder reconstructs **clean waveform**  
- Skip connections preserve **fine speech detail**  
- Model learns **data-driven filtering** far beyond DSP rules

---

## 3. Encoder (Downsampling Path)

Each encoder stage:

1. **Conv1d layers** extract patterns (multiple filters → multiple feature channels)
2. **Downsampling** (via stride=2 Conv or MaxPool) halves the time dimension
3. **Channels increase** to store richer features

Example:

Input: [1, 16000]
Conv → [16, 16000]
Downsample → [16, 8000]
Conv → [32, 8000]
Downsample → [32, 4000]


Increasing channels allows the model to store **more representations** per time step
even though time resolution decreases.

---

## 4. Bottleneck

The most compressed layer:

Here the model has:

- wide contextual awareness  
- abstract features  
- ability to separate speech vs noise  

This representation drives the reconstruction.

---

## 5. Decoder (Upsampling Path)

Each decoder stage performs:

1. **Upsampling**
   - Double the time dimension (e.g., 4000 → 8000)
2. **Concatenation with saved encoder features** (skip connection)
3. **Conv1d layers** to merge and refine features
4. **Channel reduction**
   - e.g., from 96 channels → 32 channels

Example:
Upsample: [128, 4000] → [128, 8000]
Concat skip: [128+32 = 160, 8000]
Conv → [64, 8000]


---

## 6. Skip Connections

At each encoder stage, we **save** a feature map before downsampling:

Encoder E1: [16, 16000]
Encoder E2: [32, 8000]
Encoder E3: [64, 4000]


During decoding, when we upsample back to those sizes, we **concatenate**:

Decoder D2: [128, 8000]
Concat E2: [32, 8000]
Combined: [160, 8000]


Benefits:

- restore fine waveform details  
- help reconstruct consonants and formants  
- prevent oversmoothing  
- improve restoration of natural speech shape  

This is the key reason U-Net works so well for denoising.

---

## 7. Model Output

The final layer is a **Conv1d(kernel_size=1)** that compresses channels to 1:

[batch, 16, time] → [batch, 1, time]


This 1-channel output is the **predicted clean waveform**.

---

## 8. Training the Denoiser

### Training Data
We train on pairs:

input: noisy waveform
target: clean waveform

### Loss Function
We use **MSE** or **L1**:

loss = mean((predicted_clean - true_clean)^2)


This penalizes differences between the output waveform and the clean waveform.

### Optimization
Using Adam:

loss backward() → compute gradients
optimizer.step() → update Conv1d filter weights


Filters change each step to reduce the reconstruction error.

After training for multiple epochs, the U-Net learns:

- to suppress noise patterns  
- to preserve speech structure  
- to generate cleaner waveforms  

---

# 📝 Evaluation with Whisper-small

After training:

1. Feed noisy test audio to each pipeline:
   - A: No denoiser  
   - B: DSP  
   - C: U-Net  

2. Send resulting waveform into Whisper-small:
whisper(input_audio) → text


3. Compare transcripts:
   - Word Error Rate (WER)
   - qualitative listening
   - spectrogram comparison
   - visual waveform comparison

Pipeline C should show:
- smoother spectrograms  
- reduced background noise  
- higher ASR accuracy  


---

# 📈 Results

(Your numbers/images go here)

You may include:
- Waveform plots (noisy vs denoised)
- Spectrograms
- Whisper transcripts for each pipeline
- WER comparison table

---

# 🔧 Installation
Requirements:
- torch
- torchaudio
- librosa
- noisereduce
- numpy
- transformers
- matplotlib

---

# 🙌 Credits

- OpenAI Whisper-small  
- UNet architecture inspired by Ronneberger et al.  
- Classic DSP denoising using noisereduce  

---







