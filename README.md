# FastSpeech 2 — Text-to-Speech Synthesis (English & Telugu)

> Implementation of FastSpeech 2 for English and Telugu TTS as part of an academic assignment.  
> Reference: [FastSpeech 2 Paper](https://arxiv.org/abs/2006.04558) | [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)

---

## 📁 Repository Structure

```
FastSpeech2_Assignment/
├── English/
│   ├── preprocess.py          # Svarah dataset preprocessing + MFA alignment
│   ├── train.py               # FastSpeech 2 English training loop
│   ├── inference.py           # English inference + variance adaptor plots
│   ├── config.yaml            # English training configuration
│   └── checkpoints/           # Saved model checkpoints (see note below)
├── Telugu/
│   ├── preprocess.py          # IndicTTS Telugu preprocessing + phonemization
│   ├── train.py               # FastSpeech 2 Telugu training loop
│   ├── inference.py           # Telugu inference + variance adaptor plots
│   ├── config.yaml            # Telugu training configuration
│   └── checkpoints/           # Saved model checkpoints (see note below)
├── model/
│   ├── fastspeech2.py         # Main FastSpeech 2 model
│   ├── variance_adaptor.py    # Duration, pitch, energy predictors + length regulator
│   ├── modules.py             # FFTBlock, MultiHeadAttention, PostNet, etc.
│   ├── loss.py                # FastSpeech2Loss (mel + duration + pitch + energy)
│   └── __init__.py
├── utils/
│   ├── audio.py               # Mel spectrogram, pitch (pyworld), energy extraction
│   ├── text_english.py        # English phonemization (phonemizer + ARPAbet)
│   ├── text_telugu.py         # Telugu phonemization (epitran + indic-transliteration)
│   └── plot_utils.py          # Plotting: mel, duration, pitch, energy, loss curves
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/     # E1.wav, E2a.wav, E2b.wav, E3.wav, E4.wav
│   │   └── variance_adaptor_plots/
│   │       ├── duration/              # E1_duration.png ... E4_duration.png
│   │       ├── pitch/                 # E1_pitch.png ... E4_pitch.png
│   │       └── energy/                # E1_energy.png ... E4_energy.png
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/     # T1.wav, T2a.wav, T2b.wav, T3.wav, T4.wav
│       └── variance_adaptor_plots/
│           ├── duration/
│           ├── pitch/
│           └── energy/
├── Report/
│   └── Technical_Report.pdf
├── Colab_English.ipynb        # 🔑 Run this on Google Colab for English training
├── Colab_Telugu.ipynb         # 🔑 Run this on Google Colab for Telugu training
├── README.md
└── requirements.txt
```

---

## 🚀 Quick Start (Google Colab — Recommended)

> **Hardware requirement:** This project requires 8+ GB VRAM GPU. Use Google Colab (free T4 GPU).

### Step 1 — Open Colab Notebooks

| Language | Notebook | Open in Colab |
|---|---|---|
| English | `Colab_English.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) |
| Telugu | `Colab_Telugu.ipynb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) |

**Run both notebooks in parallel** in separate browser tabs to save time.

### Step 2 — Upload Project Files to Colab

Option A — ZIP upload (recommended):
```bash
# Zip this repo locally and upload to Colab session storage
zip -r FastSpeech2_Assignment.zip FastSpeech2_Assignment/
```

Option B — GitHub clone (after pushing to GitHub):
```bash
!git clone https://github.com/YOUR_USERNAME/FastSpeech2_Assignment /content/FastSpeech2
```

### Step 3 — Run All Cells
Run all cells top-to-bottom in each notebook. Each notebook handles:
1. Installing dependencies (espeak-ng, MFA, phonemizer, pyworld, etc.)
2. Downloading the dataset from HuggingFace
3. Preprocessing (mel spectrograms, pitch, energy, MFA alignment)
4. Training FastSpeech 2 (~2–5 hours on T4)
5. Inference on test sentences + variance adaptor plots
6. Saving everything to Google Drive

---

## 🗃️ Datasets

| Language | Dataset | Source | Train | Validation |
|---|---|---|---|---|
| English | [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | AI4Bharat | ~5 hours | ~1 hour |
| Telugu | [IndicTTS Telugu](https://www.iitm.ac.in/donlab/tts/database.php) | IIT Madras | ~5 hours | ~1 hour |

### Data Selection Strategy

**English (Svarah):**
- Streamed from HuggingFace in order, accumulating duration until 6 hours total
- First ~5 hours → training set | Next ~1 hour → validation set
- No speaker-level stratification (Svarah is multi-speaker Indian English)

**Telugu (IndicTTS):**
- Selected from the beginning of the corpus, ordered by file ID
- First ~5 hours → training | Next ~1 hour → validation
- Single-speaker corpus (IIT Madras TTS speaker)

---

## 🔧 Local Setup (Advanced — requires 8+ GB VRAM GPU)

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/FastSpeech2_Assignment
cd FastSpeech2_Assignment

# 2. Create virtual environment
conda create -n fastspeech2 python=3.9 -y
conda activate fastspeech2

# 3. Install requirements
pip install -r requirements.txt

# 4. Install MFA (Montreal Forced Aligner)
conda install -c conda-forge montreal-forced-aligner -y
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
# For Telugu:
mfa model download acoustic telugu_mfa  # if available

# 5. Install system dependencies
# Linux: sudo apt-get install espeak-ng sox ffmpeg
# macOS: brew install espeak sox ffmpeg
```

### Preprocessing

```bash
# English
cd FastSpeech2_Assignment
python English/preprocess.py --config English/config.yaml

# Telugu
python Telugu/preprocess.py --config Telugu/config.yaml
```

### Training

```bash
# English
python English/train.py --config English/config.yaml

# Telugu (run in a separate terminal simultaneously)
python Telugu/train.py --config Telugu/config.yaml
```

### Inference

```bash
# English test sentences
python English/inference.py \
  --checkpoint English/checkpoints/best_checkpoint.pth \
  --config English/config.yaml \
  --output_dir Results/English

# Telugu test sentences
python Telugu/inference.py \
  --checkpoint Telugu/checkpoints/best_checkpoint.pth \
  --config Telugu/config.yaml \
  --output_dir Results/Telugu
```

---

## 🧪 Test Sentences

### English

| ID | Sentence |
|---|---|
| E1 | The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings. |
| E2a | The orchestra's conduct during the ceremony was praised by the audience. |
| E2b | The researchers will conduct an experiment to evaluate the proposed method. |
| E3 | The rhythm of the orchestra changed unexpectedly. |
| E4 | The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time. |

### Telugu

| ID | Sentence |
|---|---|
| T1 | న్యూ రోఫార్మ కోలాజికల్ పరిశోధనలో నూతన ఔషధాల ప్రభావాన్ని విశ్లేషించారు. |
| T2a | తాడుపై అతని పట్టు చాలా బలంగా ఉంది. |
| T2b | ఆమె పట్టు చీరను ధరించి వేడుకకు వచ్చింది. |
| T3 | పులి అడవిలో నిశ్శబ్దంగా నడిచింది. |
| T4 | ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి. |

---

## 🏗️ Model Architecture

FastSpeech 2 consists of:
1. **Encoder** — 4× FFT (Feed-Forward Transformer) blocks with multi-head self-attention
2. **Variance Adaptor** — Duration predictor + Length Regulator + Pitch predictor + Energy predictor
3. **Decoder** — 4× FFT blocks
4. **Mel-Linear** + **PostNet** — Project to 80-band mel spectrogram + 5-conv refinement

**Vocoder:** HiFi-GAN (pretrained LJ_V2 checkpoint) for mel → waveform conversion

Key hyperparameters (same for both languages):
| Parameter | Value |
|---|---|
| Encoder/Decoder hidden dim | 256 |
| Attention heads | 2 |
| FFT layers | 4 |
| Mel channels | 80 |
| Sample rate | 22050 Hz |
| Batch size | 16 |
| Learning rate | 1e-4 (Adam + warmup) |
| Warmup steps | 4000 |

---

## 📦 Model Checkpoints

Checkpoints exceed GitHub's 100MB file size limit. Download from:

> 🔗 **[Google Drive — Model Checkpoints](https://drive.google.com/drive/folders/YOUR_FOLDER_ID)**
>
> Place downloaded checkpoints at:
> - `English/checkpoints/best_checkpoint.pth`
> - `Telugu/checkpoints/best_checkpoint.pth`

---

## 📊 Results Summary

See [`Results/`](./Results/) for:
- Training and validation loss curves (PNG)
- Generated audio for all 10 test sentences (WAV)
- Variance adaptor plots — duration, pitch (F0), energy — per sentence (PNG)

See [`Report/Technical_Report.pdf`](./Report/Technical_Report.pdf) for full analysis.

---

## 🔄 Modifications from Reference Implementation

| Component | English | Telugu |
|---|---|---|
| Phonemizer | `phonemizer` (espeak backend, ARPAbet) | `epitran` (tel-Telu → IPA) |
| Text cleaner | Standard English normalizer | Telugu Unicode normalizer + numeral expansion |
| MFA dictionary | `english_us_arpa` | `telugu_mfa` (fallback: proportional alignment) |
| Dataset loader | HuggingFace `ai4bharat/Svarah` | HuggingFace IndicTTS / IIT Madras local |
| Vocab size | ~50 (ARPAbet phones + special) | ~70 (Telugu phoneme units + special) |

---

## 📋 Dependencies

See [`requirements.txt`](./requirements.txt) for full list. Key packages:
- `torch >= 2.0`, `torchaudio`
- `librosa`, `pyworld`, `soundfile`
- `phonemizer`, `epitran`, `indic-transliteration`
- `datasets` (HuggingFace)
- `montreal-forced-aligner` (conda)

---

## 👤 Author

Submitted as part of TTS Assignment — FastSpeech 2 Implementation  
Deadline: 07-11-2026

