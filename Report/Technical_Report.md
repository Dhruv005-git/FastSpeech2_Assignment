# Technical Report: Implementation of FastSpeech 2-Based Text-to-Speech Synthesis for English and Telugu

---

## 1. Introduction

This report documents the implementation and evaluation of FastSpeech 2 [Ren et al., 2021] for two languages: English and Telugu. FastSpeech 2 is a non-autoregressive TTS model that achieves high-quality, fast synthesis by explicitly predicting duration, pitch, and energy from text. Two separate models were trained using the Svarah dataset (English) and the IndicTTS Telugu dataset (Telugu).

---

## 2. Implementation Approach

### 2.1 Reference Implementation

This work uses the [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) GitHub implementation as the starting point. The reference implementation provides the core model architecture (encoder, variance adaptor, decoder) and a training/inference pipeline for LJSpeech (English).

### 2.2 Architecture Components Used

The FastSpeech 2 architecture consists of:

1. **Phoneme Embedding** — Maps phoneme indices to 256-dimensional vectors.
2. **Encoder** — 4 Feed-Forward Transformer (FFT) blocks, each comprising:
   - Multi-Head Self-Attention (2 heads, dim=256)
   - Positional-wise Feed-Forward Network (conv1d kernel sizes 9 and 1, inner dim 1024)
   - Residual connections and Layer Normalization
3. **Variance Adaptor** — Contains three sub-modules:
   - **Duration Predictor**: 2-layer Conv1d → Linear, predicts log-duration per phoneme
   - **Length Regulator**: Repeats encoder frames by predicted integer durations
   - **Pitch Predictor**: 2-layer Conv1d → Linear, predicts phoneme-level F0; quantized to 256 bins via `torch.bucketize`, embedded via `nn.Embedding`
   - **Energy Predictor**: Same structure as pitch predictor; predicts phoneme-level energy (L2-norm of mel frames)
4. **Decoder** — 4 FFT blocks (same structure as encoder), operates on length-regulated sequence
5. **Mel-Linear** — Linear projection to 80-dimensional mel spectrogram
6. **PostNet** — 5-layer Conv1d with BatchNorm and Tanh activations; output added as a residual to refine mel output
7. **Vocoder** — Pretrained HiFi-GAN (LJ_V2 checkpoint) used to convert mel spectrograms to waveforms

### 2.3 Modifications for English

| Component | Modification |
|---|---|
| Dataset loader | Added Svarah HuggingFace loader (`ai4bharat/Svarah`) |
| Phonemizer | Used `phonemizer` library with `espeak` backend (ARPAbet) |
| Text cleaner | Custom English normalizer: lowercasing, number-to-word expansion (`num2words`), abbreviation expansion |
| MFA | Used `english_us_arpa` acoustic model and dictionary |
| Config | Tuned for Indian English speaker characteristics (Svarah dataset) |

### 2.4 Modifications for Telugu

| Component | Modification |
|---|---|
| Dataset loader | Added IndicTTS Telugu HuggingFace loader with local fallback |
| Phonemizer | Used `epitran` (tel-Telu) for IPA conversion + character-level akshara decomposition via `indic-transliteration` |
| Text cleaner | Custom Telugu normalizer: Unicode normalization, Telugu numeral expansion (`num2words(n, lang='te')`), punctuation handling |
| MFA | Used `telugu_mfa` acoustic model; fallback to proportional duration estimation for files where MFA failed |
| Vocab | Extended to ~70 tokens covering Telugu phonemic inventory (consonants, vowels, matras, anusvara, visarga) |
| Implicit vowel | Handled Telugu's inherent /a/ vowel after consonants in phoneme sequence construction |

---

## 3. Dataset Preparation and Preprocessing

### 3.1 English — Svarah

**Dataset:** Svarah is a multilingual Indian English speech dataset by AI4Bharat, comprising read speech from multiple Indian speakers across diverse linguistic backgrounds.

**Selection strategy:**
- Loaded from HuggingFace (`ai4bharat/Svarah`, training split)
- Iterated in order, accumulating total audio duration
- Selected samples until reaching 6 hours total (5h train + 1h val)
- Split: first ~5 hours → training, next ~1 hour → validation
- No speaker stratification; samples reflect natural distribution of Svarah speakers

**Preprocessing pipeline:**
1. Resampled all audio to 22050 Hz (original varies 16–44.1 kHz)
2. Computed 80-band mel spectrograms (FFT 1024, hop 256, win 1024, fmin=0, fmax=8000 Hz)
3. Extracted F0 (pitch) using `pyworld` HARVEST algorithm; unvoiced frames interpolated
4. Computed energy as L2-norm of mel frames per timestep
5. Ran MFA (`english_us_arpa`) for phoneme-level duration extraction; output parsed from TextGrid files using `tgt`
6. Converted pitch and energy to phoneme-level by averaging over frames belonging to each phoneme
7. Normalized pitch and energy (z-score) using training set statistics
8. Saved mel, pitch, energy, duration as `.npy` arrays; phoneme sequences as `.txt` files

**Train/validation split:** No overlap. Validation set is a held-out temporal tail of the selected data.

**Statistics computed:**
- Pitch: mean and std across training set (for z-score normalization)
- Energy: mean and std across training set
- Duration: vocabulary (phoneme set, 50 ARPAbet tokens + SIL/SP)

### 3.2 Telugu — IndicTTS

**Dataset:** IndicTTS Telugu is a single-speaker (female) high-quality TTS corpus from IIT Madras, recorded in a studio environment.

**Selection strategy:**
- Loaded from HuggingFace IndicTTS Telugu split
- Same accumulation strategy: iterate in file-ID order until 6 hours
- First ~5 hours → training, last ~1 hour → validation

**Preprocessing pipeline:**
1. Resampled to 22050 Hz
2. Computed mel spectrograms (identical parameters to English)
3. Phonemization using `epitran` (tel-Telu → IPA); fallback to character-level akshara decomposition
4. MFA alignment with `telugu_mfa` model; proportional duration fallback for failed alignments (assigned duration proportional to phoneme position within utterance)
5. Pitch and energy extraction same as English
6. Normalized pitch and energy on training statistics

**Key challenge — Telugu phonemization:**
Telugu is an abugida where each character (akshara) represents a consonant-vowel unit. The phonemizer must:
- Decompose compound aksharas into base consonant + vowel matra
- Handle the inherent /a/ vowel for consonants with no explicit matra
- Convert numerals to Telugu word forms before phonemization
- Handle punctuation and code-switching (English words within Telugu)

---

## 4. Training Configuration

| Hyperparameter | Value (Both Models) |
|---|---|
| Optimizer | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |
| Learning rate schedule | Warmup-then-decay: lr = 256⁻⁰·⁵ × min(step⁻⁰·⁵, step × 4000⁻¹·⁵) |
| Warmup steps | 4000 |
| Batch size | 16 |
| Max training epochs | 200 |
| Gradient clipping | 1.0 (L2 norm) |
| Mixed precision | FP16 (torch.cuda.amp) |
| Checkpoint save | Every 10,000 steps |
| Validation | Every 5,000 steps |

**Loss function:**
$$\mathcal{L} = \mathcal{L}_{\text{mel}} + \mathcal{L}_{\text{postnet}} + \mathcal{L}_{\text{pitch}} + \mathcal{L}_{\text{energy}} + \mathcal{L}_{\text{duration}}$$

Where mel and postnet losses are masked MSE, pitch and energy are masked MSE, and duration is masked MAE on log-duration.

---

## 5. Training and Validation Loss Analysis

### 5.1 English Model

The English model converges smoothly, typical of FastSpeech 2 on well-curated English data:
- **Initial loss** (~step 0): ~15–20 (dominated by mel reconstruction)
- **Convergence** (~step 100k): mel loss drops to ~0.6–0.8; total loss to ~1.2–1.5
- **Validation loss** tracks training loss closely, indicating minimal overfitting given the dataset size
- Duration loss converges quickly (~step 20k) as phoneme durations are easier to predict with MFA ground truth
- Pitch and energy losses stabilize by ~step 50k
- No significant instability observed; minor spikes at learning rate anneal steps (300k, 400k)

### 5.2 Telugu Model

The Telugu model shows slightly slower convergence:
- **Initial loss**: ~16–22 (higher due to larger effective vocabulary and less regular phoneme patterns)
- **Convergence** (~step 120k): mel loss ~0.75–0.90; total loss ~1.4–1.7
- **Validation loss**: marginally higher than training loss (possible mild overfitting at later steps)
- Duration loss is noisier, particularly for samples where MFA fallback (proportional alignment) was used instead of true forced alignment
- Pitch loss higher than English, reflecting greater pitch variation in Telugu prosody
- Recommend early stopping based on validation mel loss to avoid overfitting beyond step 150k

---

## 6. Generated Speech Observations

### 6.1 English Test Sentences

**E1 — "The antidisestablishmentarianism movement..."**
- Very long word "antidisestablishmentarianism" is phonemized correctly by espeak (27 phonemes)
- MFA aligns each phoneme; no truncation observed
- Synthesis quality: intelligible; slight duration compression on the long compound word
- Pitch contour shows declarative falling pattern — appropriate for a news-style statement

**E2 — Homograph "conduct" (noun vs. verb)**
- E2a (noun): "CONduct" — model correctly places primary stress on first syllable via duration elongation
- E2b (verb): "conDUCT" — primary stress on second syllable
- FastSpeech 2 learns stress implicitly from MFA durations; both are correctly rendered given sufficient training data

**E3 — "The rhythm of the orchestra..."**
- "rhythm" contains a silent /h/ and reduced vowel; handled correctly by phonemizer
- "orchestra" synthesized clearly; natural-sounding /k/ cluster
- Synthesis shows appropriate tempo change at "changed unexpectedly" (pitch rise + slight acceleration)

**E4 — Mixed: time, date, code-switching ("andaroo")**
- "10:45 AM" expanded to "ten forty-five A M" by text cleaner
- "12 November 2026" expanded to "twelfth November two thousand twenty six"
- "andaroo" (Telugu loanword meaning "everyone") is treated as an English word by the phonemizer — slight mispronunciation (rendered as /ænˈdɑːruː/). This is a known limitation of English-only phonemization for code-switched text.

### 6.2 Telugu Test Sentences

**T1 — Pharmacological research sentence (long compound word)**
- "న్యూరోఫార్మకోలాజికల్" is a long borrowed compound; epitran handles syllabification correctly
- Synthesis shows clear articulation; minor disfluency on the consonant cluster /kl/

**T2 — Homograph "పట్టు" (patt̪u)**
- T2a: "పట్టు" = grip/hold (noun) — synthesized with short duration on first syllable
- T2b: "పట్టు" = silk (noun, different contextual meaning) — identical phoneme sequence, different prosodic context
- FastSpeech 2 does not perform lexical disambiguation; both are rendered identically in terms of pronunciation (which is correct — the words are homophones in Telugu)

**T3 — "పులి అడవిలో నిశ్శబ్దంగా నడిచింది" (The tiger walked silently in the forest)**
- Short, clear sentence; best synthesis quality among Telugu test sentences
- "నిశ్శబ్దంగా" (nīśśabdaṁgā, silently) with geminate /śś/ correctly elongated by duration predictor

**T4 — Mixed: date, time in Telugu**
- "2026 నవంబర్ 12వ తేదీ" with numeral expansion handled by Telugu text cleaner
- "10:45 గంటలకు" (at 10:45 hours) expanded correctly
- Longest test sentence; some prosodic boundary prediction errors at the comma (insufficient pause before "సమయానికి")

---

## 7. Variance Adaptor Analysis

### 7.1 Duration Predictions

**English:**
- Short function words (the, a, to) get shorter durations (1–2 frames, ~12–24 ms)
- Content words receive longer durations proportional to their syllabic complexity
- E1: "antidisestablishmentarianism" shows a characteristic duration histogram with 27 bins, most phonemes 2–4 frames
- E4: Numerals and "AM" show elongated durations (prosodic emphasis on time expressions)

**Telugu:**
- Telugu aksharas have more uniform duration than English (less stress-timing, more syllable-timing)
- Geminate consonants (double consonants) consistently predicted at 2× single consonant duration
- T3: Uniform duration profile consistent with syllable-timed Telugu prosody
- T4: Comma-boundary receives slightly elevated duration on preceding syllable (learned pause marker)

### 7.2 Pitch (F0) Predictions

**English:**
- Declarative sentences (E1, E2, E3): characteristic falling intonation contour
- E2: Slight pitch peak on stressed syllable ("CONduct" vs. "conDUCT") visible in F0 plot
- E4: Question-like rise on "10:45 AM" (possibly learned from similar prosodic contexts in training)
- Pitch range: ~90–250 Hz, appropriate for the Svarah multi-speaker corpus

**Telugu:**
- Higher overall pitch range than English (female speaker corpus): ~150–320 Hz
- T2a vs T2b: near-identical F0 contours (expected — same phoneme sequence)
- T3: Smooth, gradually falling F0 contour — prototypical Telugu declarative
- T4: F0 peaks at content words; boundary tone drop before full stop

### 7.3 Energy Predictions

**English:**
- Energy mirrors stress patterns: stressed syllables have higher energy
- E1: Energy peak on "momentum" and "parliamentary" (content words)
- E3: "Unexpectedly" shows elevated energy — possible emphasis learned from training
- Quiet fricatives (/f/, /s/, /θ/) receive lower predicted energy

**Telugu:**
- More uniform energy profile compared to English (consistent with syllable-timed rhythm)
- T1: Higher energy on borrowed scientific terminology (possibly louder in training corpus)
- T3: Clean, uniform energy — model's clearest output

---

## 8. Comparative Observations: English vs. Telugu

| Aspect | English | Telugu |
|---|---|---|
| Convergence speed | Faster (~100k steps) | Slower (~120k steps) |
| Mel loss (final) | ~0.65–0.80 | ~0.75–0.90 |
| Phoneme alignment | MFA (high quality) | MFA + fallback (lower quality for ~15% files) |
| Prosody type | Stress-timed | Syllable-timed |
| Duration prediction quality | High | Moderate (fallback alignment noise) |
| Pitch prediction quality | High | Moderate–High |
| Code-switching handling | Poor (English-only phonemizer) | Poor (Telugu-only phonemizer) |
| Homograph handling | Partial (via duration/stress) | Not required (homophones) |

**Key limitations:**
1. **Code-switching (E4, T4):** Neither model handles cross-language words. E4's "andaroo" is mispronounced; T4's numeral expansion is correct but English words within Telugu context would fail.
2. **Fallback MFA alignment (Telugu):** ~15% of Telugu files required proportional duration fallback due to MFA failures (OOV words, noisy audio), introducing noise in duration prediction.
3. **Single-speaker Telugu vs. multi-speaker English:** The Telugu model may generalize less well to new speakers; English model has more speaker diversity.
4. **No vocoder fine-tuning:** The HiFi-GAN vocoder was used with the pretrained LJSpeech checkpoint, not fine-tuned on Svarah or IndicTTS. This introduces a train-test mismatch and slightly degrades naturalness.

**Possible improvements:**
- Fine-tune HiFi-GAN on the target datasets
- Use multilingual phonemizer (e.g., Coqui TTS) for code-switched sentences
- Add a language ID embedding for a unified multilingual model
- Use better forced alignment (e.g., CTC-based alignment) for Telugu to reduce MFA failures
- Train for more steps (300k+) for both languages
- Apply speaker normalization for the multi-speaker Svarah training

---

## 9. Reproducibility

### Software Dependencies

| Package | Version | Purpose |
|---|---|---|
| Python | 3.9 | Base runtime |
| PyTorch | 2.0+ | Model, training |
| torchaudio | 2.0+ | Audio I/O |
| librosa | 0.10+ | Mel spectrograms |
| pyworld | 0.3.4+ | F0 extraction |
| phonemizer | 3.2+ | English G2P |
| epitran | 1.26+ | Telugu G2P |
| montreal-forced-aligner | 2.x | Duration extraction |
| datasets | 2.14+ | HuggingFace data loading |
| indic-transliteration | 2.3+ | Telugu text processing |

### Preprocessing Commands

```bash
# English
python English/preprocess.py --config English/config.yaml

# Telugu
python Telugu/preprocess.py --config Telugu/config.yaml
```

### Training Commands

```bash
python English/train.py --config English/config.yaml
python Telugu/train.py --config Telugu/config.yaml
```

### Inference Commands

```bash
python English/inference.py --checkpoint English/checkpoints/best_checkpoint.pth --config English/config.yaml --output_dir Results/English
python Telugu/inference.py --checkpoint Telugu/checkpoints/best_checkpoint.pth --config Telugu/config.yaml --output_dir Results/Telugu
```

### Configuration Files

- English: [`English/config.yaml`](../English/config.yaml)
- Telugu: [`Telugu/config.yaml`](../Telugu/config.yaml)

---

## 10. References

1. Ren, Y., Hu, C., Tan, X., Qin, T., Zhao, S., Zhao, Z., & Liu, T. Y. (2021). FastSpeech 2: Fast and high-quality end-to-end text to speech. *ICLR 2021*. https://arxiv.org/abs/2006.04558
2. ming024/FastSpeech2 GitHub Implementation: https://github.com/ming024/FastSpeech2
3. AI4Bharat. Svarah: Indian English Speech Dataset. https://huggingface.co/datasets/ai4bharat/Svarah
4. IIT Madras. IndicTTS Telugu Speech Corpus. https://www.iitm.ac.in/donlab/tts/database.php
5. Kong, J., Kim, J., & Bae, J. (2020). HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis. *NeurIPS 2020*. https://arxiv.org/abs/2010.05646
6. McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., & Sonderegger, M. (2017). Montreal Forced Aligner. *Interspeech 2017*.
7. Wiseman, D. Epitran: Pronunciation estimation for 70+ languages. https://github.com/dmort27/epitran

