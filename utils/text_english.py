"""
utils/text_english.py
=====================
English text-processing utilities for the FastSpeech 2 pipeline.

Phoneme backend: CMU ARPAbet (39 standard phones + silence/padding tokens).

Dependencies:
    pip install num2words nltk
    python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('cmudict')"
"""

import json
import os
import re
import unicodedata
from typing import Dict, List, Optional

# Try optional imports with graceful fallbacks
try:
    from num2words import num2words as _num2words
    _HAS_NUM2WORDS = True
except ImportError:
    _HAS_NUM2WORDS = False

try:
    import nltk
    _cmudict = dict(nltk.corpus.cmudict.entries())
    _HAS_CMUDICT = True
except Exception:
    _cmudict = {}
    _HAS_CMUDICT = False


# ---------------------------------------------------------------------------
# ARPAbet phoneme set
# ---------------------------------------------------------------------------
# 39 standard ARPAbet phonemes (stress marks stripped at lookup time)
_ARPABET_PHONES: List[str] = [
    "AA", "AE", "AH", "AO", "AW", "AY",
    "B",  "CH", "D",  "DH", "EH", "ER",
    "EY", "F",  "G",  "HH", "IH", "IY",
    "JH", "K",  "L",  "M",  "N",  "NG",
    "OW", "OY", "P",  "R",  "S",  "SH",
    "T",  "TH", "UH", "UW", "V",  "W",
    "Y",  "Z",  "ZH",
]

_SPECIAL_TOKENS: List[str] = ["<pad>", "<unk>", "<sil>"]

# Default vocabulary (special + ARPAbet)
_DEFAULT_VOCAB: Dict[str, int] = {
    tok: i for i, tok in enumerate(_SPECIAL_TOKENS + _ARPABET_PHONES)
}

# Reverse mapping
_DEFAULT_ID2PHONE: Dict[int, str] = {v: k for k, v in _DEFAULT_VOCAB.items()}

# Commonly expanded abbreviations
_ABBREVIATIONS = {
    r"\bmr\.":    "mister",
    r"\bmrs\.":   "misess",
    r"\bdr\.":    "doctor",
    r"\bst\.":    "saint",
    r"\bco\.":    "company",
    r"\bjr\.":    "junior",
    r"\bmaj\.":   "major",
    r"\bgen\.":   "general",
    r"\bprof\.":  "professor",
    r"\bcapt\.":  "captain",
    r"\blb\.":    "pound",
    r"\bft\.":    "foot",
    r"\bin\.":    "inch",
    r"\bvs\.":    "versus",
    r"\betc\.":   "et cetera",
    r"\be\.g\.":  "for example",
    r"\bi\.e\.":  "that is",
}


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _expand_abbreviations(text: str) -> str:
    for pattern, replacement in _ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _expand_numbers(text: str) -> str:
    """Replace integer and decimal numbers with their English word forms."""
    if not _HAS_NUM2WORDS:
        return text

    def _replace(m: re.Match) -> str:
        num_str = m.group(0).replace(",", "")
        try:
            if "." in num_str:
                return _num2words(float(num_str), lang="en")
            return _num2words(int(num_str), lang="en")
        except Exception:
            return num_str

    return re.sub(r"\d[\d,]*\.?\d*", _replace, text)


def english_cleaners(text: str) -> str:
    """Clean English text for TTS preprocessing.

    Pipeline:
        1. Unicode normalise (NFKC)
        2. Lowercase
        3. Expand common abbreviations
        4. Expand numbers to words (using num2words)
        5. Remove characters outside [a-z ,.'!?-]
        6. Collapse multiple spaces

    Args:
        text: Raw English text string.

    Returns:
        Cleaned ASCII-ish text suitable for CMU dict / G2P lookup.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _expand_abbreviations(text)
    text = _expand_numbers(text)
    # Keep only basic printable chars
    text = re.sub(r"[^a-z0-9 ,.'!?\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Phoneme ↔ index mapping
# ---------------------------------------------------------------------------

def phoneme_to_id(phoneme: str, vocab: Dict[str, int]) -> int:
    """Map a single phoneme string to its integer ID.

    Returns the ``<unk>`` ID if the phoneme is not in the vocabulary.
    """
    return vocab.get(phoneme, vocab.get("<unk>", 1))


def id_to_phoneme(idx: int, vocab: Dict[str, int]) -> str:
    """Map an integer ID back to its phoneme string.

    Returns ``<unk>`` if the ID is not in the vocabulary.
    """
    id2ph = {v: k for k, v in vocab.items()}
    return id2ph.get(idx, "<unk>")


def text_to_sequence(text: str, vocab: Dict[str, int]) -> List[int]:
    """Convert a cleaned English text string to a sequence of phoneme IDs.

    Uses CMU pronouncing dictionary for grapheme-to-phoneme conversion;
    falls back to character-level for out-of-vocabulary words.

    Args:
        text:  Cleaned English text (output of ``english_cleaners``).
        vocab: Phoneme-to-ID mapping (output of ``build_vocab`` or ``load_vocab``).

    Returns:
        List of integer phoneme IDs including silence boundaries.
    """
    words = text.split()
    ids: List[int] = [phoneme_to_id("<sil>", vocab)]

    for word in words:
        word_clean = re.sub(r"[^a-z]", "", word.lower())
        if not word_clean:
            continue
        if _HAS_CMUDICT and word_clean in _cmudict:
            phones = [re.sub(r"\d", "", p) for p in _cmudict[word_clean]]
        else:
            # Fallback: treat each character as a "phoneme"
            phones = list(word_clean.upper())
        ids.extend(phoneme_to_id(ph, vocab) for ph in phones)
        ids.append(phoneme_to_id("<sil>", vocab))

    return ids


# ---------------------------------------------------------------------------
# Vocabulary management
# ---------------------------------------------------------------------------

def build_vocab(phoneme_lists: List[List[str]]) -> Dict[str, int]:
    """Build a phoneme vocabulary from a collection of phoneme sequences.

    Special tokens (<pad>, <unk>, <sil>) are always present at indices 0-2.
    All remaining unique phonemes are sorted and assigned consecutive IDs.

    Args:
        phoneme_lists: Iterable of phoneme-token lists (one per utterance).

    Returns:
        Dict mapping phoneme string → integer ID.
    """
    unique = set()
    for seq in phoneme_lists:
        unique.update(seq)
    # Remove specials from the data-derived set (they are added explicitly)
    unique -= set(_SPECIAL_TOKENS)
    sorted_phones = sorted(unique)
    vocab = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS)}
    for ph in sorted_phones:
        vocab[ph] = len(vocab)
    return vocab


def save_vocab(vocab: Dict[str, int], path: str) -> None:
    """Serialise vocabulary dictionary to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(vocab_path: str) -> Dict[str, int]:
    """Load vocabulary dictionary from a JSON file.

    Falls back to the default ARPAbet vocabulary if the file is not found.

    Args:
        vocab_path: Path to a JSON file produced by ``save_vocab``.

    Returns:
        Dict mapping phoneme string → integer ID.
    """
    if not os.path.isfile(vocab_path):
        print(f"[text_english] Vocab not found at '{vocab_path}'. "
              "Using default ARPAbet vocab.")
        return dict(_DEFAULT_VOCAB)
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_vocab_size(vocab: Dict[str, int]) -> int:
    """Return the number of entries in the vocabulary."""
    return len(vocab)
