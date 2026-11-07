"""
utils/text_telugu.py
====================
Telugu text-processing utilities for the FastSpeech 2 pipeline.

Script: Telugu Unicode block U+0C00–U+0C7F
Phoneme backend: Epitran (tel-Telu → IPA) with a character-level fallback.

Dependencies (install what you need):
    pip install epitran indic-transliteration num2words
    # epitran may also need: espeak-ng, flite (system packages)
"""

import json
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

# Optional imports — each section degrades gracefully if unavailable
try:
    import epitran
    _epi = epitran.Epitran("tel-Telu")
    _HAS_EPITRAN = True
except Exception:
    _epi = None
    _HAS_EPITRAN = False

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    _HAS_INDIC_TRANSLIT = True
except ImportError:
    _HAS_INDIC_TRANSLIT = False

try:
    from num2words import num2words as _num2words
    _HAS_NUM2WORDS = True
except ImportError:
    _HAS_NUM2WORDS = False


# ---------------------------------------------------------------------------
# Telugu Unicode ranges
# ---------------------------------------------------------------------------
# Full Telugu block: 0C00–0C7F
_TELUGU_RANGE = (0x0C00, 0x0C7F)

# Independent vowels: అ ఆ ఇ ఈ ఉ ఊ ఋ ఌ ఎ ఏ ఐ ఒ ఓ ఔ
_TELUGU_VOWELS = set("అఆఇఈఉఊఋఌఎఏఐఒఓఔ")

# Dependent vowel signs (matras)
_TELUGU_VOWEL_SIGNS = set("ాిీుూృెేైొోౌ")

# Consonants: క ఖ గ ఘ ఙ చ ఛ జ ఝ ఞ ట ఠ డ ఢ ణ
#             త థ ద ధ న ప ఫ బ భ మ య ర ల వ శ ష స హ ళ క్ష జ్ఞ
_TELUGU_CONSONANTS = set("కఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరలవశషసహళ")

# Virama (halant) — suppresses the inherent vowel
_VIRAMA = "\u0C4D"

# Anusvara, Visarga
_ANUSVARA  = "\u0C02"
_VISARGA   = "\u0C03"

# Chillu / special characters
_SUNNA    = "\u0C00"  # combining candrabindu (Sunna in Telugu = anusvara here)


# ---------------------------------------------------------------------------
# Numeral mapping
# ---------------------------------------------------------------------------

# Telugu numeral digit characters → Arabic digit
_TELUGU_DIGIT_MAP: Dict[str, str] = {
    "౦": "0", "౧": "1", "౨": "2", "౩": "3", "౪": "4",
    "౫": "5", "౬": "6", "౭": "7", "౮": "8", "౯": "9",
}

# Telugu word equivalents for digits 0-9 (used when num2words unavailable)
_TELUGU_NUMBER_WORDS: Dict[int, str] = {
    0:  "సున్న",
    1:  "ఒకటి",
    2:  "రెండు",
    3:  "మూడు",
    4:  "నాలుగు",
    5:  "అయిదు",
    6:  "ఆరు",
    7:  "ఏడు",
    8:  "ఎనిమిది",
    9:  "తొమ్మిది",
    10: "పది",
    11: "పదకొండు",
    12: "పన్నెండు",
    13: "పదమూడు",
    14: "పదనాలుగు",
    15: "పదిహేను",
    16: "పదహారు",
    17: "పదిహేడు",
    18: "పదిహేనిమిది",
    19: "పంతొమ్మిది",
    20: "ఇరవై",
    30: "ముప్పై",
    40: "నలభై",
    50: "యాభై",
    60: "అరవై",
    70: "డెబ్బై",
    80: "ఎనభై",
    90: "తొంభై",
    100: "వంద",
    1000: "వేయి",
}


def _int_to_telugu_words(n: int) -> str:
    """Convert a non-negative integer to Telugu word form (simple algorithm)."""
    if _HAS_NUM2WORDS:
        try:
            return _num2words(n, lang="te")
        except Exception:
            pass
    # Fallback: basic decomposition
    if n in _TELUGU_NUMBER_WORDS:
        return _TELUGU_NUMBER_WORDS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        t_word = _TELUGU_NUMBER_WORDS.get(tens * 10, str(tens * 10))
        o_word = _TELUGU_NUMBER_WORDS.get(ones, str(ones))
        return t_word + (" " + o_word if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        h_word = _TELUGU_NUMBER_WORDS.get(hundreds, str(hundreds)) + " వందలు"
        return h_word + (" " + _int_to_telugu_words(rest) if rest else "")
    if n < 100000:
        thousands, rest = divmod(n, 1000)
        t_word = _TELUGU_NUMBER_WORDS.get(thousands, str(thousands)) + " వేలు"
        return t_word + (" " + _int_to_telugu_words(rest) if rest else "")
    # For very large numbers, return digit string
    return str(n)


# ---------------------------------------------------------------------------
# Text cleaning / normalisation
# ---------------------------------------------------------------------------

def _replace_telugu_digits(text: str) -> str:
    """Convert Telugu numeral glyphs to their Arabic equivalents."""
    for tel_digit, arabic in _TELUGU_DIGIT_MAP.items():
        text = text.replace(tel_digit, arabic)
    return text


def _expand_numbers(text: str) -> str:
    """Replace Arabic numerals (after Telugu→Arabic conversion) with words."""
    def _replace(m: re.Match) -> str:
        num_str = m.group(0)
        try:
            return _int_to_telugu_words(int(num_str))
        except Exception:
            return num_str
    return re.sub(r"\d+", _replace, text)


def _remove_non_telugu_non_ascii(text: str) -> str:
    """Keep Telugu characters, ASCII letters, spaces, and basic punctuation."""
    result = []
    for ch in text:
        cp = ord(ch)
        if _TELUGU_RANGE[0] <= cp <= _TELUGU_RANGE[1]:
            result.append(ch)
        elif ch in " \t\n,.!?;:-–—\"'()":
            result.append(ch)
        elif ch.isascii() and ch.isprintable():
            result.append(ch)
        # else: drop the character
    return "".join(result)


def telugu_cleaners(text: str) -> str:
    """Normalise Telugu text for TTS preprocessing.

    Pipeline:
        1. Unicode NFC normalisation
        2. Convert Telugu numeral glyphs (౦–౯) to Arabic digits
        3. Expand Arabic numerals to Telugu word forms
        4. Strip characters outside the Telugu Unicode block / ASCII punctuation
        5. Collapse whitespace

    Args:
        text: Raw Telugu text string.

    Returns:
        Cleaned Telugu text string.
    """
    text = unicodedata.normalize("NFC", text)
    text = _replace_telugu_digits(text)
    text = _expand_numbers(text)
    text = _remove_non_telugu_non_ascii(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Akshara / grapheme segmentation
# ---------------------------------------------------------------------------

def segment_aksharas(text: str) -> List[str]:
    """Segment a Telugu string into akshara (syllable) units.

    An akshara in Telugu is typically:
        consonant(s) + virama(s) + optional vowel sign
      OR
        independent vowel

    This function implements a hand-written finite automaton over the Telugu
    Unicode block.  It is used as the fallback when Epitran is unavailable.

    Args:
        text: NFC-normalised Telugu text.

    Returns:
        List of akshara strings.
    """
    aksharas: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        cp = ord(ch)

        # Telugu block character
        if _TELUGU_RANGE[0] <= cp <= _TELUGU_RANGE[1]:
            unit = ch
            i += 1
            # Consume conjunct consonants: consonant + virama + consonant ...
            while i < n and text[i] == _VIRAMA:
                unit += text[i]  # virama
                i += 1
                if i < n:
                    unit += text[i]  # next consonant
                    i += 1
            # Consume dependent vowel sign (matra)
            if i < n and text[i] in _TELUGU_VOWEL_SIGNS:
                unit += text[i]
                i += 1
            # Consume anusvara / visarga attached to this akshara
            if i < n and text[i] in (_ANUSVARA, _VISARGA):
                unit += text[i]
                i += 1
            aksharas.append(unit)
        else:
            # Non-Telugu character: emit as individual token
            aksharas.append(ch)
            i += 1

    return aksharas


# ---------------------------------------------------------------------------
# Grapheme-to-phoneme (G2P)
# ---------------------------------------------------------------------------

def aksharas_to_ipa(text: str) -> List[str]:
    """Convert Telugu text to a list of IPA phoneme tokens.

    Strategy:
      1. If Epitran (tel-Telu) is available → use it (most accurate).
      2. If indic_transliteration is available → transliterate to SLP1
         then map to approximate IPA via a hand-coded table.
      3. Fall back to raw akshara segmentation (character-level tokens).

    Args:
        text: Cleaned Telugu text (output of ``telugu_cleaners``).

    Returns:
        List of phoneme / grapheme token strings.
    """
    # Strategy 1: Epitran
    if _HAS_EPITRAN and _epi is not None:
        try:
            ipa_str = _epi.transliterate(text)
            # Split IPA string into individual IPA characters / digraphs
            # Simple approach: split on whitespace boundaries first, then
            # keep multi-character IPA phones (ː, ʰ, etc.) attached.
            tokens = _split_ipa(ipa_str)
            return [t for t in tokens if t.strip()]
        except Exception as e:
            print(f"[text_telugu] Epitran failed ({e}). Falling back.")

    # Strategy 2: indic_transliteration → approximate IPA
    if _HAS_INDIC_TRANSLIT:
        try:
            slp1 = transliterate(text, sanscript.TELUGU, sanscript.SLP1)
            tokens = [_SLP1_TO_IPA.get(ch, ch) for ch in slp1 if ch.strip()]
            return tokens
        except Exception as e:
            print(f"[text_telugu] indic_transliteration failed ({e}). Falling back.")

    # Strategy 3: Character-level akshara segmentation
    return segment_aksharas(text)


def _split_ipa(ipa_str: str) -> List[str]:
    """Naively split an IPA string into phoneme tokens.

    Handles multi-char phones like tʃ, dʒ, ː (length mark), ʰ (aspiration).
    """
    # Diacritics / modifiers that attach to the previous token
    _MODIFIERS = set("ʰʲʷːˑ̃̈̈͡")
    tokens: List[str] = []
    for ch in ipa_str:
        if not ch.strip():
            # Whitespace → boundary
            if tokens and tokens[-1] != " ":
                tokens.append(" ")
        elif ch in _MODIFIERS and tokens:
            tokens[-1] = tokens[-1] + ch  # attach modifier
        else:
            tokens.append(ch)
    # Remove space tokens (boundaries already implicit)
    return [t for t in tokens if t != " " and t.strip()]


# Approximate SLP1 → IPA mapping for Telugu consonants & vowels
_SLP1_TO_IPA: Dict[str, str] = {
    "a": "ə", "A": "aː", "i": "i", "I": "iː",
    "u": "u", "U": "uː", "f": "r̩",
    "e": "eː", "E": "ɛː", "o": "oː", "O": "ɔː",
    "M": "m̃", "H": "h",
    "k": "k",  "K": "kʰ", "g": "ɡ",  "G": "ɡʱ", "N": "ŋ",
    "c": "tʃ", "C": "tʃʰ","j": "dʒ", "J": "dʒʱ","Y": "ɲ",
    "T": "ʈ",  "W": "ʈʰ", "D": "ɖ",  "Q": "ɖʱ", "R": "ɳ",
    "t": "t̪",  "w": "t̪ʰ", "d": "d̪",  "q": "d̪ʱ", "n": "n̪",
    "p": "p",  "P": "pʰ", "b": "b",  "B": "bʱ", "m": "m",
    "y": "j",  "r": "ɾ",  "l": "l",  "L": "ɭ",  "v": "ʋ",
    "S": "ʃ",  "z": "ʂ",  "s": "s",  "h": "ɦ",
    "x": "k",  "X": "kʰ",             # alternative mappings
}


# ---------------------------------------------------------------------------
# Vocabulary management
# ---------------------------------------------------------------------------

_SPECIAL_TOKENS: List[str] = ["<pad>", "<unk>", "<sil>"]


def build_telugu_vocab(texts: List[str]) -> Dict[str, int]:
    """Build a phoneme / grapheme vocabulary from a list of Telugu texts.

    Each text is first cleaned with ``telugu_cleaners``, then converted to
    phoneme tokens via ``aksharas_to_ipa``.  Unique tokens are sorted and
    assigned consecutive integer IDs (special tokens at indices 0-2).

    Args:
        texts: List of raw Telugu text strings.

    Returns:
        Dict mapping token string → integer ID.
    """
    unique: set = set()
    for raw in texts:
        cleaned = telugu_cleaners(raw)
        tokens  = aksharas_to_ipa(cleaned)
        unique.update(tokens)
    unique -= set(_SPECIAL_TOKENS)
    sorted_tokens = sorted(unique)

    vocab = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS)}
    for tok in sorted_tokens:
        vocab[tok] = len(vocab)
    return vocab


def save_vocab(vocab: Dict[str, int], path: str) -> None:
    """Serialise vocabulary to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(vocab_path: str) -> Dict[str, int]:
    """Load vocabulary from a JSON file produced by ``save_vocab``."""
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(
            f"Telugu vocab not found at '{vocab_path}'. "
            "Run preprocess.py first to build the vocabulary."
        )
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def text_to_sequence(text: str, vocab: Dict[str, int]) -> List[int]:
    """Convert a Telugu text string to a list of phoneme IDs.

    Args:
        text:  Raw Telugu text.
        vocab: Phoneme-to-ID mapping (from ``build_telugu_vocab``).

    Returns:
        List of integer IDs with silence tokens at start and end.
    """
    cleaned = telugu_cleaners(text)
    tokens  = aksharas_to_ipa(cleaned)
    unk_id  = vocab.get("<unk>", 1)
    sil_id  = vocab.get("<sil>", 2)
    ids = [sil_id] + [vocab.get(t, unk_id) for t in tokens] + [sil_id]
    return ids
