"""Text front-ends.
English : normalisation (numbers / times / years -> words) -> g2p_en (CMUdict + neural OOV model,
          POS-aware for homographs such as 'conduct') -> ARPAbet phonemes with stress.
Telugu  : Unicode NFC normalisation, Telugu number expansion -> grapheme (Unicode code-point) units.
          Telugu script is almost phonemic, so each Telugu code point (consonant, vowel sign,
          virama ...) is used as one input symbol; no external G2P is required.
"""
import re
import unicodedata

PAD, SP = "<pad>", "<sp>"
PUNCT = list(",.?!;:-")

# ---------------------------------------------------------------- English
ARPA = ("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH").split()
EN_PHONES = [p + s for p in ARPA for s in ("",)] + [p + str(i) for p in ARPA if p[0] in "AEIOU" for i in range(3)]
EN_PHONES = sorted(set(EN_PHONES))
EN_SYMBOLS = [PAD, SP] + PUNCT + EN_PHONES

_g2p = None


def _num2words(n, **kw):
    from num2words import num2words
    return num2words(n, **kw)


def normalize_en(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'")

    def _time(m):
        h, mi, ap = int(m.group(1)), int(m.group(2)), m.group(3)
        s = _num2words(h)
        if mi == 0:
            s += " o'clock" if not ap else ""
        else:
            s += (" oh " if mi < 10 else " ") + _num2words(mi)
        if ap:
            s += " " + " ".join(ap.upper().replace(".", ""))      # "A M"
        return s
    text = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?[Mm]\.?)?", _time, text)
    text = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: _num2words(int(m.group(1)), to="year"), text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: _num2words(int(m.group(1)), to="ordinal"), text)
    text = re.sub(r"\d+", lambda m: _num2words(int(m.group(0))), text)
    text = text.replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def text_to_units_en(text):
    global _g2p
    if _g2p is None:
        from g2p_en import G2p
        _g2p = G2p()
    out = []
    for p in _g2p(normalize_en(text)):
        if p == " ":
            out.append(SP)
        elif p in PUNCT:
            out.append(p)
        elif p in EN_SYMBOLS:
            out.append(p)
    return out


# ---------------------------------------------------------------- Telugu
TE_CHARS = [chr(c) for c in range(0x0C01, 0x0C4E)] + [chr(0x0C55), chr(0x0C56)]
TE_SYMBOLS = [PAD, SP] + PUNCT + TE_CHARS

_ONES = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు", "ఎనిమిది", "తొమ్మిది", "పది",
         "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పంతొమ్మిది"]
_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}


def telugu_number(n):
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + (" " + _ONES[o] if o else "")
    if n < 1000:
        h, r = divmod(n, 100)
        head = "వంద" if h == 1 else _ONES[h] + " వందల"
        return head + (" " + telugu_number(r) if r else "")
    if n < 100000:
        k, r = divmod(n, 1000)
        head = "వెయ్యి" if k == 1 else telugu_number(k) + " వేల"
        if r == 0 and k > 1:
            head = telugu_number(k) + " వేలు"
        return head + (" " + telugu_number(r) if r else "")
    return " ".join(_ONES[int(d)] for d in str(n))        # fall back to digit-by-digit


def _te_ordinal(n):
    w = telugu_number(n)
    return (w[:-1] if w[-1] in "ుి" else w) + "వ"


def normalize_te(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("‌", "").replace("‍", "")
    text = re.sub(r"\b(\d{1,2}):(\d{2})\b",
                  lambda m: telugu_number(int(m.group(1))) + (" " + telugu_number(int(m.group(2))) if int(m.group(2)) else ""), text)
    text = re.sub(r"(\d+)\s*వ\b", lambda m: _te_ordinal(int(m.group(1))), text)
    text = re.sub(r"\d+", lambda m: telugu_number(int(m.group(0))), text)
    return re.sub(r"\s+", " ", text).strip()


def text_to_units_te(text):
    out = []
    for ch in normalize_te(text):
        if ch == " ":
            out.append(SP)
        elif ch in PUNCT or ch in TE_CHARS:
            out.append(ch)
    return out


# ---------------------------------------------------------------- common
def get_frontend(language):
    if language == "english":
        return EN_SYMBOLS, text_to_units_en
    if language == "telugu":
        return TE_SYMBOLS, text_to_units_te
    raise ValueError(language)


def units_to_ids(units, symbols):
    table = {s: i for i, s in enumerate(symbols)}
    return [table[u] for u in units if u in table]
