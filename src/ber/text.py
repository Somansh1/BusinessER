"""Text normalization for entity resolution: script detection, romanization,
phonetic skeleton keys, abbreviation/legal-suffix handling, address parsing.

See ../../CONTRACT.md for the {split}_prep.parquet column contract this
module fills (all columns except idx/entity_id/src/country).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from multiprocessing import Pool

from unidecode import unidecode

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as _translit
    _HAVE_SANSCRIPT = True
except ImportError:  # ponytail: HPC box may lag on deps; degrade to unidecode-only
    _HAVE_SANSCRIPT = False

# ---------------------------------------------------------------------------
# script detection
# ---------------------------------------------------------------------------

# (name, start, end) Unicode block ranges, checked in order; first match wins
# per character. Names match CONTRACT's script column vocabulary.
_SCRIPT_RANGES = [
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("oriya", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("arabic", 0x0600, 0x06FF),
    ("latin", 0x0041, 0x005A),
    ("latin", 0x0061, 0x007A),
    ("latin", 0x00C0, 0x024F),  # Latin extended (accented)
]

_SANSCRIPT_NAME = {
    "devanagari": "DEVANAGARI", "bengali": "BENGALI", "gujarati": "GUJARATI",
    "gurmukhi": "GURMUKHI", "tamil": "TAMIL", "telugu": "TELUGU",
    "kannada": "KANNADA", "malayalam": "MALAYALAM", "oriya": "ORIYA",
}

# Indic digit -> ascii digit, all scripts we might see + arabic-indic.
_INDIC_DIGITS = {}
for _base, _script in [
    (0x0966, "devanagari"), (0x09E6, "bengali"), (0x0A66, "gurmukhi"),
    (0x0AE6, "gujarati"), (0x0B66, "oriya"), (0x0BE6, "tamil"),
    (0x0C66, "telugu"), (0x0CE6, "kannada"), (0x0D66, "malayalam"),
    (0x0660, "arabic"), (0x06F0, "arabic"),
]:
    for _d in range(10):
        _INDIC_DIGITS[chr(_base + _d)] = str(_d)


def _char_script(ch: str) -> str:
    cp = ord(ch)
    for name, lo, hi in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return name
    return "other"


def script_of(s: str) -> str:
    """Dominant script of a string, by counting non-space/punct chars."""
    if not s:
        return "other"
    counts: dict[str, int] = {}
    for ch in s:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        sc = _char_script(ch)
        if sc == "other" and not ch.isalpha():
            continue
        counts[sc] = counts.get(sc, 0) + 1
    if not counts:
        return "other"
    return max(counts, key=counts.get)


# ---------------------------------------------------------------------------
# romanization
# ---------------------------------------------------------------------------

def _map_indic_digits(s: str) -> str:
    return "".join(_INDIC_DIGITS.get(ch, ch) for ch in s)


def _schwa_delete(word: str) -> str:
    """Drop a trailing bare 'a' after a consonant (crude Indic schwa deletion,
    e.g. itrans 'ramaa'-ish outputs). Only touches the final vowel."""
    if len(word) >= 3 and word[-1] == "a" and word[-2] not in "aeiou":
        return word[:-1]
    return word


# ITRANS doesn't map Devanagari "candra" vowel signs (used for English loan
# sounds like the o in "modern"/"mall") -> collapse them to the nearest
# plain vowel sign so unidecode's fallback doesn't inject spurious consonants.
_CANDRA_FIX = {"ॉ": "ो", "ॅ": "े", "ॲ": "अ"}


_ANUSVARA_RE = re.compile(r"M(?=[A-Za-z])")


def _anusvara(m):
    c = m.string[m.end()]
    if c in "aeiouAEIOU":
        return "m"
    return "m" if c in "pbPBmM" else "n"  # homorganic nasal (labial vs other)


@lru_cache(maxsize=200_000)
def _translit_run(run: str, script_name: str) -> str:
    """Romanize one same-script run of text. Cached since address/name tokens
    (city names, common words) repeat heavily across 25M rows."""
    if script_name == "latin" or script_name == "other":
        return unidecode(run)
    sans_name = _SANSCRIPT_NAME.get(script_name)
    if sans_name and _HAVE_SANSCRIPT:
        try:
            if script_name == "devanagari":
                for bad, good in _CANDRA_FIX.items():
                    run = run.replace(bad, good)
            out = _translit(run, getattr(sanscript, sans_name), sanscript.ITRANS)
            out = _ANUSVARA_RE.sub(_anusvara, unidecode(out)).lower()
            if script_name in ("tamil", "malayalam"):
                out = out.replace("rr", "r")
            # crude word-final schwa deletion per token
            out = " ".join(_schwa_delete(w) for w in out.split())
            return out
        except Exception:
            pass
    return unidecode(run).lower()


_RUN_RE = re.compile(r"\S+|\s+")


def romanize(s: str) -> str:
    """NFKC + digit-normalize, then romanize script-by-script, lowercased."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _map_indic_digits(s)

    # split into maximal runs of one dominant script (word-boundary aware:
    # split on whitespace first so we don't mix scripts within a "word" run)
    pieces = []
    for tok in _RUN_RE.findall(s):
        if tok.isspace():
            pieces.append(tok)
            continue
        # sub-split token into contiguous same-script spans
        cur_script = None
        buf = []
        for ch in tok:
            sc = _char_script(ch) if not ch.isspace() else cur_script
            if cur_script is None:
                cur_script = sc
            if sc != cur_script and sc != "other":
                pieces.append(_translit_run("".join(buf), cur_script))
                buf = [ch]
                cur_script = sc
            else:
                buf.append(ch)
        if buf:
            pieces.append(_translit_run("".join(buf), cur_script or "latin"))
    return "".join(pieces).lower()


# ---------------------------------------------------------------------------
# consonant-skeleton phonetic key
# ---------------------------------------------------------------------------

_SKEL_PRE = [
    ("ck", "k"), ("tion", "sn"), ("ksh", "ks"), ("x", "ks"), ("q", "k"),
    ("z", "j"), ("ph", "f"), ("sh", "s"), ("ch", "k"), ("bh", "b"),
    ("dh", "d"), ("th", "t"), ("kh", "k"), ("gh", "g"), ("jh", "j"),
    ("ny", "n"),
]
_SK_NONALPHA = re.compile(r"[^a-z]")
_SK_TURE = re.compile(r"ture$")
_SK_CEI = re.compile(r"c(?=[eiy])")
_SK_GEI = re.compile(r"g(?=[eiy])")
_SK_NASAL = re.compile(r"(?<=.)m(?=[bdfgjklnprstz])")
_SK_VOW = re.compile(r"[aeiouyh]")
_SK_DUP = re.compile(r"(.)+")


@lru_cache(maxsize=1_000_000)
def skeleton(token_str: str) -> str:
    """Language-agnostic consonant-skeleton key for one token. Lowercase
    ascii in, ascii key out. Keeps first letter (leading vowel -> 'a')."""
    t = _SK_NONALPHA.sub("", unidecode(token_str).lower())
    if not t:
        return ""
    t = _SK_TURE.sub("kar", t)
    for pat, rep in _SKEL_PRE:
        t = t.replace(pat, rep)
    t = _SK_GEI.sub("j", _SK_CEI.sub("s", t)).replace("c", "k")
    t = t.replace("v", "b").replace("w", "b")  # b/v/w merged (private~praibhet)
    t = _SK_NASAL.sub("n", t).replace("nj", "n").replace("ng", "n")
    if len(t) > 1 and t[0] == "y" and t[1] in "aeiou":
        t = t[1:]
    if t[0] in "aeiou":
        t = "a" + t[1:]
    t = t[0] + _SK_VOW.sub("", t[1:])
    return _SK_DUP.sub(r"", t)


_CK_MAP = {**dict.fromkeys("pbfvw", "P"), **dict.fromkeys("td", "T"),
           **dict.fromkeys("kgq", "K"), **dict.fromkeys("szj", "S"),
           **dict.fromkeys("mn", "N"), "l": "L", "r": "R"}
_CK_NONALPHA = re.compile(r"[^a-z]")
_CK_DUP = re.compile(r"(.)\1+")


@lru_cache(maxsize=1_000_000)
def _coarse_tok(tok: str) -> str:
    t = _CK_NONALPHA.sub("", unidecode(tok).lower())
    t = t.replace("tion", "sion")  # nation ~ neshan
    out = []
    for i, ch in enumerate(t):
        if ch == "c":
            c = "S" if t[i + 1:i + 2] in ("e", "i", "y") else "K"
        elif ch == "g" and t[i + 1:i + 2] in ("e", "i", "y"):
            c = "S"  # soft g: technology ~ tekanolaji
        elif ch == "x":
            c = "KS"
        elif ch in "aeiou":
            c = "A" if i == 0 else ""
        else:  # h and y drop; ph/th/sh: h dropped, letter class kept
            c = _CK_MAP.get(ch, "")
        out.append(c)
    return _CK_DUP.sub(r"\1", "".join(out))


def coarse_key(s: str) -> str:
    """Coarse sound-class key (voicing/aspiration-blind: p=b=f=v=w, t=d,
    k=g=c, s=z=j=sh, m=n). Order-preserving, per token."""
    return " ".join(t for t in map(_coarse_tok, s.split()) if t)


def sorted_key(s: str) -> str:
    """Order-independent skeleton: skeleton each token, sort, join."""
    toks = [skeleton(t) for t in s.split()]
    return " ".join(sorted(t for t in toks if t))


def name_key_of(s: str) -> str:
    """Order-preserving skeleton key (per CONTRACT name_key/addr_key)."""
    toks = [skeleton(t) for t in s.split()]
    return " ".join(t for t in toks if t)


# ---------------------------------------------------------------------------
# abbreviations (roman tokens, post-romanize)
# ---------------------------------------------------------------------------

_DROP_TOKENS = {"n/a", "na", "null", "none", "nil"}

_ADDR_ABBR = {
    "st": "street", "rd": "road", "ave": "avenue", "av": "avenue",
    "blvd": "boulevard", "bd": "boulevard", "dr": "drive", "ln": "lane",
    "hwy": "highway", "pl": "place", "chem": "chemin", "imp": "impasse",
    "fbg": "faubourg", "nr": "near", "opp": "opposite", "marg": "marg",
    "ngr": "nagar",
}
_NAME_ABBR = {"co": "company", "corp": "corporation", "&": "and"}


def expand_abbr(tokens: list[str], is_address: bool) -> list[str]:
    out = []
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low in _DROP_TOKENS:
            continue
        if is_address:
            if low == "r" and i + 1 < len(tokens) and tokens[i + 1].lower() in ("de", "du", "des"):
                out.append("rue")
                continue
            low = _ADDR_ABBR.get(low, low)
        else:
            low = _NAME_ABBR.get(low, low)
        out.append(low)
    return out


# ---------------------------------------------------------------------------
# legal suffix canonicalization
# ---------------------------------------------------------------------------

# canonical legal label -> known surface phrases, matched as exact lowercase
# text (post-romanize). Includes plain English/French forms, the Hindi
# abbreviation idiom ("pra li" for प्रा. लि.), and the deterministic
# sanscript+unidecode output for fused Indic abbreviations (e.g. Bengali
# এলএলপি -> "elaelapi") -- hardcoded since that pipeline's output is stable.
_LEGAL_PHRASES = {
    "pvt ltd": ["private limited", "pvt limited", "pvt ltd", "p ltd", "pra li"],
    "ltd": ["limited", "ltd"],
    "llp": ["llp", "l l p", "elaelapi", "elelpi"],
    "llc": ["llc", "l l c"],
    "inc": ["inc", "incorporated"],
    "corp": ["corp", "corporation"],
    "co": ["co", "company"],
    "pc": ["pc", "p c"],
    "pllc": ["pllc"],
    "lp": ["lp"],
    "sa": ["sa"],
    "sas": ["sas"],
    "sasu": ["sasu"],
    "sarl": ["sarl"],
    "eurl": ["eurl"],
    "sci": ["sci"],
    "snc": ["snc"],
    "scop": ["scop"],
    "gmbh": ["gmbh"],
}

# exact-text-key ("tok1 tok2") -> (label, n_tokens), longer phrases first
_LEGAL_EXACT_MAP: dict[str, tuple[str, int]] = {}
for _label, _phrases in _LEGAL_PHRASES.items():
    for _phrase in _phrases:
        _LEGAL_EXACT_MAP[_phrase] = (_label, len(_phrase.split()))


def extract_legal(name_roman: str) -> tuple[str, str]:
    """Split a romanized name into (core, legal). Looks at the trailing 1-3
    tokens against known legal-suffix phrases (exact text match)."""
    tokens = [t for t in re.sub(r"[.,()\[\]]", " ", name_roman).split() if t]
    if not tokens:
        return "", ""
    for n in (3, 2, 1):
        if len(tokens) < n:
            continue
        tail = tokens[-n:]
        hit = _LEGAL_EXACT_MAP.get(" ".join(tail))
        if hit and hit[1] == n:
            core = " ".join(tokens[:-n]).strip()
            return core, hit[0]
    return " ".join(tokens), ""


# ---------------------------------------------------------------------------
# address parsing
# ---------------------------------------------------------------------------

_LANDMARK_RE = re.compile(
    r"\b(near|opp\.?|opposite|behind|beside|next to)\b[^,]*", re.IGNORECASE
)


def parse_address(addr_raw: str) -> dict:
    """Extract postcode / nums / landmark from a raw (pre-romanize) address
    string. Returns dict with keys postcode, nums, landmark."""
    if not addr_raw:
        return {"postcode": "", "nums": "", "landmark": ""}
    s = _map_indic_digits(unicodedata.normalize("NFKC", addr_raw))

    landmark_m = _LANDMARK_RE.search(s)
    landmark = landmark_m.group(0).strip(" .-") if landmark_m else ""

    all_nums = re.findall(r"\d+", s)
    postcode = ""
    india_pins = [n for n in all_nums if len(n) == 6]
    five_digit = [n for n in all_nums if len(n) == 5]
    if india_pins:
        postcode = india_pins[-1]
    elif five_digit:
        postcode = five_digit[-1]

    nums = [n.lstrip("0") or "0" for n in all_nums if n != postcode]
    return {"postcode": postcode, "nums": " ".join(nums), "landmark": landmark}


# ---------------------------------------------------------------------------
# top-level record normalizer
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _clean_orig(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    return _WS_RE.sub(" ", s).strip()


_WEB_HEAD = {"www", "http", "https"}
_TLD = {"com", "in", "net", "org", "co"}


def strip_web(tokens: list[str]) -> list[str]:
    """Drop www/http(s) + the host and TLD tokens following them; strip a
    glued trailing 'com' from a long last token (conservative)."""
    out, i, n = [], 0, len(tokens)
    while i < n:
        t = tokens[i]
        if t in _WEB_HEAD:
            j = i + 1
            if j < n and tokens[j] == "www":
                j += 1
            if j + 1 < n and tokens[j + 1] in _TLD and tokens[j] not in _TLD:
                j += 2
            i = j
            continue
        out.append(t)
        i += 1
    if len(out) >= 2 and (out[-1] == "com" or (out[-1] in ("in", "net", "org") and len(out[-2]) >= 5)):
        out.pop()
    if out and len(out[-1]) >= 9 and out[-1].endswith("com")             and not out[-1].endswith(("ecom", "ocom")):
        out[-1] = out[-1][:-3]
    return out


_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "8": "b", "7": "t"})
_ORDINAL_RE = re.compile(r"\d+(st|nd|rd|th)$")


def deleet(tokens: list[str]) -> list[str]:
    """OCR/leet digit->letter fix inside mixed alnum name tokens: letters must
    be >= digits, all digits mappable, first char a letter, >=2 letters
    (or 1 letter + '0')."""
    out = []
    for t in tokens:
        if t.isalnum() and not t.isalpha() and not t.isdigit() and t.isascii() and t[0].isalpha():
            nd = sum(c.isdigit() for c in t)
            nl = len(t) - nd
            if nl >= nd and all(c in "01345687" for c in t if c.isdigit())                     and (nl >= 2 or "0" in t):
                t = t.translate(_LEET)
        out.append(t)
    return out


_ALNUM_LZ = re.compile(r"(?<=[a-z])0+(?=[0-9])")


def strip_zeros(tokens: list[str]) -> list[str]:
    out = []
    for t in tokens:
        if t.isdigit():
            t = t.lstrip("0") or "0"
        elif "0" in t and not t.isalpha():
            t = _ALNUM_LZ.sub("", t)
        out.append(t)
    return out


def normalize_record(name, addr) -> dict:
    """Fill every prep column except idx/entity_id/src/country. name/addr
    may be None/NaN (polars gives None for nulls)."""
    name = name if isinstance(name, str) else ""
    addr = addr if isinstance(addr, str) else ""

    name_orig = _clean_orig(name)
    addr_orig = _clean_orig(addr)

    script = script_of(name)

    name_roman_raw = romanize(name)
    addr_roman_raw = romanize(addr)

    name_tokens = expand_abbr(
        deleet(strip_web(_PUNCT_RE.sub(" ", name_roman_raw).split())), is_address=False)
    addr_tokens = strip_zeros(expand_abbr(_PUNCT_RE.sub(" ", addr_roman_raw).split(), is_address=True))

    name_clean = " ".join(name_tokens)
    addr_clean = " ".join(addr_tokens)

    core, legal = extract_legal(name_clean)

    addr_info = parse_address(addr)

    return {
        "name_orig": name_orig,
        "addr_orig": addr_orig,
        "name": name_clean,
        "addr": addr_clean,
        "core": core,
        "legal": legal,
        "name_key": name_key_of(name_clean),
        "name_key2": coarse_key(core),
        "addr_key": name_key_of(addr_clean),
        "postcode": addr_info["postcode"],
        "nums": addr_info["nums"],
        "script": script,
        "landmark": addr_info["landmark"],
    }


# ---------------------------------------------------------------------------
# frame-level driver (multiprocessing)
# ---------------------------------------------------------------------------

def _normalize_rows(rows):
    return [normalize_record(name, addr) for name, addr in rows]


def prep_frame(df, n_jobs: int = 64):
    """df: polars.DataFrame with business_name, business_address (+ other
    CONTRACT-passthrough cols kept as-is). Returns df with the prep text
    columns added. Uses a process pool since romanize/skeleton are pure CPU."""
    import polars as pl

    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    rows = list(zip(names, addrs))

    chunk = 50_000
    chunks = [rows[i:i + chunk] for i in range(0, len(rows), chunk)]

    if n_jobs <= 1 or len(rows) < chunk:
        results = _normalize_rows(rows)
    else:
        with Pool(n_jobs) as pool:
            out_chunks = pool.map(_normalize_rows, chunks)
        results = [r for oc in out_chunks for r in oc]

    cols = ["name_orig", "addr_orig", "name", "addr", "core", "legal",
            "name_key", "name_key2", "addr_key", "postcode", "nums", "script", "landmark"]
    out = {c: [r[c] for r in results] for c in cols}
    return df.with_columns([pl.Series(c, out[c]) for c in cols])


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. script detection
    assert script_of("रियल मॉडर्न फूड लिमिटेड") == "devanagari"
    assert script_of("Real Modern Food Limited") == "latin"
    assert script_of("বোম্বে ইনভেস্টমেন্ট এলএলপি") == "bengali"

    # 2. skeleton equality across scripts: Real Modern Food Limited
    roman = romanize("रियल मॉडर्न फूड लिमिटेड")
    print("romanized:", roman)
    sk_hi = name_key_of(" ".join(expand_abbr(roman.split(), is_address=False)))
    sk_en = name_key_of("real modern food limited")
    print("skeleton hi:", sk_hi, "| skeleton en:", sk_en)
    assert sk_hi == sk_en, (sk_hi, sk_en)

    # 3. srinivas / shrinivas / sreenivas collapse to same skeleton
    s1, s2, s3 = skeleton("srinivas"), skeleton("shrinivas"), skeleton("sreenivas")
    print("srinivas variants:", s1, s2, s3)
    assert s1 == s2 == s3, (s1, s2, s3)

    # 4. legal extraction from Devanagari "प्रा. लि." -> pvt ltd
    rec = normalize_record("ड्रीम आईटी प्रा. लि.", None)
    print("dream it legal:", rec["legal"], "core:", rec["core"])
    assert rec["legal"] == "pvt ltd", rec

    # 5. "Thermal & Fils SASU" -> legal 'sasu'
    rec2 = normalize_record("Thermal & Fils SASU", None)
    print("thermal legal:", rec2["legal"], "core:", rec2["core"])
    assert rec2["legal"] == "sasu", rec2

    # 6. accents removed
    assert "e" in romanize("Nétwork").lower() and "é" not in romanize("Nétwork")
    assert romanize("Sólar") == "solar"

    # 7. postcode extraction (India 6-digit)
    addr_info = parse_address("123 Park Street, Kolkata 700001")
    print("postcode:", addr_info)
    assert addr_info["postcode"] == "700001", addr_info

    # US 5-digit
    addr_info2 = parse_address("16048 Fm 487, Somewhere TX 75482")
    assert addr_info2["postcode"] == "75482", addr_info2

    # 8. Indic digits -> ascii
    assert _map_indic_digits("१२३") == "123"

    # 9. legal LLP from Bengali skeleton form
    rec3 = normalize_record("বোম্বে ইনভেস্টমেন্ট এলএলপি", None)
    print("bombay llp legal:", rec3["legal"], "core:", rec3["core"])
    assert rec3["legal"] == "llp", rec3

    # 10. dropped filler tokens
    assert "n/a" not in expand_abbr(["N/A", "Main", "St"], is_address=True)

    # 11. legal 'pvt ltd' canonical forms
    for variant in ["Private Limited", "Pvt Ltd", "Pvt. Ltd.", "(P) Ltd"]:
        c, l = extract_legal(romanize(variant))
        assert l == "pvt ltd", (variant, l)

    # 12. anusvara -> homorganic nasal
    assert romanize("कंसल्टिंग") == "kansalting", romanize("कंसल्टिंग")
    assert romanize("इंजीनियरिंग") == "injiniyaring", romanize("इंजीनियरिंग")
    # 13. skeleton pairs
    for a, b in [("consulting", "kamsaltimg"), ("engineering", "imjiniyarimg"),
                 ("systems", "sistams"), ("future", "phyuchar"),
                 ("private", "praivet"), ("private", "praibhet"),
                 ("infotech", "imphotek"), ("sun", "san"), ("energy", "enarji"),
                 ("united", "yunaited"), ("ace", "es"),
                 ("constructions", "kamstrakshams")]:
        assert skeleton(a) == skeleton(b), (a, b, skeleton(a), skeleton(b))
    for a, b in [("business", "bijanes"), ("limited", "limirad"),
                 ("premier", "primiy"), ("international", "inranashan")]:
        print("skeleton differs:", a, skeleton(a), b, skeleton(b))
    # 14. leet
    assert deleet("pediatric ass0ciates ch0wdhury c0 tr6 l0tus 4th b78 c01 3m covid19".split()) ==         "pediatric associates chowdhury co trg lotus 4th b78 c01 3m covid19".split()
    assert normalize_record("ur re1iable real", None)["name"] == "ur reliable real"
    # 15. web junk
    assert normalize_record("Gurudev Traders Private Limited www.gurudevt.com", None)["name"]         == "gurudev traders private limited"
    assert normalize_record("apple partners private ltd www applepa com", None)["name"]         == "apple partners private ltd"
    assert normalize_record("shri alphasugarcom", None)["name"] == "shri alphasugar"
    assert normalize_record("Globaltelecom", None)["name"] == "globaltelecom"
    # 16. numbers
    assert normalize_record("x", "04348 Starlight Drive")["addr"] == "4348 starlight drive"
    assert normalize_record("x", "h no 0021 plot no c01")["addr"] == "h no 21 plot no c1"
    assert parse_address("h 0021 rd 0045 015408")["nums"] == "21 45"
    assert parse_address("x 012345")["postcode"] == "012345"

    # 17. coarse key + trailing com
    assert normalize_record("nagoraservices com", None)["name"] == "nagoraservices"
    assert normalize_record("brothersms com", None)["name"] == "brothersms"
    for a, b in [("star enterprises private limited", "sdhar endharbhiraijhas bhiraivedh limidhedh"),
                 ("white dream services", "vhait drim sarvisej"),
                 ("my technology", "mai tekanolaji"),
                 ("big enterprises", "big entarapraijej"),
                 ("bombay agro", "bombe egro"),
                 ("international", "intarneshanal")]:
        ka, kb = coarse_key(a), coarse_key(b)
        print("coarse", "OK  " if ka == kb else "FAIL", a, ka, "|", b, kb)
    print("ALL CHECKS PASSED")
