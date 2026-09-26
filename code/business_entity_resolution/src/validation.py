import os
from pathlib import Path
import re
import unicodedata
import ftfy
import polars as pl

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
VAL_DIR.mkdir(parents=True, exist_ok=True)

LEGAL_SUFFIX_REGEX = re.compile(
    r"\b("
    r"pvt\s+ltd|private\s+limited|public\s+limited|pvt|ltd|limited|"
    r"inc|incorporated|corp|corporation|llc|llp|pc|co|company|"
    r"sarl|sasu|sas|sa|eurl|sci|snc|gie"
    r")\b",
    re.IGNORECASE,
)

ADDR_REPLACEMENTS = {
    r"\brd\b\.?": "road",
    r"\bst\b\.?": "street",
    r"\bave\b\.?": "avenue",
    r"\bblvd\b\.?": "boulevard",
    r"\bdr\b\.?": "drive",
    r"\bln\b\.?": "lane",
    r"\bct\b\.?": "court",
    r"\bflr?\b\.?": "floor",
    r"\bapt\b\.?": "apartment",
    r"\bopp\b\.?": "opposite",
    r"\bnr\b\.?": "near",
    r"\br\.\b": "rue",
    r"\bbd\b\.?": "boulevard",
    r"\bste\b\.?": "suite",
}

RE_DIGITS = re.compile(r"\b\d+\b")
RE_PIN = re.compile(r"\b\d{5,6}\b")


def is_non_latin(text: str) -> bool:
    """Detects Indic, Cyrillic, or non-Latin alphabets."""
    return any(ord(c) > 0x0590 and c.isalpha() for c in str(text or ""))


def extract_pin(text: str) -> str:
    """Extracts 5-digit US ZIP code or 6-digit Indian PIN code."""
    m = RE_PIN.findall(str(text or ""))
    return m[-1] if m else ""


def extract_primary_bldg(text: str) -> str:
    """Extracts the first building / door / plot number."""
    m = RE_DIGITS.findall(str(text or ""))
    for d in m:
        if len(d) not in (5, 6):  # Skip postal codes
            return d
    return m[0] if m else ""


def clean_name(text: str) -> str:
    if text is None or not isinstance(text, str):
        return ""

    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()

    # Expand Common symbols
    text = text.replace("&", " and ").replace("+", " plus ").replace("@", " at ")

    # Remove content in parenthesis and brackets
    text = re.sub(r"[\[\(\{][^\]\)\}]+[\]\)\}]", " ", text)

    # Strip legal suffixes
    text = LEGAL_SUFFIX_REGEX.sub(" ", text)

    # Clean alphanumeric characters while keeping words & digits
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_address(text: str) -> str:
    if text is None or not isinstance(text, str):
        return ""
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()

    for pattern, repl in ADDR_REPLACEMENTS.items():
        text = re.sub(pattern, repl, text)

    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_digits(text: str) -> list[str]:
    if not text:
        return []
    return RE_DIGITS.findall(str(text or ""))