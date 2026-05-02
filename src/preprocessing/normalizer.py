"""
AE2 — Ingredient Normalizer
Normalisasi nama bahan: sinonim, typo, multi-kata.
Kamus sinonim dibangun sesuai jobdesk AE2 Sprint 3.
"""

import re
import unicodedata
from functools import lru_cache

# ── Kamus Sinonim Bahan Masakan Indonesia ──────────────────────────────────
SYNONYM_MAP: dict[str, str] = {
    # Telur
    "telor": "telur",
    "telor ayam": "telur ayam",
    "telor bebek": "telur bebek",
    # Cabai
    "cabe": "cabai",
    "cabe merah": "cabai merah",
    "cabe rawit": "cabai rawit",
    "lombok": "cabai",
    # Bawang
    "bawang bombai": "bawang bombay",
    "bawang bombay": "bawang bombay",
    "brambang": "bawang merah",
    # Kecap
    "kecap manis": "kecap manis",
    "kecap asin": "kecap asin",
    "kecap": "kecap manis",
    # Minyak
    "minyak sayur": "minyak goreng",
    "minyak kelapa": "minyak goreng",
    "minyak nabati": "minyak goreng",
    # Santan
    "santen": "santan",
    "santan kelapa": "santan",
    # Tepung
    "tepung terigu": "tepung terigu",
    "terigu": "tepung terigu",
    "tepung sagu": "tepung sagu",
    "sagu": "tepung sagu",
    # Gula
    "gula pasir": "gula",
    "gula putih": "gula",
    "gula merah": "gula merah",
    "gula jawa": "gula merah",
    "gula aren": "gula merah",
    # Daging
    "daging sapi": "daging sapi",
    "daging ayam": "ayam",
    "ayam": "ayam",
    "chicken": "ayam",
    # Tomat
    "tomat merah": "tomat",
    # Jahe
    "jahe merah": "jahe merah",
    # Kemiri
    "kemiri": "kemiri",
    # Serai
    "sereh": "serai",
    "lemongrass": "serai",
    # Terasi
    "belacan": "terasi",
    "trassi": "terasi",
    # Nasi
    "beras": "beras",
    "nasi": "nasi",
    # Tahu & Tempe
    "tahu putih": "tahu",
    "tempe": "tempe",
    # Daun-daunan
    "daun jeruk purut": "daun jeruk",
    "daun salam": "daun salam",
    "daun pandan": "daun pandan",
    # Lainnya
    "vetsin": "penyedap rasa",
    "msg": "penyedap rasa",
    "royco": "penyedap rasa",
    "masako": "penyedap rasa",
    "bumbu penyedap": "penyedap rasa",
    "air secukupnya": "air",
    "garam secukupnya": "garam",
}

# ── Stopword angka & satuan yang dibuang ──────────────────────────────────
UNIT_PATTERN = re.compile(
    r"\b(\d+[\.,]?\d*)\s*(gram|gr|g|kg|ml|liter|l|sdt|sdm|"
    r"butir|lembar|batang|siung|buah|biji|genggam|secukupnya|"
    r"iris|potong|helai|cup|cc|ons)\b",
    re.IGNORECASE,
)


def normalize_unicode(text: str) -> str:
    """Hapus karakter unicode non-standar."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def remove_quantities(text: str) -> str:
    """Hapus angka dan satuan ukuran dari nama bahan."""
    text = UNIT_PATTERN.sub("", text)
    text = re.sub(r"\b\d+\b", "", text)
    return text


def clean_text(text: str) -> str:
    """Pipeline cleaning dasar."""
    text = text.lower().strip()
    text = remove_quantities(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s]", "", text)  # hapus tanda baca
    return text


@lru_cache(maxsize=512)
def normalize_ingredient(raw: str) -> str:
    """
    Normalisasi satu nama bahan:
    1. Lowercase + strip
    2. Hapus satuan/angka
    3. Lookup kamus sinonim
    4. Return bentuk baku
    """
    cleaned = clean_text(raw)

    # Exact match dulu
    if cleaned in SYNONYM_MAP:
        return SYNONYM_MAP[cleaned]

    # Partial match: cari sinonim yang ada di dalam teks
    for alias, canonical in SYNONYM_MAP.items():
        if alias in cleaned:
            cleaned = cleaned.replace(alias, canonical)

    return cleaned.strip()


def normalize_ingredients_list(ingredients: list[str]) -> list[dict]:
    """
    Normalisasi list bahan dari satu resep.
    Returns list of { raw, normalized, changed }.
    """
    results = []
    for raw in ingredients:
        normalized = normalize_ingredient(raw)
        results.append(
            {
                "raw": raw,
                "normalized": normalized,
                "changed": raw.lower().strip() != normalized,
            }
        )
    return results
