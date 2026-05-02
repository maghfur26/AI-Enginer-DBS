"""
AE2 — Mock Artifacts Generator (Sprint 3)
Buat dummy artifacts agar AE2 bisa develop & test API
tanpa menunggu model AE1 selesai.

Usage:
    python scripts/create_mock_artifacts.py
"""

import json
import os
import struct
import sys


OUTPUT_DIR = "artifacts"


def create_label_map(output_dir: str) -> None:
    label_map = {
        "utama": 0,
        "substitusi": 1,
        "opsional": 2,
    }
    path = os.path.join(output_dir, "label_map.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"  ✅ label_map.json dibuat: {path}")


def create_mock_vocab(output_dir: str) -> None:
    """Buat vocab.json sederhana berisi bahan masakan Indonesia."""
    vocab = {
        "[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
        "bawang": 100, "merah": 101, "putih": 102, "telur": 103,
        "ayam": 104, "garam": 105, "gula": 106, "minyak": 107,
        "goreng": 108, "cabai": 109, "rawit": 110, "serai": 111,
        "jahe": 112, "tempe": 113, "tahu": 114, "nasi": 115,
        "santan": 116, "kecap": 117, "manis": 118, "asin": 119,
        "daun": 120, "bawang_putih": 121, "bawang_merah": 122,
    }
    path = os.path.join(output_dir, "vocab.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"  ✅ vocab.json dibuat: {path}")


def create_mock_tokenizer(output_dir: str) -> None:
    """Buat folder tokenizer dengan file minimal."""
    tok_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)

    # tokenizer_config.json
    config = {
        "tokenizer_class": "BertTokenizer",
        "do_lower_case": True,
        "model_max_length": 64,
    }
    with open(os.path.join(tok_dir, "tokenizer_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # vocab.txt minimal (BERT format)
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    common_words = [
        "bawang", "merah", "putih", "telur", "ayam", "garam", "gula",
        "minyak", "goreng", "cabai", "rawit", "serai", "jahe", "tempe",
        "tahu", "nasi", "santan", "kecap", "manis", "asin", "daun",
        "kentang", "wortel", "tomat", "lengkuas", "kunyit", "ketumbar",
    ]
    vocab_lines = special_tokens + common_words
    with open(os.path.join(tok_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(vocab_lines) + "\n")

    # special_tokens_map.json
    special_map = {
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "mask_token": "[MASK]",
    }
    with open(os.path.join(tok_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special_map, f, indent=2)

    print(f"  ✅ tokenizer/ dibuat: {tok_dir}")


def create_mock_model_placeholder(output_dir: str) -> None:
    """
    Buat file model.pt placeholder.
    File ini akan diganti dengan model asli dari AE1 saat handoff.
    """
    placeholder_path = os.path.join(output_dir, "model.pt")
    # Tulis file kecil sebagai marker
    with open(placeholder_path, "wb") as f:
        f.write(b"MOCK_MODEL_PLACEHOLDER_CC26PSU127")
    print(f"  ✅ model.pt placeholder dibuat: {placeholder_path}")
    print(f"     ⚠️  File ini akan diganti model AE1 saat Handoff 1 (~14 Mei)")


def create_sample_prediction(output_dir: str) -> None:
    """Buat contoh_prediksi.json sesuai format handoff AE1."""
    sample = {
        "_note": "Contoh output prediksi model — digunakan AE2 untuk validasi format",
        "examples": [
            {
                "ingredient": "bawang merah",
                "label": "utama",
                "confidence": 0.92,
                "all_scores": {"utama": 0.92, "substitusi": 0.05, "opsional": 0.03},
            },
            {
                "ingredient": "daun bawang",
                "label": "opsional",
                "confidence": 0.84,
                "all_scores": {"utama": 0.06, "substitusi": 0.10, "opsional": 0.84},
            },
            {
                "ingredient": "telur",
                "label": "utama",
                "confidence": 0.89,
                "all_scores": {"utama": 0.89, "substitusi": 0.08, "opsional": 0.03},
            },
        ],
    }
    path = os.path.join(output_dir, "contoh_prediksi.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"  ✅ contoh_prediksi.json dibuat: {path}")


def main():
    print(f"\n{'=' * 55}")
    print("  🛠  OLAH — Mock Artifacts Generator")
    print("  AE2 · CC26-PSU127 · Sprint 3 Dev Mode")
    print(f"{'=' * 55}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output dir: {os.path.abspath(OUTPUT_DIR)}\n")

    create_label_map(OUTPUT_DIR)
    create_mock_vocab(OUTPUT_DIR)
    create_mock_tokenizer(OUTPUT_DIR)
    create_mock_model_placeholder(OUTPUT_DIR)
    create_sample_prediction(OUTPUT_DIR)

    print(f"\n{'=' * 55}")
    print("  ✅ Mock artifacts siap!")
    print("  API akan berjalan dalam MOCK mode.")
    print("  Jalankan: uvicorn api.main:app --reload")
    print(f"{'=' * 55}\n")


if __name__ == "__main__":
    main()
