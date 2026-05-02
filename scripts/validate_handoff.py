"""
AE2 — Handoff Validator
Script validasi artefak dari AE1 sebelum digunakan untuk production.
Dijalankan setelah menerima handoff dari AE1 (Sprint 3 & Sprint 4).

Usage:
    python scripts/validate_handoff.py --artifacts-dir artifacts/
    python scripts/validate_handoff.py --artifacts-dir artifacts/ --version v1
"""

import argparse
import json
import os
import sys


REQUIRED_FILES = {
    "model.pt": "PyTorch model weights",
    "tokenizer/tokenizer_config.json": "Tokenizer config",
    "tokenizer/vocab.txt": "BERT vocabulary",
    "label_map.json": "Label mapping (utama/substitusi/opsional)",
}

REQUIRED_LABELS = {"utama", "substitusi", "opsional"}


def check_file_exists(path: str, description: str) -> bool:
    exists = os.path.isfile(path)
    status = "✅" if exists else "❌"
    size = f"({os.path.getsize(path) / 1024:.1f} KB)" if exists else ""
    print(f"  {status} {description}: {path} {size}")
    return exists


def validate_label_map(label_map_path: str) -> bool:
    try:
        with open(label_map_path, "r") as f:
            label_map = json.load(f)

        labels = set(label_map.keys())
        print(f"  📋 Label map keys: {labels}")

        if not REQUIRED_LABELS.issubset(labels):
            missing = REQUIRED_LABELS - labels
            print(f"  ❌ Label yang kurang: {missing}")
            return False

        print(f"  ✅ Semua label tersedia: {REQUIRED_LABELS}")
        return True
    except Exception as e:
        print(f"  ❌ Gagal baca label_map.json: {e}")
        return False


def validate_model_loadable(model_path: str) -> bool:
    try:
        import torch
        model = torch.load(model_path, map_location="cpu")
        print(f"  ✅ Model berhasil di-load (type: {type(model).__name__})")
        return True
    except ImportError:
        print("  ⚠️  torch tidak terinstall — skip model load test.")
        return True
    except Exception as e:
        print(f"  ❌ Model gagal di-load: {e}")
        return False


def validate_tokenizer(tokenizer_dir: str) -> bool:
    try:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        # Test tokenisasi sederhana
        encoded = tokenizer("bawang merah", return_tensors="pt")
        print(f"  ✅ Tokenizer berfungsi. Input IDs shape: {encoded['input_ids'].shape}")
        return True
    except ImportError:
        print("  ⚠️  transformers tidak terinstall — skip tokenizer test.")
        return True
    except Exception as e:
        print(f"  ❌ Tokenizer gagal: {e}")
        return False


def run_validation(artifacts_dir: str, version: str = "") -> bool:
    label = f" [{version}]" if version else ""
    print(f"\n{'=' * 55}")
    print(f"  🔍 OLAH Handoff Validator{label}")
    print(f"  AE2 — CC26-PSU127")
    print(f"  Artifacts dir: {artifacts_dir}")
    print(f"{'=' * 55}\n")

    all_passed = True

    # 1. Cek keberadaan file
    print("1️⃣  Cek file artefak:")
    for filename, desc in REQUIRED_FILES.items():
        full_path = os.path.join(artifacts_dir, filename)
        if not check_file_exists(full_path, desc):
            all_passed = False

    # 2. Validasi label_map.json
    print("\n2️⃣  Validasi label_map.json:")
    label_map_path = os.path.join(artifacts_dir, "label_map.json")
    if os.path.isfile(label_map_path):
        if not validate_label_map(label_map_path):
            all_passed = False
    else:
        print("  ⏭  Skip (file tidak ada)")

    # 3. Load model
    print("\n3️⃣  Validasi model.pt dapat di-load:")
    model_path = os.path.join(artifacts_dir, "model.pt")
    if os.path.isfile(model_path):
        if not validate_model_loadable(model_path):
            all_passed = False
    else:
        print("  ⏭  Skip (file tidak ada)")

    # 4. Test tokenizer
    print("\n4️⃣  Validasi tokenizer:")
    tokenizer_dir = os.path.join(artifacts_dir, "tokenizer")
    if os.path.isdir(tokenizer_dir):
        if not validate_tokenizer(tokenizer_dir):
            all_passed = False
    else:
        print("  ⏭  Skip (folder tidak ada)")

    # Hasil akhir
    print(f"\n{'=' * 55}")
    if all_passed:
        print("  🎉 HANDOFF VALID — AE2 siap integrasi!")
    else:
        print("  ⚠️  HANDOFF BELUM LENGKAP — Koordinasi dengan AE1.")
    print(f"{'=' * 55}\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Validasi artefak handoff dari AE1 ke AE2."
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Path ke folder artifacts (default: artifacts/)",
    )
    parser.add_argument(
        "--version",
        default="",
        help="Versi handoff (e.g. v1, v2-final)",
    )
    args = parser.parse_args()

    success = run_validation(args.artifacts_dir, args.version)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
