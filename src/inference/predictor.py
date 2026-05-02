"""
AE2 — Inference Module
Proyek OLAH · CC26-PSU127 · Coding Camp 2026
Handles loading BERT model dari AE1 handoff dan menjalankan prediksi klasifikasi bahan.
"""

import json
import logging
import os
from typing import Any

import numpy as np
import torch
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class IngredientPredictor:
    """
    Modul utama inference untuk klasifikasi bahan makanan.
    Menerima model handoff dari AE1 (model.pt + tokenizer + label_map.json).
    Mengklasifikasikan bahan ke kategori: utama, substitusi, opsional.
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.tokenizer = None
        self.label_map = None
        self.id_to_label = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_loaded = False

    def load(self) -> None:
        """Load semua artefak dari AE1 handoff."""
        try:
            # Load label map
            label_map_path = os.path.join(self.artifacts_dir, "label_map.json")
            with open(label_map_path, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)
            self.id_to_label = {v: k for k, v in self.label_map.items()}
            logger.info(f"Label map loaded: {list(self.label_map.keys())}")

            # Load tokenizer dari AE1
            tokenizer_path = os.path.join(self.artifacts_dir, "tokenizer")
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
            logger.info("Tokenizer loaded.")

            # Load model — cek placeholder dulu sebelum torch.load
            model_path = os.path.join(self.artifacts_dir, "model.pt")
            with open(model_path, "rb") as f:
                header = f.read(32)
            if b"MOCK_MODEL_PLACEHOLDER" in header:
                logger.warning(
                    "model.pt adalah placeholder mock — menunggu handoff AE1. "
                    "Running in MOCK mode."
                )
                self._is_loaded = False
                return

            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"Model loaded on device: {self.device}")

            self._is_loaded = True

        except FileNotFoundError as e:
            logger.warning(f"Artifact not found: {e}. Running in mock mode.")
            self._is_loaded = False

    def is_ready(self) -> bool:
        return self._is_loaded

    def preprocess(self, ingredient: str) -> dict[str, Any]:
        """Tokenisasi input bahan untuk BERT."""
        encoding = self.tokenizer(
            ingredient,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    def predict_single(self, ingredient: str) -> dict[str, Any]:
        """
        Prediksi kategori satu bahan.
        Returns dict: { ingredient, label, confidence, all_scores }
        """
        if not self._is_loaded:
            return self._mock_predict(ingredient)

        inputs = self.preprocess(ingredient)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            pred_id = int(np.argmax(probs))

        label = self.id_to_label.get(pred_id, "unknown")
        confidence = float(probs[pred_id])
        all_scores = {self.id_to_label[i]: float(probs[i]) for i in range(len(probs))}

        return {
            "ingredient": ingredient,
            "label": label,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
        }

    def predict_batch(self, ingredients: list[str]) -> list[dict[str, Any]]:
        """Prediksi batch bahan dari satu resep."""
        return [self.predict_single(ing) for ing in ingredients]

    def classify_recipe_ingredients(
        self, ingredients: list[str]
    ) -> dict[str, Any]:
        """
        Klasifikasi seluruh bahan resep dan kelompokkan per kategori.
        Output format yang disepakati dengan AE1 & Backend.
        """
        predictions = self.predict_batch(ingredients)

        result: dict[str, list] = {
            "utama": [],
            "substitusi": [],
            "opsional": [],
        }

        for pred in predictions:
            label = pred["label"]
            if label in result:
                result[label].append(
                    {
                        "name": pred["ingredient"],
                        "confidence": pred["confidence"],
                    }
                )
            else:
                result["opsional"].append(
                    {
                        "name": pred["ingredient"],
                        "confidence": pred["confidence"],
                    }
                )

        return {
            "classified": result,
            "raw_predictions": predictions,
            "total_ingredients": len(ingredients),
        }

    def _mock_predict(self, ingredient: str) -> dict[str, Any]:
        """
        Mock response saat model belum tersedia (Sprint 3).
        AE2 dapat develop API paralel tanpa menunggu AE1.
        """
        import random

        labels = ["utama", "substitusi", "opsional"]
        # Heuristic sederhana untuk mock yang lebih realistis
        if any(
            k in ingredient.lower()
            for k in ["bawang", "garam", "gula", "minyak", "air"]
        ):
            label = "utama"
        elif any(k in ingredient.lower() for k in ["daun", "merica", "kaldu"]):
            label = "opsional"
        else:
            label = random.choice(labels)

        confidence = round(random.uniform(0.65, 0.95), 4)
        remaining = 1 - confidence
        other_labels = [l for l in labels if l != label]

        return {
            "ingredient": ingredient,
            "label": label,
            "confidence": confidence,
            "all_scores": {
                label: confidence,
                other_labels[0]: round(remaining * 0.6, 4),
                other_labels[1]: round(remaining * 0.4, 4),
            },
            "_mock": True,
        }
