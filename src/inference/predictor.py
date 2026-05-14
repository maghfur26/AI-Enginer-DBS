"""
AE2 — Inference Module
Proyek OLAH · CC26-PSU127 · Coding Camp 2026
Framework: TensorFlow (sesuai kesepakatan dengan AE1)
Format model yang didukung:
  - artifacts/saved_model/   → TF SavedModel format (direkomendasikan)
  - artifacts/model.h5       → Keras HDF5 format (alternatif)
"""

import json
import logging
import os
from typing import Any
from transformers import BertTokenizer
import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


class IngredientPredictor:
    MODEL_CANDIDATES = [
        ("saved_model", "dir"),
        ("model.h5", "file"),
        ("model", "dir"),
    ]
    PLACEHOLDER_MARKER = b"MOCK_MODEL_PLACEHOLDER"

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.tokenizer = None
        self.label_map = None
        self.id_to_label = None
        self._is_loaded = False
        self._model_format = None

    def load(self) -> None:
        try:
            self._load_label_map()
            self._load_tokenizer()

            model_path, model_format = self._detect_model()
            if model_path is None:
                logger.warning("Tidak ada model AE1 ditemukan. Running in MOCK mode.")
                self._is_loaded = False
                return

            if self._is_placeholder(model_path, model_format):
                logger.warning(
                    f"{model_path} adalah placeholder mock — menunggu handoff AE1. "
                    "Running in MOCK mode."
                )
                self._is_loaded = False
                return

            self._load_model(model_path, model_format)
            self._is_loaded = True
            logger.info(f"Model loaded ({model_format}): {model_path}")

        except FileNotFoundError as e:
            logger.warning(f"Artifact tidak ditemukan: {e}. Running in MOCK mode.")
            self._is_loaded = False
        except Exception as e:
            logger.error(f"Gagal load model: {e}. Running in MOCK mode.")
            self._is_loaded = False

    def _load_label_map(self) -> None:
        path = os.path.join(self.artifacts_dir, "label_map.json")
        with open(path, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        logger.info(f"Label map loaded: {list(self.label_map.keys())}")

    def _load_tokenizer(self) -> None:
        tokenizer_path = os.path.join(self.artifacts_dir, "tokenizer")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        logger.info("Tokenizer loaded.")

    def _detect_model(self) -> tuple[str | None, str | None]:
        for name, kind in self.MODEL_CANDIDATES:
            full_path = os.path.join(self.artifacts_dir, name)
            if kind == "dir" and os.path.isdir(full_path):
                return full_path, "saved_model"
            elif kind == "file" and os.path.isfile(full_path):
                return full_path, "h5"
        return None, None

    def _is_placeholder(self, path: str, fmt: str) -> bool:
        try:
            if fmt == "h5":
                with open(path, "rb") as f:
                    return self.PLACEHOLDER_MARKER in f.read(64)
            elif fmt == "saved_model":
                pb_path = os.path.join(path, "saved_model.pb")
                if not os.path.isfile(pb_path):
                    return True
                with open(pb_path, "rb") as f:
                    return self.PLACEHOLDER_MARKER in f.read(64)
        except Exception:
            pass
        return False

    def _load_model(self, path: str, fmt: str) -> None:
        if fmt == "saved_model":
            self.model = tf.saved_model.load(path)
            self._model_format = "saved_model"
        elif fmt == "h5":
            self.model = tf.keras.models.load_model(path)
            self._model_format = "h5"
        logger.info(f"Model TensorFlow loaded (format: {fmt})")

    def is_ready(self) -> bool:
        return self._is_loaded

    def preprocess(self, ingredient: str) -> dict[str, Any]:
        encoding = self.tokenizer(
            ingredient,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )
        return dict(encoding)

    def predict_single(self, ingredient: str) -> dict[str, Any]:
        if not self._is_loaded:
            return self._mock_predict(ingredient)

        inputs = self.preprocess(ingredient)

        if self._model_format == "saved_model":
            outputs = self.model(**inputs)
        else:
            outputs = self.model(inputs, training=False)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = tf.nn.softmax(logits, axis=-1).numpy().squeeze()
        pred_id = int(np.argmax(probs))

        label = self.id_to_label.get(pred_id, "unknown")
        confidence = float(probs[pred_id])
        all_scores = {
            self.id_to_label[i]: round(float(probs[i]), 4)
            for i in range(len(probs))
        }

        return {
            "ingredient": ingredient,
            "label": label,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
        }

    def predict_batch(self, ingredients: list[str]) -> list[dict[str, Any]]:
        return [self.predict_single(ing) for ing in ingredients]

    def classify_recipe_ingredients(self, ingredients: list[str]) -> dict[str, Any]:
        predictions = self.predict_batch(ingredients)

        result: dict[str, list] = {"utama": [], "substitusi": [], "opsional": []}

        for pred in predictions:
            label = pred["label"]
            bucket = label if label in result else "opsional"
            result[bucket].append({
                "name": pred["ingredient"],
                "confidence": pred["confidence"],
            })

        return {
            "classified": result,
            "raw_predictions": predictions,
            "total_ingredients": len(ingredients),
        }

    def _mock_predict(self, ingredient: str) -> dict[str, Any]:
        import random

        labels = ["utama", "substitusi", "opsional"]
        utama_keywords = ["bawang", "garam", "gula", "minyak", "air", "nasi", "ayam", "daging", "telur", "tahu", "tempe"]
        opsional_keywords = ["daun", "merica", "kaldu", "penyedap", "serai","lengkuas", "kunyit", "ketumbar"]

        ing_lower = ingredient.lower()
        if any(k in ing_lower for k in utama_keywords):
            label = "utama"
        elif any(k in ing_lower for k in opsional_keywords):
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