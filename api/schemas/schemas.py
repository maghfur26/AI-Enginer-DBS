"""
AE2 — API Schemas (Pydantic)
Request & Response models yang disepakati dengan AE1 dan Backend (Full-Stack).
Format JSON ini adalah kontrak handoff resmi antar tim.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Health ─────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mode: str  # "production" | "mock"


# ── Single Classify ────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    ingredient: Optional[str] = Field(None, example="telor ayam")
    ingredients: Optional[list[str]] = Field(
        None, example=["bawang merah", "cabe", "telor"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"ingredient": "telor ayam"},
                {"ingredients": ["bawang merah", "cabe rawit", "telor", "garam"]},
            ]
        }
    }


class ClassifyResponse(BaseModel):
    ingredient_raw: str
    ingredient_normalized: str
    label: str = Field(..., description="utama | substitusi | opsional")
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_scores: dict[str, float]
    mock: bool = Field(False, description="True jika menggunakan dummy response")


# ── Recipe Classify ────────────────────────────────────────────────────────

class RecipeClassifyRequest(BaseModel):
    recipe_id: str = Field(..., example="resep_001")
    recipe_name: Optional[str] = Field(None, example="Nasi Goreng Spesial")
    ingredients: list[str] = Field(
        ...,
        min_length=1,
        example=["bawang merah", "bawang putih", "cabe rawit", "telor",
                 "kecap manis", "garam", "minyak goreng", "nasi", "daun bawang"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "recipe_id": "resep_001",
                    "recipe_name": "Nasi Goreng Spesial",
                    "ingredients": [
                        "bawang merah", "bawang putih", "cabe rawit",
                        "telor", "kecap manis", "garam", "minyak goreng",
                        "nasi", "daun bawang",
                    ],
                }
            ]
        }
    }


class ClassifiedIngredients(BaseModel):
    utama: list[dict[str, Any]]
    substitusi: list[dict[str, Any]]
    opsional: list[dict[str, Any]]


class RecipeClassifyResponse(BaseModel):
    recipe_id: str
    recipe_name: Optional[str]
    classified: ClassifiedIngredients
    raw_predictions: list[dict[str, Any]]
    total_ingredients: int
    normalization_log: list[dict[str, Any]]
    mock: bool = False

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "recipe_id": "resep_001",
                    "recipe_name": "Nasi Goreng Spesial",
                    "classified": {
                        "utama": [
                            {"name": "nasi", "confidence": 0.93},
                            {"name": "telur", "confidence": 0.91},
                            {"name": "minyak goreng", "confidence": 0.88},
                        ],
                        "substitusi": [
                            {"name": "bawang merah", "confidence": 0.76},
                        ],
                        "opsional": [
                            {"name": "daun bawang", "confidence": 0.82},
                            {"name": "kecap manis", "confidence": 0.79},
                        ],
                    },
                    "total_ingredients": 9,
                    "mock": False,
                }
            ]
        }
    }


# ── Normalize ──────────────────────────────────────────────────────────────

class NormalizeRequest(BaseModel):
    ingredients: list[str] = Field(
        ..., example=["telor", "cabe rawit", "2 sdm minyak goreng"]
    )


class NormalizeResponse(BaseModel):
    results: list[dict[str, Any]]
    count: int
