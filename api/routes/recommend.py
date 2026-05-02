"""
AE2 — Recommendation Router
Endpoint rekomendasi resep untuk sistem OLAH.
Diintegrasi ke api/main.py sebagai router terpisah.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Optional

from src.inference.predictor import IngredientPredictor
from src.inference.recommender import RecipeRecommender
from src.preprocessing.normalizer import normalize_ingredients_list

router = APIRouter(prefix="/api/v1/recommend", tags=["Recommendation"])

# Instance recommender (recipes di-load dari DB / file)
recommender = RecipeRecommender()


# ── Schemas ────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    user_ingredients: list[str] = Field(
        ...,
        min_length=1,
        example=["bawang merah", "telor", "minyak goreng", "nasi", "garam"],
    )
    top_n: int = Field(5, ge=1, le=20, description="Jumlah resep yang dikembalikan")
    min_score: float = Field(
        0.3, ge=0.0, le=1.0, description="Skor minimum kecocokan bahan"
    )
    prefer_can_cook: bool = Field(
        True, description="Prioritaskan resep yang bisa langsung dimasak"
    )


class RecommendResponse(BaseModel):
    user_ingredients_raw: list[str]
    user_ingredients_normalized: list[str]
    total_recipes_checked: int
    recommendations: list[dict[str, Any]]


class LoadRecipesRequest(BaseModel):
    recipes: list[dict[str, Any]] = Field(
        ...,
        description="List resep dengan format: {recipe_id, recipe_name, classified: {utama, substitusi, opsional}}",
    )


# ── Endpoints ──────────────────────────────────────────────────────────────

@router.post("/", response_model=RecommendResponse)
async def get_recommendations(body: RecommendRequest):
    """
    Endpoint utama rekomendasi resep OLAH.

    Alur:
    1. Normalisasi bahan user (telor → telur, cabe → cabai)
    2. Hitung skor kecocokan tiap resep di database
    3. Kembalikan top-N resep terbaik

    Digunakan oleh backend Node.js → frontend React.
    """
    if not recommender.recipes_db:
        raise HTTPException(
            status_code=503,
            detail="Database resep belum di-load. Hubungi backend untuk POST /recipes/load.",
        )

    # Normalisasi input user
    norm_results = normalize_ingredients_list(body.user_ingredients)
    normalized_names = [r["normalized"] for r in norm_results]

    # Rekomendasi
    results = recommender.recommend_as_dict(
        user_ingredients=normalized_names,
        top_n=body.top_n,
        min_score=body.min_score,
    )

    return RecommendResponse(
        user_ingredients_raw=body.user_ingredients,
        user_ingredients_normalized=normalized_names,
        total_recipes_checked=len(recommender.recipes_db),
        recommendations=results,
    )


@router.post("/load-recipes", status_code=201)
async def load_recipes(body: LoadRecipesRequest):
    """
    Load/update database resep ke recommender.
    Dipanggil oleh backend saat startup atau setelah update dataset.
    Format recipes harus menyertakan hasil klasifikasi AE1 (classified).
    """
    recommender.load_recipes(body.recipes)
    return {
        "message": f"{len(body.recipes)} resep berhasil dimuat.",
        "total": len(body.recipes),
    }


@router.get("/status")
async def recommender_status():
    """Status recommender — jumlah resep yang sudah di-load."""
    return {
        "recipes_loaded": len(recommender.recipes_db),
        "ready": len(recommender.recipes_db) > 0,
    }
