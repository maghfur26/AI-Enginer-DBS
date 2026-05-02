"""
AE2 — Recipe Recommender
Sistem rekomendasi resep berbasis ketersediaan bahan pengguna.
Menggunakan hasil klasifikasi BERT (AE1) untuk scoring adaptif.

Alur:
  user_ingredients → normalize → classify (BERT) → score recipes → top-N
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RecipeScore:
    recipe_id: str
    recipe_name: str
    match_score: float          # 0.0 – 1.0
    matched_utama: list[str]    # bahan utama yang cocok
    missing_utama: list[str]    # bahan utama yang tidak dimiliki user
    matched_substitusi: list[str]
    matched_opsional: list[str]
    total_ingredients: int
    can_cook: bool              # True jika semua bahan utama tersedia


# ── Bobot per kategori bahan ───────────────────────────────────────────────
WEIGHT_UTAMA = 1.0
WEIGHT_SUBSTITUSI = 0.5
WEIGHT_OPSIONAL = 0.2


class RecipeRecommender:
    """
    Menghitung skor kecocokan antara bahan user dengan resep-resep di database.
    Mengintegrasikan hasil klasifikasi AE1 model untuk scoring yang adaptif.
    """

    def __init__(self, recipes_db: list[dict] | None = None):
        """
        Args:
            recipes_db: List resep dari database.
                        Setiap resep: { recipe_id, recipe_name, classified: {utama, substitusi, opsional} }
        """
        self.recipes_db = recipes_db or []

    def load_recipes(self, recipes: list[dict]) -> None:
        """Load/update database resep."""
        self.recipes_db = recipes
        logger.info(f"Loaded {len(recipes)} recipes into recommender.")

    def _normalize_set(self, ingredients: list[str]) -> set[str]:
        """Normalisasi list bahan menjadi lowercase set."""
        return {ing.lower().strip() for ing in ingredients}

    def score_recipe(
        self,
        user_ingredients: set[str],
        recipe_classified: dict[str, list[dict]],
    ) -> tuple[float, dict]:
        """
        Hitung skor satu resep terhadap bahan user.

        Returns:
            (score, detail_dict)
        """
        utama = [i["name"] for i in recipe_classified.get("utama", [])]
        substitusi = [i["name"] for i in recipe_classified.get("substitusi", [])]
        opsional = [i["name"] for i in recipe_classified.get("opsional", [])]

        utama_set = self._normalize_set(utama)
        substitusi_set = self._normalize_set(substitusi)
        opsional_set = self._normalize_set(opsional)

        matched_utama = list(utama_set & user_ingredients)
        missing_utama = list(utama_set - user_ingredients)
        matched_substitusi = list(substitusi_set & user_ingredients)
        matched_opsional = list(opsional_set & user_ingredients)

        # Hitung max score teoritis
        max_score = (
            len(utama_set) * WEIGHT_UTAMA
            + len(substitusi_set) * WEIGHT_SUBSTITUSI
            + len(opsional_set) * WEIGHT_OPSIONAL
        )

        if max_score == 0:
            return 0.0, {}

        # Actual score
        actual_score = (
            len(matched_utama) * WEIGHT_UTAMA
            + len(matched_substitusi) * WEIGHT_SUBSTITUSI
            + len(matched_opsional) * WEIGHT_OPSIONAL
        )

        score = round(actual_score / max_score, 4)
        can_cook = len(missing_utama) == 0  # bisa masak jika semua bahan utama ada

        return score, {
            "matched_utama": matched_utama,
            "missing_utama": missing_utama,
            "matched_substitusi": matched_substitusi,
            "matched_opsional": matched_opsional,
            "can_cook": can_cook,
        }

    def recommend(
        self,
        user_ingredients: list[str],
        top_n: int = 5,
        min_score: float = 0.3,
        prefer_can_cook: bool = True,
    ) -> list[RecipeScore]:
        """
        Rekomendasikan resep terbaik berdasarkan bahan user.

        Args:
            user_ingredients: List bahan yang dimiliki user (sudah dinormalisasi)
            top_n: Jumlah resep yang dikembalikan
            min_score: Skor minimum untuk dimasukkan hasil
            prefer_can_cook: Prioritaskan resep yang bisa langsung dimasak

        Returns:
            List RecipeScore, diurutkan dari skor tertinggi
        """
        user_set = self._normalize_set(user_ingredients)

        if not self.recipes_db:
            logger.warning("recipes_db kosong. Pastikan sudah di-load.")
            return []

        scored: list[RecipeScore] = []
        for recipe in self.recipes_db:
            classified = recipe.get("classified", {})
            score, detail = self.score_recipe(user_set, classified)

            if score < min_score:
                continue

            total = (
                len(classified.get("utama", []))
                + len(classified.get("substitusi", []))
                + len(classified.get("opsional", []))
            )

            scored.append(
                RecipeScore(
                    recipe_id=recipe["recipe_id"],
                    recipe_name=recipe.get("recipe_name", ""),
                    match_score=score,
                    matched_utama=detail.get("matched_utama", []),
                    missing_utama=detail.get("missing_utama", []),
                    matched_substitusi=detail.get("matched_substitusi", []),
                    matched_opsional=detail.get("matched_opsional", []),
                    total_ingredients=total,
                    can_cook=detail.get("can_cook", False),
                )
            )

        # Sort: can_cook dulu, lalu score tertinggi
        if prefer_can_cook:
            scored.sort(key=lambda r: (r.can_cook, r.match_score), reverse=True)
        else:
            scored.sort(key=lambda r: r.match_score, reverse=True)

        return scored[:top_n]

    def recommend_as_dict(
        self,
        user_ingredients: list[str],
        top_n: int = 5,
        min_score: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Wrapper yang mengembalikan dict (untuk JSON response API)."""
        results = self.recommend(user_ingredients, top_n, min_score)
        return [
            {
                "recipe_id": r.recipe_id,
                "recipe_name": r.recipe_name,
                "match_score": r.match_score,
                "can_cook": r.can_cook,
                "matched": {
                    "utama": r.matched_utama,
                    "substitusi": r.matched_substitusi,
                    "opsional": r.matched_opsional,
                },
                "missing_utama": r.missing_utama,
                "total_ingredients": r.total_ingredients,
            }
            for r in results
        ]
