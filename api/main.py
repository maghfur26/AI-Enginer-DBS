"""

Endpoint REST API untuk inference model BERT klasifikasi bahan.
Bridge antara model Python (AE1) dan backend Node.js Express (Full-Stack).
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.inference.predictor import IngredientPredictor
from src.preprocessing.normalizer import normalize_ingredients_list
from api.schemas.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    NormalizeRequest,
    NormalizeResponse,
    RecipeClassifyRequest,
    RecipeClassifyResponse,
)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global predictor (loaded once at startup) ──────────────────────────────
predictor = IngredientPredictor(artifacts_dir="artifacts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model saat startup, cleanup saat shutdown."""
    logger.info("Loading model artifacts...")
    predictor.load()
    if predictor.is_ready():
        logger.info("✅ Model loaded. Running in PRODUCTION mode.")
    else:
        logger.warning("⚠️  Model not found. Running in MOCK mode (development).")
    yield
    logger.info("Shutting down OLAH Inference API.")


# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="OLAH Inference API",
    description=(
        "AE2 — REST API untuk klasifikasi bahan makanan menggunakan model BERT. "
        "Bagian dari Proyek OLAH · CC26-PSU127 · Coding Camp 2026."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sesuaikan di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: Request timing ─────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(duration)
    return response


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "project": "OLAH",
        "team": "CC26-PSU127",
        "role": "AE2 — Inference & Integrasi",
        "status": "running",
        "mode": "production" if predictor.is_ready() else "mock",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint untuk monitoring dan integrasi frontend.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_ready(),
        mode="production" if predictor.is_ready() else "mock",
    )


@app.post("/api/v1/classify", response_model=ClassifyResponse, tags=["Inference"])
async def classify_ingredient(body: ClassifyRequest):
    """
    Klasifikasi satu bahan makanan.
    
    - **ingredient**: Nama bahan (raw, belum dinormalisasi)
    - Returns label: `utama`, `substitusi`, atau `opsional`
    """
    if not body.ingredient or not body.ingredient.strip():
        raise HTTPException(status_code=422, detail="Ingredient tidak boleh kosong.")

    # Normalisasi dulu
    from src.preprocessing.normalizer import normalize_ingredient
    normalized = normalize_ingredient(body.ingredient)

    result = predictor.predict_single(normalized)

    return ClassifyResponse(
        ingredient_raw=body.ingredient,
        ingredient_normalized=normalized,
        label=result["label"],
        confidence=result["confidence"],
        all_scores=result["all_scores"],
        mock=result.get("_mock", False),
    )


@app.post(
    "/api/v1/classify/batch",
    response_model=list[ClassifyResponse],
    tags=["Inference"],
)
async def classify_batch(body: ClassifyRequest):
    """
    Klasifikasi batch bahan (list).
    Gunakan endpoint ini untuk efisiensi saat mengirim banyak bahan sekaligus.
    """
    if not body.ingredients:
        raise HTTPException(status_code=422, detail="List bahan tidak boleh kosong.")
    if len(body.ingredients) > 50:
        raise HTTPException(
            status_code=422, detail="Maksimal 50 bahan per request."
        )

    normalized_list = normalize_ingredients_list(body.ingredients)
    results = []
    for item in normalized_list:
        pred = predictor.predict_single(item["normalized"])
        results.append(
            ClassifyResponse(
                ingredient_raw=item["raw"],
                ingredient_normalized=item["normalized"],
                label=pred["label"],
                confidence=pred["confidence"],
                all_scores=pred["all_scores"],
                mock=pred.get("_mock", False),
            )
        )
    return results


@app.post(
    "/api/v1/recipe/classify",
    response_model=RecipeClassifyResponse,
    tags=["Inference"],
)
async def classify_recipe(body: RecipeClassifyRequest):
    """
    Endpoint utama: klasifikasi seluruh bahan dalam satu resep.
    
    Menerima `recipe_id` dan list `ingredients`, mengembalikan bahan
    yang sudah dikelompokkan: utama, substitusi, opsional.
    
    Digunakan oleh backend Node.js untuk sistem rekomendasi OLAH.
    """
    if not body.ingredients:
        raise HTTPException(status_code=422, detail="Bahan resep tidak boleh kosong.")

    # Normalisasi semua bahan
    normalized_list = normalize_ingredients_list(body.ingredients)
    normalized_names = [item["normalized"] for item in normalized_list]

    # Klasifikasi
    classification = predictor.classify_recipe_ingredients(normalized_names)

    return RecipeClassifyResponse(
        recipe_id=body.recipe_id,
        recipe_name=body.recipe_name,
        classified=classification["classified"],
        raw_predictions=classification["raw_predictions"],
        total_ingredients=classification["total_ingredients"],
        normalization_log=normalized_list,
        mock=any(p.get("_mock") for p in classification["raw_predictions"]),
    )


@app.post(
    "/api/v1/normalize",
    response_model=NormalizeResponse,
    tags=["Preprocessing"],
)
async def normalize_ingredients(body: NormalizeRequest):
    """
    Normalisasi nama bahan (sinonim, typo, satuan).
    Utility endpoint — berguna untuk frontend auto-suggestion.
    """
    results = normalize_ingredients_list(body.ingredients)
    return NormalizeResponse(results=results, count=len(results))


# ── Global exception handler ───────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Cek log untuk detail."},
    )
