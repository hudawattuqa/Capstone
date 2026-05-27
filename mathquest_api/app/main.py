"""
MathQuest — Placement Model REST API
=====================================
FastAPI server untuk model prediksi topik lemah siswa
berdasarkan hasil pre-test.

Endpoint:
  POST /predict        → prediksi untuk satu / banyak siswa
  POST /predict/single → prediksi untuk satu siswa saja
  GET  /health         → cek status server + model
  GET  /model/info     → metadata model (versi, fitur, performa)
  GET  /docs           → Swagger UI (otomatis dari FastAPI)
"""

from contextlib import asynccontextmanager
from typing import Optional
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import (
    PretestRecord,
    PredictRequest,
    PredictSingleRequest,
    PredictResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.model import PlacementModel
from app.config import settings

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("mathquest.api")

# ─── Global model instance ────────────────────────────────────────────────────
placement_model: Optional[PlacementModel] = None


# ─── Lifespan (load model saat startup) ──────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model saat server start, bersihkan saat shutdown."""
    global placement_model
    logger.info("Memuat model dari: %s", settings.MODEL_PATH)
    try:
        placement_model = PlacementModel(
            model_path=settings.MODEL_PATH,
            scaler_path=settings.SCALER_PATH,
            features_path=settings.FEATURES_PATH,
            metadata_path=settings.MODEL_META_PATH,
        )
        placement_model.load()
        logger.info("Model berhasil dimuat ✓")
    except Exception as exc:
        logger.error("Gagal memuat model: %s", exc)
        # Server tetap jalan, tapi /predict akan return 503
        placement_model = None
    yield
    logger.info("Server shutdown — membersihkan resource...")


# ─── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="MathQuest Placement API",
    description=(
        "REST API untuk memprediksi topik lemah siswa "
        "berdasarkan hasil pre-test menggunakan model TensorFlow/Keras."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS (sesuaikan origin backend kamu) ─────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Middleware: request logging + timing ─────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 2)
    logger.info(
        "%s %s → %d  (%.1f ms)",
        request.method, request.url.path, response.status_code, elapsed,
    )
    return response


# ─── Helper ───────────────────────────────────────────────────────────────────
def _require_model():
    if placement_model is None or not placement_model.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model belum dimuat. Periksa path file di konfigurasi.",
        )


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Cek status server dan model",
    tags=["Monitoring"],
)
async def health_check():
    """
    Kembalikan status server.

    - `status: ok`   → server + model siap
    - `status: degraded` → server jalan, model gagal dimuat
    """
    if placement_model and placement_model.is_loaded:
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=placement_model.version,
        )
    return HealthResponse(status="degraded", model_loaded=False)


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Metadata model (versi, fitur, performa)",
    tags=["Monitoring"],
)
async def model_info():
    """Kembalikan metadata lengkap model yang sedang berjalan."""
    _require_model()
    return placement_model.get_info()


@app.post(
    "/predict",
    response_model=list[PredictResponse],
    summary="Prediksi topik lemah (batch — banyak siswa)",
    tags=["Prediksi"],
)
async def predict_batch(body: PredictRequest):
    """
    Terima daftar jawaban pre-test dari satu atau banyak siswa,
    kembalikan daftar topik lemah beserta confidence untuk tiap siswa.

    **Format input:**
    ```json
    {
      "records": [
        {"user_id": "stu_001", "no_soal": 1, "materi": "penjumlahan",
         "benar_salah": 1, "waktu_pengerjaan": 12.5},
        ...
      ]
    }
    ```

    **Format output:**
    ```json
    [
      {"user_id": "stu_001", "weak_topics": ["pengurangan"], "confidence": 0.87}
    ]
    ```
    """
    _require_model()
    if not body.records:
        raise HTTPException(status_code=422, detail="Field 'records' tidak boleh kosong.")

    try:
        records_dict = [r.model_dump() for r in body.records]
        results = placement_model.predict(records_dict)
        return results
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat prediksi.")


@app.post(
    "/predict/single",
    response_model=PredictResponse,
    summary="Prediksi topik lemah (single — satu siswa)",
    tags=["Prediksi"],
)
async def predict_single(body: PredictSingleRequest):
    """
    Shortcut untuk prediksi satu siswa saja.
    Semua record HARUS memiliki `user_id` yang sama.

    Kembalikan satu objek hasil (bukan list).
    """
    _require_model()
    if not body.records:
        raise HTTPException(status_code=422, detail="Field 'records' tidak boleh kosong.")

    user_ids = {r.user_id for r in body.records}
    if len(user_ids) > 1:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Endpoint ini hanya untuk satu siswa. "
                f"Ditemukan {len(user_ids)} user_id berbeda: {user_ids}. "
                f"Gunakan POST /predict untuk batch."
            ),
        )

    try:
        records_dict = [r.model_dump() for r in body.records]
        results = placement_model.predict(records_dict)
        return results[0]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Terjadi kesalahan saat prediksi.")


# ─── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "MathQuest Placement API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
