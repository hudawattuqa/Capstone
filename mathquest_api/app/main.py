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
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from groq import Groq, RateLimitError as GroqRateLimitError

from app.schemas import (
    PretestRecord,
    PredictRequest,
    PredictSingleRequest,
    PredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PembahasanRequest,
    PembahasanResponse,
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

# ─── Konfigurasi Generative AI (Pembahasan) ───────────────────────────────────
# Satu API key Groq, dua model — fallback otomatis kalau model utama rate limit
GROQ_API_KEY         = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL           = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL_FALLBACK  = os.getenv("GROQ_MODEL_FALLBACK", "llama3-8b-8192")

_groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


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


def _generate_with_fallback(system_prompt: str, user_prompt: str) -> tuple[str, str]:
    """
    Coba generate dengan model Groq utama dulu.
    Kalau kena rate limit, otomatis fallback ke model Groq kedua.

    Returns:
        tuple: (teks_pembahasan, nama_model_yang_dipakai)
    """
    if _groq_client is None:
        raise Exception("GROQ_API_KEY belum diset di .env")

    models = [GROQ_MODEL, GROQ_MODEL_FALLBACK]

    for model in models:
        try:
            logger.info("Mencoba generate dengan Groq (%s)...", model)
            response = _groq_client.chat.completions.create(
                model       = model,
                messages    = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature = 0.3,
                max_tokens  = 500,
            )
            teks = response.choices[0].message.content
            logger.info("Generate berhasil via Groq (%s).", model)
            return teks, f"groq/{model}"
        except GroqRateLimitError:
            logger.warning("Groq rate limit pada %s — mencoba model berikutnya.", model)
        except Exception as exc:
            logger.warning("Groq error pada %s: %s — mencoba model berikutnya.", model, exc)

    raise Exception("Semua model Groq gagal. Coba lagi nanti.")


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


@app.post(
    "/generate/pembahasan",
    response_model=PembahasanResponse,
    summary="Generate pembahasan soal (benar maupun salah)",
    tags=["Generative AI"],
)
async def generate_pembahasan(request: PembahasanRequest):
    """
    Generate pembahasan soal otomatis menggunakan AI.
    Dipanggil setiap kali siswa selesai menjawab satu soal,
    baik jawaban benar maupun salah.

    - `is_benar = true`  → pembahasan berisi apresiasi + penjelasan langkah
    - `is_benar = false` → pembahasan berisi koreksi + langkah yang benar + tips

    **Format input:**
    ```json
    {
      "user_id": "stu_001",
      "soal": "Jika 3x + 5 = 20, berapakah nilai x?",
      "pilihan": {"A": "3", "B": "4", "C": "5", "D": "6"},
      "jawaban_siswa": "C",
      "jawaban_benar": "C",
      "materi": "aljabar",
      "jenjang": "SMA"
    }
    ```

    **Format output:**
    ```json
    {
      "user_id": "stu_001",
      "pembahasan": "Hebat, jawabanmu benar! ...",
      "materi": "aljabar",
      "is_benar": true,
      "model_dipakai": "groq/llama-3.3-70b-versatile",
      "sukses": true
    }
    ```
    """
    if request.jawaban_benar not in request.pilihan:
        raise HTTPException(
            status_code=422,
            detail="jawaban_benar tidak ditemukan di dalam pilihan.",
        )

    is_benar = request.jawaban_siswa == request.jawaban_benar

    try:
        # Ambil teks jawaban siswa dan jawaban benar langsung dari dict pilihan
        # Pakai TEKS jawaban (bukan huruf A/B/C/D) agar AI tidak sebut huruf
        # yang urutannya bisa berbeda-beda tiap user
        teks_jawaban_siswa = request.pilihan.get(request.jawaban_siswa, request.jawaban_siswa)
        teks_jawaban_benar = request.pilihan.get(request.jawaban_benar, request.jawaban_benar)

        # Susun teks pilihan — tampilkan semua opsi tanpa label huruf
        # Tandai jawaban benar dengan "(JAWABAN BENAR)" agar AI tahu mana yang benar
        teks_pilihan = ""
        for kunci, nilai in request.pilihan.items():
            if kunci == request.jawaban_benar:
                teks_pilihan += f"- {nilai} (JAWABAN BENAR)\n"
            else:
                teks_pilihan += f"- {nilai}\n"

        system_prompt = (
            f"Kamu adalah tutor matematika yang sabar dan ramah untuk siswa "
            f"{request.jenjang} di Indonesia. Berikan pembahasan soal yang edukatif "
            f"menggunakan bahasa yang mudah dipahami. "
            f"Gunakan Bahasa Indonesia yang santai namun jelas."
        )

        if is_benar:
            user_prompt = (
                f'Siswa {request.jenjang} menjawab soal materi "{request.materi}" dengan BENAR.\n\n'
                f"SOAL:\n{request.soal}\n\n"
                f"PILIHAN JAWABAN:\n{teks_pilihan}\n"
                f'Siswa menjawab "{teks_jawaban_benar}" — BENAR!\n\n'
                f"Buat pembahasan yang:\n"
                f"1. Berikan APRESIASI singkat karena menjawab benar\n"
                f"2. Tunjukkan LANGKAH-LANGKAH penyelesaian yang benar\n"
                f"3. Berikan INSIGHT tambahan atau variasi soal serupa\n\n"
                f"PENTING: Jangan sebut huruf pilihan (A/B/C/D). "
                f"Gunakan teks jawaban secara langsung.\n"
                f"Maksimal 200 kata. Bahasa ramah untuk siswa {request.jenjang}."
            )
        else:
            user_prompt = (
                f'Siswa {request.jenjang} menjawab soal materi "{request.materi}" dengan SALAH.\n\n'
                f"SOAL:\n{request.soal}\n\n"
                f"PILIHAN JAWABAN:\n{teks_pilihan}\n"
                f'Siswa menjawab "{teks_jawaban_siswa}", '
                f'jawaban benar adalah "{teks_jawaban_benar}".\n\n'
                f"Buat pembahasan yang:\n"
                f'1. Jelaskan MENGAPA jawaban "{teks_jawaban_siswa}" kurang tepat (1-2 kalimat)\n'
                f"2. Tunjukkan LANGKAH-LANGKAH cara menjawab dengan benar\n"
                f"3. Berikan TIPS singkat agar tidak salah lagi\n\n"
                f"PENTING: Jangan sebut huruf pilihan (A/B/C/D). "
                f"Gunakan teks jawaban secara langsung.\n"
                f"Maksimal 200 kata. Bahasa ramah untuk siswa {request.jenjang}."
            )

        # ── Generate pembahasan (Groq utama, Gemini fallback) ──────────────────
        pembahasan, model_dipakai = _generate_with_fallback(system_prompt, user_prompt)

        return PembahasanResponse(
            user_id       = request.user_id,
            pembahasan    = pembahasan,
            materi        = request.materi,
            is_benar      = is_benar,
            model_dipakai = model_dipakai,
            sukses        = True,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Pembahasan error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generate pembahasan: {str(exc)}")


# ─── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "MathQuest Placement API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }