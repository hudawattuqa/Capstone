"""
Skema Pydantic untuk validasi request dan response API.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ══════════════════════════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class PretestRecord(BaseModel):
    """Satu baris jawaban siswa pada soal pre-test."""

    user_id: str = Field(
        ...,
        description="ID unik siswa",
        examples=["stu_0001"],
    )
    no_soal: int = Field(
        ...,
        ge=1,
        description="Nomor urut soal (≥ 1)",
        examples=[1],
    )
    materi: str = Field(
        ...,
        description="Nama materi/topik soal",
        examples=["penjumlahan"],
    )
    benar_salah: int = Field(
        ...,
        description="1 = benar, 0 = salah",
        examples=[1],
    )
    waktu_pengerjaan: float = Field(
        ...,
        ge=1.0,
        le=300.0,
        description="Waktu pengerjaan dalam detik (1–300)",
        examples=[12.5],
    )

    @field_validator("benar_salah")
    @classmethod
    def validate_binary(cls, v):
        if v not in (0, 1):
            raise ValueError("benar_salah harus 0 atau 1")
        return v

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v):
        if not v.strip():
            raise ValueError("user_id tidak boleh kosong")
        return v.strip()

    @field_validator("materi")
    @classmethod
    def validate_materi(cls, v):
        if not v.strip():
            raise ValueError("materi tidak boleh kosong")
        return v.strip().lower()


class PredictRequest(BaseModel):
    """Request body untuk endpoint POST /predict (batch)."""

    records: list[PretestRecord] = Field(
        ...,
        min_length=1,
        description="Daftar jawaban pre-test (satu atau banyak siswa)",
        examples=[
            [
                {
                    "user_id": "stu_0001", "no_soal": 1,
                    "materi": "penjumlahan", "benar_salah": 0,
                    "waktu_pengerjaan": 55.0,
                },
                {
                    "user_id": "stu_0001", "no_soal": 2,
                    "materi": "pengurangan", "benar_salah": 0,
                    "waktu_pengerjaan": 62.0,
                },
            ]
        ],
    )


class PredictSingleRequest(BaseModel):
    """Request body untuk endpoint POST /predict/single (satu siswa)."""

    records: list[PretestRecord] = Field(
        ...,
        min_length=1,
        description="Daftar jawaban pre-test — semua harus user_id yang sama",
    )


# ══════════════════════════════════════════════════════════════════════════════
# RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class PredictResponse(BaseModel):
    """Hasil prediksi untuk satu siswa."""

    user_id: str = Field(..., description="ID siswa")
    weak_topics: list[str] = Field(
        ...,
        description="Daftar topik yang dianggap lemah (akurasi < threshold)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score prediksi level (0–1)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "stu_0001",
                "weak_topics": ["pengurangan", "perkalian"],
                "confidence": 0.87,
            }
        }
    }


class HealthResponse(BaseModel):
    """Response untuk endpoint /health."""

    status: str = Field(..., description="'ok' atau 'degraded'")
    model_loaded: bool = Field(..., description="Apakah model berhasil dimuat")
    model_version: Optional[str] = Field(
        None,
        description="Versi model (jika dimuat)",
    )


class ModelPerformance(BaseModel):
    test_accuracy: float
    test_mae: float
    target_accuracy_tercapai: bool
    target_mae_tercapai: bool


class ModelKonfigurasi(BaseModel):
    threshold_weak: float
    max_weak_topics: int
    batas_pemula: float
    batas_menengah: float


class ModelInfoResponse(BaseModel):
    """Response untuk endpoint /model/info."""

    nama_model: str
    versi: str
    deskripsi: str
    jumlah_fitur: int
    nama_fitur: list[str]
    konfigurasi: ModelKonfigurasi
    performa: ModelPerformance
