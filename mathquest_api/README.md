# MathQuest — Placement Model REST API

API mandiri berbasis **FastAPI** untuk memprediksi topik lemah siswa
berdasarkan hasil pre-test.

---

## Struktur Proyek

```
mathquest_api/
├── app/
│   ├── __init__.py
│   ├── main.py        ← FastAPI app, endpoint, middleware
│   ├── model.py       ← load model, preprocessing, inference
│   ├── schemas.py     ← Pydantic request/response schemas
│   └── config.py      ← konfigurasi via env / .env
├── saved_model/       ← ⚠️ letakkan file model di sini
│   ├── placement_model.keras
│   ├── placement_scaler.pkl
│   ├── feature_names.txt
│   └── placement_model_metadata.json
├── .env.example
├── requirements.txt
└── run.py             ← entry point
```

---

## Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Letakkan file model
Salin semua file dari folder `saved_model/` hasil training ke
`mathquest_api/saved_model/`:
```
placement_model.keras
placement_scaler.pkl
feature_names.txt
placement_model_metadata.json
```

### 3. Konfigurasi (opsional)
```bash
cp .env.example .env
# Edit .env sesuai kebutuhan
```

### 4. Jalankan server
```bash
python run.py
# atau
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Buka Swagger UI
```
http://localhost:8000/docs
```

---

## Endpoint

| Method | Path             | Deskripsi                           |
|--------|------------------|-------------------------------------|
| GET    | `/health`        | Status server + model               |
| GET    | `/model/info`    | Metadata model (versi, fitur, dll)  |
| POST   | `/predict`       | Prediksi batch (banyak siswa)       |
| POST   | `/predict/single`| Prediksi satu siswa                 |
| GET    | `/docs`          | Swagger UI                          |

---

## Contoh Request

### POST /predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {"user_id":"stu_001","no_soal":1,"materi":"penjumlahan","benar_salah":0,"waktu_pengerjaan":55.0},
      {"user_id":"stu_001","no_soal":2,"materi":"pengurangan","benar_salah":0,"waktu_pengerjaan":62.0},
      {"user_id":"stu_001","no_soal":3,"materi":"perkalian",  "benar_salah":1,"waktu_pengerjaan":48.0}
    ]
  }'
```

**Response:**
```json
[
  {
    "user_id": "stu_001",
    "weak_topics": ["penjumlahan", "pengurangan"],
    "confidence": 0.8724
  }
]
```

### GET /health
```bash
curl http://localhost:8000/health
```
```json
{"status":"ok","model_loaded":true,"model_version":"1.0.0"}
```

---

## Panduan Integrasi Backend

### Tahap 1 — Deploy AI Service
Jalankan API ini sebagai service terpisah:
- **Development**: `localhost:8000`
- **Produksi**: container Docker / VM tersendiri, misalnya `http://ai-service:8000`

> ⚠️ **TensorFlow tidak aman di-fork.** Gunakan `workers=1` di Uvicorn.
> Untuk concurrency tinggi, gunakan async atau load model per-request dengan
> caching (tidak disarankan — gunakan queue seperti Celery).

### Tahap 2 — Backend Memanggil API

**Node.js / Express:**
```javascript
async function getPrediction(pretestRecords) {
  const response = await fetch('http://ai-service:8000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ records: pretestRecords }),
  });
  if (!response.ok) throw new Error(`AI service error: ${response.status}`);
  return response.json(); // [{user_id, weak_topics, confidence}]
}
```

**Python / Django / Flask:**
```python
import httpx

AI_SERVICE_URL = "http://ai-service:8000"

def get_prediction(records: list[dict]) -> list[dict]:
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{AI_SERVICE_URL}/predict",
            json={"records": records},
        )
        resp.raise_for_status()
        return resp.json()
```

### Tahap 3 — Format Data yang Dikirim Backend

Backend harus mengirim data dalam format ini:
```json
{
  "records": [
    {
      "user_id": "stu_001",
      "no_soal": 1,
      "materi": "penjumlahan",
      "benar_salah": 1,
      "waktu_pengerjaan": 12.5
    }
  ]
}
```

### Tahap 4 — Proses Hasil di Backend

```javascript
// Contoh: simpan hasil ke database dan kembalikan ke frontend
app.post('/api/pretest/submit', async (req, res) => {
  const { userId, answers } = req.body;

  // Format ulang ke struktur yang diharapkan AI
  const records = answers.map((a, idx) => ({
    user_id: userId,
    no_soal: idx + 1,
    materi: a.topic,
    benar_salah: a.isCorrect ? 1 : 0,
    waktu_pengerjaan: a.timeSpent,
  }));

  // Panggil AI service
  const predictions = await getPrediction(records);
  const result = predictions[0]; // satu siswa

  // Simpan ke DB
  await db.placement.upsert({
    where: { userId },
    update: { weakTopics: result.weak_topics, confidence: result.confidence },
    create: { userId, weakTopics: result.weak_topics, confidence: result.confidence },
  });

  res.json({
    weakTopics: result.weak_topics,
    confidence: result.confidence,
    message: result.weak_topics.length > 0
      ? `Topik yang perlu diperkuat: ${result.weak_topics.join(', ')}`
      : 'Semua topik sudah dikuasai dengan baik!',
  });
});
```

### Tahap 5 — Health Check & Error Handling

Selalu cek `/health` sebelum mengirim prediksi, atau tangani error 503:

```javascript
async function callAIWithFallback(records) {
  try {
    return await getPrediction(records);
  } catch (err) {
    if (err.status === 503) {
      // Model sedang tidak siap — log dan kembalikan fallback
      console.error('AI service unavailable');
      return records
        .map(r => r.user_id)
        .filter((v, i, a) => a.indexOf(v) === i)
        .map(uid => ({ user_id: uid, weak_topics: [], confidence: 0 }));
    }
    throw err;
  }
}
```

---

## Deployment dengan Docker (Produksi)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "run.py"]
```

```bash
docker build -t mathquest-api .
docker run -p 8000:8000 \
  -v $(pwd)/saved_model:/app/saved_model \
  mathquest-api
```

---

## Catatan Penting

- **workers=1** — TensorFlow tidak thread-safe untuk multi-process fork
- **saved_model/** — pastikan path ini benar sebelum menjalankan server
- **CORS** — ubah `ALLOWED_ORIGINS` di `.env` untuk membatasi akses di produksi
- **Timeout** — set timeout HTTP client backend minimal 30 detik (inference bisa lambat)
