"""
Kelas PlacementModel — wrapper untuk load dan inference model TF/Keras.
Memisahkan logika ML dari layer HTTP (FastAPI).
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("mathquest.model")


# ══════════════════════════════════════════════════════════════════════════════
# Custom Keras Objects
# (HARUS sama persis dengan definisi saat training)
# ══════════════════════════════════════════════════════════════════════════════

def _build_custom_objects():
    """
    Lazy-import TensorFlow dan definisikan custom layers/losses.
    Dijalankan sekali saat PlacementModel.load() dipanggil.
    Kompatibel dengan TF 2.x (lama) maupun TF 2.16+ (keras standalone).
    """
    import tensorflow as tf
    try:
        import keras  # TF 2.16+ — keras sebagai package terpisah
    except ImportError:
        from tensorflow import keras  # TF < 2.16

    class TopicAttentionLayer(keras.layers.Layer):
        def __init__(self, units, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.attention_dense = keras.layers.Dense(units, activation="sigmoid")

        def call(self, inputs):
            attention_weights = self.attention_dense(inputs)
            return inputs * attention_weights

        def get_config(self):
            config = super().get_config()
            config.update({"units": self.units})
            return config

    class WeightedCrossEntropyLoss(keras.losses.Loss):
        def __init__(self, class_weights=None, name="weighted_cross_entropy", **kwargs):
            super().__init__(name=name, **kwargs)
            self.class_weights_list = class_weights if class_weights else [3.0, 1.0, 1.0]
            self.class_weights_tensor = tf.constant(
                self.class_weights_list, dtype=tf.float32
            )

        def call(self, y_true, y_pred):
            ce_loss = keras.losses.sparse_categorical_crossentropy(
                y_true, y_pred, from_logits=False
            )
            y_true_int = tf.cast(y_true, tf.int32)
            sample_weights = tf.gather(self.class_weights_tensor, y_true_int)
            return tf.reduce_mean(ce_loss * sample_weights)

        def get_config(self):
            config = super().get_config()
            config.update({"class_weights": self.class_weights_list})
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    return {
        "TopicAttentionLayer": TopicAttentionLayer,
        "WeightedCrossEntropyLoss": WeightedCrossEntropyLoss,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PlacementModel
# ══════════════════════════════════════════════════════════════════════════════

class PlacementModel:
    """
    Wrapper untuk model placement MathQuest.

    Responsibilities:
    - Load model .keras, scaler .pkl, feature names .txt, metadata .json
    - Preprocessing: raw records → feature matrix
    - Inference: feature matrix → weak_topics + confidence
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        features_path: str,
        metadata_path: str,
    ):
        self.model_path    = Path(model_path)
        self.scaler_path   = Path(scaler_path)
        self.features_path = Path(features_path)
        self.metadata_path = Path(metadata_path)

        self._model     = None
        self._scaler    = None
        self._feat_names: list[str] = []
        self._metadata: dict = {}
        self.is_loaded  = False
        self.version: Optional[str] = None

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self):
        """Load semua artefak model dari disk."""
        self._validate_paths()

        logger.info("Loading TensorFlow model dari %s ...", self.model_path)
        try:
            import keras  # TF 2.16+
        except ImportError:
            from tensorflow import keras  # TF < 2.16
        custom_objects = _build_custom_objects()
        self._model = keras.models.load_model(
            str(self.model_path),
            custom_objects=custom_objects,
        )

        logger.info("Loading scaler dari %s ...", self.scaler_path)
        self._scaler = joblib.load(str(self.scaler_path))

        logger.info("Loading feature names dari %s ...", self.features_path)
        self._feat_names = (
            self.features_path.read_text(encoding="utf-8").splitlines()
        )
        self._feat_names = [f for f in self._feat_names if f.strip()]

        logger.info("Loading metadata dari %s ...", self.metadata_path)
        with open(self.metadata_path, encoding="utf-8") as f:
            self._metadata = json.load(f)

        self.version   = self._metadata.get("versi", "unknown")
        self.is_loaded = True

        logger.info(
            "Model '%s' v%s dimuat — %d fitur",
            self._metadata.get("nama_model"),
            self.version,
            len(self._feat_names),
        )

    def _validate_paths(self):
        missing = []
        for p in [self.model_path, self.scaler_path,
                  self.features_path, self.metadata_path]:
            if not p.exists():
                missing.append(str(p))
        if missing:
            raise FileNotFoundError(
                f"File model tidak ditemukan:\n" + "\n".join(f"  - {m}" for m in missing)
            )

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def _preprocess(self, records: list[dict]) -> tuple[np.ndarray, list[str], pd.DataFrame]:
        """
        Ubah raw records menjadi feature matrix yang siap di-predict.

        Returns:
            X_scaled   : ndarray shape (n_siswa, n_fitur)
            user_ids   : list user_id sesuai urutan baris X_scaled
            akurasi_raw: DataFrame akurasi per materi per siswa (untuk deteksi topik lemah)
        """
        df = pd.DataFrame(records)

        # Validasi kolom
        required = {"user_id", "materi", "benar_salah", "waktu_pengerjaan"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Kolom wajib tidak ada: {missing}")

        # Cleaning
        df = df.dropna(subset=["benar_salah", "waktu_pengerjaan", "materi"])
        df = df[df["waktu_pengerjaan"].between(1, 300)]
        df = df[df["benar_salah"].isin([0, 1])]
        df["materi"] = df["materi"].str.strip().str.lower()

        if df.empty:
            raise ValueError("Semua record tidak valid setelah cleaning.")

        # Akurasi per materi per siswa
        akurasi_per_materi = (
            df.groupby(["user_id", "materi"])["benar_salah"]
            .mean()
            .unstack(fill_value=0)
        )
        akurasi_raw = akurasi_per_materi.copy()
        akurasi_per_materi.columns = [
            f"akurasi_{m}" for m in akurasi_per_materi.columns
        ]

        # Rata-rata waktu per materi per siswa
        avg_waktu = (
            df.groupby(["user_id", "materi"])["waktu_pengerjaan"]
            .mean()
            .unstack(fill_value=0)
        )
        avg_waktu.columns = [f"avg_waktu_{m}" for m in avg_waktu.columns]

        # Gabung
        df_siswa = akurasi_per_materi.merge(avg_waktu, on="user_id", how="inner")

        # Reindex agar urutan kolom sama dengan saat training
        df_siswa = df_siswa.reindex(columns=self._feat_names, fill_value=0)

        user_ids = df_siswa.index.tolist()
        X_scaled = self._scaler.transform(df_siswa.values)

        return X_scaled, user_ids, akurasi_raw

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, records: list[dict]) -> list[dict]:
        """
        Jalankan inference dari raw records.

        Args:
            records: list of dict dengan key
                     user_id, no_soal, materi, benar_salah, waktu_pengerjaan

        Returns:
            list of dict: [{"user_id": ..., "weak_topics": [...], "confidence": ...}]
        """
        if not self.is_loaded:
            raise RuntimeError("Model belum dimuat. Panggil load() terlebih dahulu.")

        X_scaled, user_ids, akurasi_raw = self._preprocess(records)

        cfg             = self._metadata.get("konfigurasi", {})
        threshold_weak  = cfg.get("threshold_weak", 0.6)
        max_weak_topics = cfg.get("max_weak_topics", 3)

        # Prediksi probabilitas untuk semua siswa sekaligus (lebih efisien)
        proba_all = self._model.predict(X_scaled, verbose=0)

        results = []
        for i, user_id in enumerate(user_ids):
            proba       = proba_all[i]
            level_idx   = int(np.argmax(proba))
            confidence  = float(proba[level_idx])

            # Topik lemah: materi dengan akurasi < threshold
            if user_id in akurasi_raw.index:
                akurasi_siswa = akurasi_raw.loc[user_id]
                topik_lemah   = akurasi_siswa[akurasi_siswa < threshold_weak]
                topik_lemah   = topik_lemah.sort_values(ascending=True)
                if max_weak_topics:
                    topik_lemah = topik_lemah.head(max_weak_topics)
                weak_topics = topik_lemah.index.tolist()
            else:
                weak_topics = []

            results.append({
                "user_id"    : user_id,
                "weak_topics": weak_topics,
                "confidence" : round(confidence, 4),
            })

        return results

    # ── Info ──────────────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        """Kembalikan metadata model untuk endpoint /model/info."""
        return self._metadata
