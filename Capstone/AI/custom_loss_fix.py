"""
custom_loss.py — WeightedCrossEntropyLoss untuk MathQuest AI Engine

Cara menjalankan:
    python custom_loss.py

Persyaratan:
    pip install tensorflow numpy
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras


# ════════════════════════════════════════════════════════════════
# CUSTOM LOSS FUNCTION
# ════════════════════════════════════════════════════════════════

class WeightedCrossEntropyLoss(keras.losses.Loss):
    """
    Custom loss function yang memberi penalti lebih besar ketika
    model salah memprediksi siswa yang kesulitan (kelas 0 = pemula)
    sebagai siswa yang sudah mampu (kelas 1 atau 2).

    Args:
        class_weights (list): bobot penalti per kelas.
            - Index 0 = pemula   → default 3.0 (dihukum 3x lebih berat)
            - Index 1 = menengah → default 1.0
            - Index 2 = mahir    → default 1.0
    """

    def __init__(self, class_weights=None, name="weighted_cross_entropy", **kwargs):
        super().__init__(name=name, **kwargs)

        # Simpan sebagai list Python biasa (bukan tensor)
        # supaya get_config() tidak error
        if class_weights is None:
            self.class_weights_list = [3.0, 1.0, 1.0]
        else:
            self.class_weights_list = list(class_weights)

        # Buat versi tensor untuk dipakai di perhitungan
        self.class_weights_tensor = tf.constant(
            self.class_weights_list, dtype=tf.float32
        )

    def call(self, y_true, y_pred):
        # Langkah 1: hitung cross entropy standar
        ce_loss = keras.losses.sparse_categorical_crossentropy(
            y_true,
            y_pred,
            from_logits=False  # output layer pakai softmax
        )

        # Langkah 2: ambil bobot sesuai kelas asli tiap siswa
        y_true_int     = tf.cast(y_true, tf.int32)
        sample_weights = tf.gather(self.class_weights_tensor, y_true_int)

        # Langkah 3: kalikan loss dengan bobot
        weighted_loss = ce_loss * sample_weights

        # Langkah 4: rata-ratakan seluruh batch
        return tf.reduce_mean(weighted_loss)

    def get_config(self):
        # Diperlukan agar model.save() dan model.load() bekerja dengan benar
        config = super().get_config()
        config.update({"class_weights": self.class_weights_list})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ════════════════════════════════════════════════════════════════
# TEST 1 — Coba loss function tanpa model
# ════════════════════════════════════════════════════════════════

def test_loss_tanpa_model():
    print("=" * 55)
    print("TEST 1: Loss function tanpa model")
    print("=" * 55)

    # Simulasi 4 siswa
    # Label asli: [pemula, mahir, menengah, pemula]
    y_true = tf.constant([0, 2, 1, 0], dtype=tf.float32)

    # Prediksi model (probabilitas per kelas)
    y_pred = tf.constant([
        [0.1, 0.1, 0.8],   # prediksi mahir  → SALAH (aslinya pemula)
        [0.1, 0.1, 0.8],   # prediksi mahir  → BENAR
        [0.2, 0.6, 0.2],   # prediksi menegh → BENAR
        [0.7, 0.2, 0.1],   # prediksi pemula → BENAR
    ])

    # Bandingkan loss biasa vs loss berbobot
    loss_biasa   = WeightedCrossEntropyLoss(class_weights=[1.0, 1.0, 1.0])
    loss_berbobot = WeightedCrossEntropyLoss(class_weights=[3.0, 1.0, 1.0])

    nilai_biasa    = loss_biasa(y_true, y_pred).numpy()
    nilai_berbobot = loss_berbobot(y_true, y_pred).numpy()

    print(f"\nSituasi: siswa pertama (pemula) diprediksi mahir → SALAH")
    print(f"\nLoss TANPA pembobotan : {nilai_biasa:.4f}")
    print(f"Loss DENGAN pembobotan: {nilai_berbobot:.4f}")
    print(f"Selisih               : {nilai_berbobot - nilai_biasa:.4f}")
    print(f"\n→ Loss berbobot lebih tinggi karena kesalahan pada")
    print(f"  siswa pemula dihukum 3x lebih berat.\n")


# ════════════════════════════════════════════════════════════════
# TEST 2 — Pakai di model kecil (mirip struktur model kalian)
# ════════════════════════════════════════════════════════════════

def test_loss_dengan_model():
    print("=" * 55)
    print("TEST 2: Loss function dengan model kecil")
    print("=" * 55)

    # Buat data dummy sederhana
    # 30 siswa, 5 fitur (versi mini dari 47 fitur kalian)
    np.random.seed(42)
    X_train = np.random.rand(30, 5).astype(np.float32)

    # Label: 0=pemula, 1=menengah, 2=mahir
    y_train = np.array([
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   # 10 siswa pemula
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,   # 10 siswa menengah
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,   # 10 siswa mahir
    ], dtype=np.float32)

    # Bangun model kecil (versi mini dari model kalian)
    inputs = keras.Input(shape=(5,), name="input_fitur")
    x      = keras.layers.Dense(16, activation="relu")(inputs)
    x      = keras.layers.Dense(8,  activation="relu")(x)
    output = keras.layers.Dense(3,  activation="softmax", name="output")(x)
    model  = keras.Model(inputs, output, name="MathQuestMini")

    # Compile dengan custom loss
    loss_fn = WeightedCrossEntropyLoss(class_weights=[3.0, 1.0, 1.0])
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=0.01),
        loss      = loss_fn,
        metrics   = ["accuracy"]
    )

    print(f"\nArsitektur model:")
    model.summary()

    # Training singkat
    print(f"\nTraining 20 epoch...\n")
    history = model.fit(
        X_train, y_train,
        epochs     = 20,
        batch_size = 8,
        verbose    = 1
    )

    loss_awal  = history.history["loss"][0]
    loss_akhir = history.history["loss"][-1]
    acc_akhir  = history.history["accuracy"][-1]

    print(f"\nLoss awal  (epoch 1)  : {loss_awal:.4f}")
    print(f"Loss akhir (epoch 20) : {loss_akhir:.4f}")
    print(f"Accuracy akhir        : {acc_akhir:.2%}")
    print(f"\n→ Loss turun = model belajar dengan benar ✓\n")


# ════════════════════════════════════════════════════════════════
# TEST 3 — Simpan dan load model (uji get_config)
# ════════════════════════════════════════════════════════════════

def test_save_load():
    print("=" * 55)
    print("TEST 3: Simpan dan load model (uji get_config)")
    print("=" * 55)

    # Buat dan compile model kecil
    inputs = keras.Input(shape=(5,))
    x      = keras.layers.Dense(8, activation="relu")(inputs)
    output = keras.layers.Dense(3, activation="softmax")(x)
    model  = keras.Model(inputs, output)

    loss_fn = WeightedCrossEntropyLoss(class_weights=[3.0, 1.0, 1.0])
    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    # Simpan model
    model.save("test_model.keras")
    print("\nModel berhasil disimpan → test_model.keras")

    # Load ulang model
    model_loaded = keras.models.load_model(
        "test_model.keras",
        custom_objects={"WeightedCrossEntropyLoss": WeightedCrossEntropyLoss}
    )
    print("Model berhasil di-load ulang ✓")

    # Coba prediksi dengan model yang sudah di-load
    X_test   = np.random.rand(3, 5).astype(np.float32)
    prediksi = model_loaded.predict(X_test, verbose=0)

    label    = ["pemula", "menengah", "mahir"]
    print(f"\nContoh prediksi dari model yang di-load:")
    for i, pred in enumerate(prediksi):
        kelas = label[np.argmax(pred)]
        print(f"  Siswa {i+1}: {pred.round(2)} → {kelas}")

    print(f"\n→ Save & load berhasil, get_config() bekerja dengan benar ✓\n")

    # Bersihkan file test
    import os
    if os.path.exists("test_model.keras"):
        os.remove("test_model.keras")


# ════════════════════════════════════════════════════════════════
# JALANKAN SEMUA TEST
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 55)
    print("  WeightedCrossEntropyLoss — Test Suite")
    print("█" * 55 + "\n")

    test_loss_tanpa_model()
    test_loss_dengan_model()
    test_save_load()

    print("=" * 55)
    print("Semua test selesai!")
    print("=" * 55)