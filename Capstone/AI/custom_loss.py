import tensorflow as tf
from tensorflow import keras


class WeightedCrossEntropyLoss(keras.losses.Loss):
    """
    Custom loss function untuk Placement Model & Adaptive Difficulty Model.

    Masalah yang diselesaikan:
        Loss standar memperlakukan semua kesalahan sama berat.
        Tapi di konteks pendidikan, ada satu jenis kesalahan yang
        jauh lebih berbahaya: model memprediksi siswa "sudah mahir"
        padahal siswa tersebut sebenarnya masih kesulitan.

        Akibatnya: siswa diberi soal terlalu susah → frustrasi → berhenti belajar.

        Loss ini memberi penalti LEBIH BESAR untuk kesalahan jenis itu.

    Args:
        class_weights (list): bobot penalti untuk tiap kelas.
                              Indeks = nomor kelas, nilai = besarnya penalti.
                              Contoh: [3.0, 1.0, 1.0] artinya kesalahan
                              pada kelas 0 (pemula) dihukum 3x lebih berat.
        name (str): nama loss function (opsional, untuk logging).

    Contoh pemakaian:
        loss_fn = WeightedCrossEntropyLoss(class_weights=[3.0, 1.0, 1.0])
        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    """

    def __init__(self, class_weights=None, name="weighted_cross_entropy", **kwargs):
        super().__init__(name=name, **kwargs)

        # Default: kelas 0 (pemula/kesulitan) diberi bobot 3x lebih berat
        # Kelas 1 (menengah) dan kelas 2 (mahir) bobot normal
        if class_weights is None:
            class_weights = [3.0, 1.0, 1.0]

        # Simpan sebagai tensor supaya bisa dipakai dalam operasi TensorFlow
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        """
        Hitung nilai loss dengan pembobotan per kelas.

        Args:
            y_true: label asli, bentuknya (batch_size,)
                    contoh: [0, 2, 1, 0] → 4 siswa dengan level berbeda
            y_pred: hasil prediksi model, bentuknya (batch_size, num_classes)
                    contoh: [[0.7, 0.2, 0.1],   ← siswa 1: 70% pemula
                             [0.1, 0.1, 0.8],   ← siswa 2: 80% mahir
                             [0.2, 0.6, 0.2],   ← siswa 3: 60% menengah
                             [0.8, 0.1, 0.1]]   ← siswa 4: 80% pemula

        Returns:
            loss: satu angka float — rata-rata weighted loss seluruh batch
        """

        # ── LANGKAH 1: Hitung cross entropy standar ──────────────────────────
        # sparse = y_true berupa integer (0,1,2), bukan one-hot ([1,0,0])
        # from_logits=False karena output layer pakai softmax (sudah jadi probabilitas)
        ce_loss = keras.losses.sparse_categorical_crossentropy(
            y_true,
            y_pred,
            from_logits=False
        )
        # Hasil: tensor 1D, satu nilai loss per siswa
        # contoh: [1.2, 0.3, 0.8, 1.5]

        # ── LANGKAH 2: Ambil bobot sesuai kelas asli tiap siswa ──────────────
        # y_true = [0, 2, 1, 0]
        # class_weights = [3.0, 1.0, 1.0]
        # → bobot tiap siswa = [3.0, 1.0, 1.0, 3.0]
        y_true_int = tf.cast(y_true, tf.int32)
        sample_weights = tf.gather(self.class_weights, y_true_int)
        # tf.gather = "ambil elemen ke-i dari class_weights sesuai indeks y_true"

        # ── LANGKAH 3: Kalikan loss dengan bobot ─────────────────────────────
        # ce_loss     = [1.2, 0.3, 0.8, 1.5]
        # bobot       = [3.0, 1.0, 1.0, 3.0]
        # hasil       = [3.6, 0.3, 0.8, 4.5]  ← siswa pemula dihukum lebih berat
        weighted_loss = ce_loss * sample_weights

        # ── LANGKAH 4: Rata-ratakan seluruh batch ────────────────────────────
        return tf.reduce_mean(weighted_loss)

    def get_config(self):
        """
        Diperlukan supaya model bisa disimpan dan di-load ulang dengan benar.
        Tanpa ini, model.save() akan error karena tidak tahu parameter loss-nya.
        """
        config = super().get_config()
        config.update({
            "class_weights": self.class_weights.numpy().tolist()
        })
        return config


# ════════════════════════════════════════════════════════════════
# CARA PAKAI
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Opsi 1: Pakai bobot default (pemula 3x lebih berat) ──────────────────
    loss_fn = WeightedCrossEntropyLoss()

    # ── Opsi 2: Sesuaikan bobot sendiri ──────────────────────────────────────
    # Misalnya kalian ingin pemula 5x lebih berat setelah lihat hasil evaluasi
    loss_fn_custom = WeightedCrossEntropyLoss(class_weights=[5.0, 1.0, 1.0])

    # ── Pakai di model.compile ────────────────────────────────────────────────
    # model.compile(
    #     optimizer = keras.optimizers.Adam(learning_rate=0.001),
    #     loss      = loss_fn,
    #     metrics   = ['accuracy']
    # )

    # ── Test manual tanpa model ───────────────────────────────────────────────
    # Simulasi: 4 siswa, 3 kelas (pemula=0, menengah=1, mahir=2)
    y_true_test = tf.constant([0, 2, 1, 0])          # label asli
    y_pred_test = tf.constant([
        [0.1, 0.1, 0.8],   # prediksi: mahir  → SALAH (aslinya pemula) ← dihukum berat
        [0.1, 0.1, 0.8],   # prediksi: mahir  → BENAR
        [0.2, 0.6, 0.2],   # prediksi: menengah → BENAR
        [0.7, 0.2, 0.1],   # prediksi: pemula → BENAR
    ])

    loss_default = WeightedCrossEntropyLoss(class_weights=[1.0, 1.0, 1.0])
    loss_weighted = WeightedCrossEntropyLoss(class_weights=[3.0, 1.0, 1.0])

    print("Loss TANPA pembobotan :", loss_default(y_true_test, y_pred_test).numpy())
    print("Loss DENGAN pembobotan:", loss_weighted(y_true_test, y_pred_test).numpy())
    # Loss dengan pembobotan akan lebih tinggi karena siswa pertama
    # (pemula yang diprediksi mahir) dihukum 3x lebih berat