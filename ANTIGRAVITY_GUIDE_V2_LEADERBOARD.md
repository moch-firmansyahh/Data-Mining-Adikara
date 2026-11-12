# Panduan V2: Naik ke Puncak Leaderboard (Data Mining ADIKARA 2026)

Dokumen ini adalah lanjutan dari panduan sebelumnya. Pipeline sudah ada dan berjalan; tugas sekarang adalah **menurunkan RMSLE**. Kerjakan per langkah, ukur dampaknya, dan simpan hanya yang terbukti membaik.

## 0. Konteks Saat Ini

- Metrik resmi: **RMSLE** (RMSE pada `log1p(likes)`). Makin kecil makin baik.
- Posisi sekarang: peringkat 26, skor publik **1.46153**.
- Target: peringkat 1 berada di sekitar **1.33768** (ada di leaderboard saat ini). Peringkat 2: 1.34197, peringkat 3: 1.34650. Selisih ke puncak sekitar 0,12, jadi diperlukan peningkatan nyata, bukan sekadar tuning kecil.
- Deadline: **16 Nov 2026, 23.59 WIB**.
- Aset yang sudah ada: `notebooks/final_submission_pipeline.ipynb`, `outputs/oof_baseline.csv`, `oof_ensemble.csv`, dan submission LGB, CatBoost, XGB, blend.
- Temuan EDA sebelumnya: skew `likes` sangat tinggi (maks 47.952), **25,66% bernilai 0**, rentang waktu train dan test sama, **68,81% tweet test berasal dari user yang ada di train** (sisanya 31,19% user baru).

**Catatan jujur:** tidak ada jaminan mencapai peringkat 1. Peringkat akhir ditentukan test set tersembunyi, jadi jangan menyesuaikan model terlalu dekat dengan skor publik. Patokan utama tetap CV yang benar. Setiap langkah di bawah adalah hipotesis yang harus diukur.

## 1. Aturan Wajib (jangan dilanggar)

1. Notebook final `.ipynb` maksimal **25 MB** (bersihkan output besar; jangan simpan embedding di notebook).
2. Notebook memuat preprocessing, konfigurasi model, training, dan inference.
3. Cantumkan **sumber pretrained model, link repo, library dan versinya**. Jika tidak ada pretrained model, tulis eksplisit.
4. Model weights yang dipakai evaluasi harus di repo HF yang **public**, tanpa token.
5. Notebook harus bisa dijalankan ulang dari atas ke bawah tanpa error.
6. Tidak boleh ada fitur yang dihitung dari `likes` di data validasi atau test (tanpa skema out-of-fold).

## 2. Langkah 1 (prioritas tertinggi): Selaraskan Semuanya dengan RMSLE

Ringkasan metodologi sebelumnya menyebut blending meminimalkan **Raw RMSE** dan memakai **Duan smearing / kalibrasi skala mentah**. Itu tidak selaras dengan RMSLE.

Lakukan:
- Semua evaluasi OOF dihitung di ruang log: `rmse(z_true, z_pred)` dengan `z = log1p(likes)`.
- **Hapus** smearing dan kalibrasi skala mentah dari pipeline prediksi akhir. Prediksi akhir = `expm1(z_pred)`, lalu `clip(min=0)`.
- Optimasi bobot blending (SLSQP atau NNLS) pada **RMSE di ruang log**, bukan skala mentah.
- Bandingkan di OOF: blend lama vs blend baru. Submit versi baru dan catat selisih skor publik.

```python
import numpy as np
from scipy.optimize import minimize

def rmse_log(z_true, z_pred):
    return float(np.sqrt(np.mean((z_true - z_pred) ** 2)))

def optimize_blend(oof_list, z_true):
    # oof_list: list prediksi OOF di ruang log, masing-masing shape (n,)
    X = np.column_stack(oof_list)
    n = X.shape[1]
    f = lambda w: rmse_log(z_true, X @ w)
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1}
    res = minimize(f, np.ones(n) / n, bounds=[(0, 1)] * n,
                   constraints=cons, method="SLSQP")
    return res.x, res.fun
```

## 3. Langkah 2: Fitur Kedekatan Waktu per User (kandidat lompatan terbesar)

Popularitas akun berubah seiring waktu. Rata-rata likes seluruh periode terlalu kasar. Tambahkan fitur dari **tweet train user yang sama yang paling dekat waktunya**.

Aturan anti-bocor:
- Untuk baris validasi: referensi hanya baris fold latih.
- Untuk baris test: referensi seluruh train.
- Gunakan jumlah fold yang lebih besar (misalnya 10 fold) saat membuat fitur agar kerapatan referensi OOF mirip dengan kondisi test.
- Skema terbaik: fitur untuk baris latih dibuat dengan inner-OOF di dalam setiap fold luar. Jika waktu terbatas, pakai fold yang sama untuk fitur dan model, dan sadari bahwa ada kebocoran ringan yang membuat CV sedikit optimistis. Selalu cek dengan leaderboard.

```python
import numpy as np
import pandas as pd

def time_neighbor_feats(query, ref, ks=(3, 10), windows_days=(7, 30)):
    """
    query, ref: DataFrame dengan kolom username, ts (detik/hari numerik), z (log1p likes, hanya di ref).
    Return DataFrame fitur berindeks sama dengan query.
    """
    out = []
    ref_g = {u: g.sort_values("ts") for u, g in ref.groupby("username")}
    prior = ref["z"].mean()
    for idx, row in query.iterrows():
        g = ref_g.get(row["username"])
        feats = {"idx": idx}
        if g is None or len(g) == 0:
            for k in ks:
                feats[f"nn_mean_{k}"] = prior
                feats[f"nn_wmean_{k}"] = prior
            feats["nn_gap"] = np.nan
            feats["nn_z1"] = prior
            for w in windows_days:
                feats[f"cnt_{w}d"] = 0
                feats[f"mean_{w}d"] = prior
            out.append(feats)
            continue
        ts = g["ts"].values
        z = g["z"].values
        dt = np.abs(ts - row["ts"])
        order = np.argsort(dt)
        feats["nn_gap"] = dt[order[0]]
        feats["nn_z1"] = z[order[0]]
        for k in ks:
            sel = order[:k]
            feats[f"nn_mean_{k}"] = z[sel].mean()
            wts = 1.0 / (1.0 + dt[sel])
            feats[f"nn_wmean_{k}"] = float((z[sel] * wts).sum() / wts.sum())
        for w in windows_days:
            mask = dt <= w * 86400  # sesuaikan satuan ts
            feats[f"cnt_{w}d"] = int(mask.sum())
            feats[f"mean_{w}d"] = z[mask].mean() if mask.any() else prior
        out.append(feats)
    return pd.DataFrame(out).set_index("idx")
```

Tambahan fitur tanpa target (aman dihitung dari train+test gabungan):
- Selisih waktu ke tweet sebelumnya dan berikutnya dari user yang sama.
- Jumlah tweet user dalam 1, 7, 30 hari sekitar tweet itu.
- Urutan tweet dalam riwayat user, dan umur akun relatif.

Ukur: latih ulang LightGBM dengan fitur baru, bandingkan CV RMSLE dengan baseline. Optimasi `ts` agar satuan jendela waktu benar (detik atau hari).

## 4. Langkah 3: Embedding Teks Transformer

TF-IDF + Ridge saja kurang menangkap makna. Tambahkan embedding.

- Kandidat (**verifikasi dulu bahwa repo ada dan public di HF Hub**): `indobenchmark/indobert-base-p1`, `indolem/indobertweet-base-uncased`, `intfloat/multilingual-e5-base`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Ekstrak embedding (mean pooling) dengan batch, simpan ke `outputs/emb_<nama>.npy` (bukan di notebook).
- Dua cara pakai:
  1. Reduksi ke 32-128 dimensi dengan PCA/SVD, lalu masukkan ke GBDT.
  2. Latih Ridge atau MLP kecil pada embedding, ambil prediksi OOF sebagai satu fitur stacking.
- Cantumkan nama model dan link repo di bagian sumber notebook.
- Jika tidak ada GPU, pilih model kecil (MiniLM) dan batch kecil.

## 5. Langkah 4: Analisis User Dikenal vs User Baru

- Hitung RMSLE OOF terpisah untuk baris yang user-nya muncul di fold latih dan yang tidak.
- Jika error user baru jauh lebih besar, buat model kedua dengan `GroupKFold` berdasarkan `username` yang tidak memakai identitas user (fitur: teks, embedding, media, device, waktu, statistik akun dari data tanpa target).
- Saat inference: user dikenal pakai model utama, user baru pakai model grup (atau campuran berbobot yang dioptimasi terpisah per kelompok).

## 6. Langkah 5: Tangani Likes Nol (25,66%)

- Latih klasifier `P(likes = 0)` (LightGBM binary).
- Latih regresor pada subset `likes > 0` dengan target `log1p(likes)`.
- Gabungan di ruang log: `z_hat = (1 - p0) * z_pos` sebagai kandidat. Alternatif: jadikan `p0` dan `z_pos` (prediksi OOF) fitur untuk model utama.
- Evaluasi RMSLE gabungan di OOF; pertahankan hanya jika membaik.

## 7. Langkah 6: Tuning dan Stabilitas

- Loss L2 pada `z = log1p(likes)`. Bandingkan Huber hanya sebagai variasi; nilai evaluasi tetap RMSE di ruang log.
- Tuning Optuna (50-100 trial) untuk LightGBM dan CatBoost; `early_stopping` di setiap fold.
- Rata-rata 3-5 seed per model.
- CatBoost dengan `username` sebagai fitur kategorikal native, sebagai satu model tersendiri di ensemble.
- Jaga keragaman model: GBDT dengan fitur berbeda (dengan atau tanpa embedding, dengan atau tanpa fitur kedekatan waktu) lebih berguna daripada banyak model yang mirip.

## 8. Langkah 7: Stacking Akhir

- Kumpulkan OOF semua model di ruang log.
- Meta-model sederhana (Ridge atau LightGBM kecil) dengan CV di atas OOF, atau blending berbobot dari Langkah 1.
- Hindari mengoptimalkan bobot dan mengevaluasi pada data yang sama tanpa pengaman; gunakan validasi bersarang bila memungkinkan.
- Pasca-proses: `expm1`, `clip(min=0)`. Cek di OOF apakah pembulatan ke integer membantu (biasanya tidak untuk RMSLE).

## 9. Jadwal Kerja (sampai 16 Nov 2026, 23.59 WIB)

| Hari | Fokus |
|---|---|
| H1 | Langkah 1 (selaraskan RMSLE) + Langkah 2 (fitur kedekatan waktu); submit untuk mengukur lompatan |
| H2 | Langkah 3 (embedding teks) dan integrasi ke GBDT |
| H3 | Langkah 4 (user baru) dan Langkah 5 (hurdle model) |
| H4 | Tuning, multi-seed, stacking |
| H5 | Finalisasi notebook, uji jalan ulang dari nol, cek ukuran dan kepatuhan; submit terakhir **sebelum deadline** |

Sisakan waktu cadangan; jangan menunggu menit terakhir untuk mengunggah.

## 10. Catatan Pengukuran

Untuk setiap eksperimen, catat di tabel:

| Eksperimen | CV RMSLE | Skor publik | Dipertahankan? |
|---|---|---|---|
| Baseline | | 1.46153 | |
| + selaras RMSLE | | | |
| + kedekatan waktu | | | |
| + embedding | | | |
| + hurdle / user baru | | | |
| Stacking akhir | | | |

Aturan keputusan:
- Pertahankan perubahan yang memperbaiki CV **dan** skor publik secara konsisten.
- Jika CV membaik tapi skor publik memburuk, curigai kebocoran atau distribution shift; periksa ulang skema OOF sebelum melanjutkan.
- Jika hanya skor publik yang membaik tanpa dukungan CV, anggap itu noise.

## 11. Checklist Pengumpulan Akhir

- [ ] Notebook < 25 MB, output besar dibersihkan
- [ ] Bagian sumber pretrained model, link repo HF, library dan versi tercantum (atau keterangan tidak memakai pretrained model)
- [ ] Repo HF public dan bisa di-load tanpa token (jika ada weights terpisah)
- [ ] Kode preprocessing, konfigurasi, training, inference lengkap
- [ ] Restart and Run All berhasil dari awal
- [ ] `submission.csv`: 5.969 baris, kolom `id` dan `likes`, `id` identik dengan `sample_submission.csv`, tanpa NaN dan nilai negatif
- [ ] File yang diunggah adalah versi terbaik (bukan salah satu versi percobaan); cek halaman pengumpulan di dm.helloprodigi.pro untuk file apa saja yang diminta

## 12. Cara Kerja Agent

- Kerjakan berurutan; setelah tiap langkah laporkan CV RMSLE dalam 2-3 kalimat.
- Jika ada keputusan yang tidak bisa dipastikan dari data (misalnya satuan `ts`, bentuk split), berhenti dan tanyakan ke pengguna.
- Jangan mengarang nama repo atau skor. Verifikasi repo HF ada sebelum dipakai, laporkan skor apa adanya.
- Prioritas: keselarasan dengan RMSLE > fitur kedekatan waktu per user > embedding teks > penanganan user baru dan nol > tuning dan stacking.
