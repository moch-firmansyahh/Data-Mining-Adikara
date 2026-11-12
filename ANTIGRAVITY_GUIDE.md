# Panduan Agent: Data Mining ADIKARA 2026 (Prediksi Likes Tweet)

Dokumen ini adalah instruksi kerja untuk agent (Antigravity). Ikuti urutan fase secara berurutan. Jangan melompat ke model rumit sebelum baseline dan validasi yang benar tersedia.

---

## 0. Tujuan dan Batasan

**Tujuan:** memprediksi jumlah `likes` pada tweet berbahasa Indonesia (masalah regresi, target berupa count yang skewed).

**Data (diunduh manual dari dm.helloprodigi.pro, simpan di folder `data/`):**
- `train.csv`: 26.188 baris, kolom `id, username, post_date, text, media_type, client_device, likes`
- `test.csv`: 5.969 baris, tanpa `likes`
- `sample_submission.csv`: 5.969 baris, kolom `id` dan `likes`

**Aturan wajib dari panitia (jangan dilanggar):**
1. Notebook `.ipynb` final maksimal **25 MB**. Bersihkan output besar sebelum simpan (`jupyter nbconvert --clear-output`, jangan simpan embedding atau array besar di notebook).
2. Notebook harus memuat kode **preprocessing, konfigurasi model, training/fine-tuning, dan inference**.
3. Pretrained model boleh dari Hugging Face Hub. Weights/checkpoint hasil training disimpan **terpisah** (misalnya HF Hub), karena panitia tidak menyediakan tempat upload.
4. Repo model yang dipakai evaluasi harus **public**, dapat diakses tanpa API key, login, atau persetujuan.
5. Wajib mencantumkan **sumber pretrained model, link repo, serta library dan versi dependensi** (buat sel markdown khusus di notebook).
6. Training dan inference boleh dipisah. Checkpoint harus bisa di-load untuk inference.
7. Semua kode dan weights harus tetap bisa dijalankan selama masa verifikasi.

**Catatan jujur:** tidak ada jaminan skor tinggi. Targetnya adalah pipeline yang benar, validasinya dapat dipercaya, dan ensemble yang kuat. Skor leaderboard hanya bisa dipastikan setelah submit.

---

## 1. Fase 1: Setup dan Pemeriksaan Awal

1. Buat struktur proyek:
   ```
   project/
     data/            # train.csv, test.csv, sample_submission.csv
     notebooks/       # notebook final + eksperimen
     src/             # fungsi bantu (opsional)
     outputs/         # oof, prediksi, submission
     requirements.txt
   ```
2. Catat versi library di `requirements.txt` (pandas, numpy, scikit-learn, lightgbm, catboost, xgboost, torch, transformers, sentence-transformers, huggingface_hub).
3. Fix seed (`SEED = 42`) di semua tempat (numpy, torch, model GBDT).
4. Cek apakah GPU tersedia. Jika tidak, pilih embedding model yang kecil dan lakukan batching.

## 2. Fase 2: EDA (wajib sebelum modeling)

Jawab pertanyaan berikut dan simpan kesimpulannya di notebook:

- Distribusi `likes`: berapa persen nol? Seberapa panjang ekornya? Bandingkan skala asli dan `log1p`.
- Apakah `post_date` di test berada **di rentang waktu yang sama** dengan train, atau train lebih lama dan test lebih baru (split berbasis waktu)? Ini menentukan strategi validasi.
- Berapa persen `username` di test yang **juga ada di train**? Berapa username unik, dan berapa tweet per username?
- Hubungan `likes` dengan `media_type`, `client_device`, jam, hari, bulan.
- Cek duplikat teks, teks kosong, nilai hilang, kategori langka atau tidak konsisten (huruf besar/kecil, spasi).
- **Adversarial validation:** latih klasifier train-vs-test. Jika AUC jauh di atas 0,5, ada distribution shift dan validasi harus meniru kondisi itu.
- Ada tidaknya kolom `id` yang berkorelasi dengan `likes` atau waktu. Jika `id` berurutan mengikuti waktu, perlakukan dengan hati-hati. Jangan pakai `id` sebagai fitur kecuali terbukti sah dan aman.

## 3. Fase 3: Strategi Validasi (paling penting)

Pilih sesuai temuan EDA:

| Kondisi | Validasi |
|---|---|
| Test acak dan username overlap tinggi | `KFold` 5 fold (stratifikasi dengan bin dari `log1p(likes)`) |
| Test lebih baru dari train | Time-based split (latih masa lalu, validasi masa depan) |
| Username di test banyak yang tidak ada di train | `GroupKFold` berdasarkan `username` |

Aturan:
- Simpan **OOF prediction** untuk setiap model (dipakai untuk blending dan stacking).
- Semua fitur berbasis target (target encoding) harus dihitung **di dalam fold** (out-of-fold), bukan di seluruh train, supaya tidak bocor.
- **Cek metrik evaluasi resmi di platform kompetisi** (RMSE, MAE, RMSLE, atau lainnya). Jika metrik tidak jelas, asumsikan RMSE pada skala asli dan RMSLE sebagai pembanding, lalu catat asumsinya. Pilih target dan loss sesuai metrik:
  - Metrik berbasis log (RMSLE) atau likes sangat skewed: latih pada `log1p(likes)` dengan loss L2, lalu `expm1` untuk prediksi.
  - Metrik RMSE skala asli: bandingkan model `log1p` vs model langsung (atau Tweedie/Poisson) dan pilih yang terbaik di validasi. Ekor panjang bisa membuat satu kesalahan besar mendominasi RMSE.
  - Metrik MAE: pertimbangkan loss L1 atau Huber pada skala log.

## 4. Fase 4: Feature Engineering

### 4.1 Username (biasanya sinyal terkuat)
- Frekuensi username (jumlah tweet di train+test).
- **Target encoding out-of-fold** pada `log1p(likes)`: mean, median, std, max, min, kuantil, jumlah tweet. Gunakan smoothing (bayesian) untuk username langka.
- Fitur turunan: rata-rata panjang teks per user, proporsi media per user, rentang aktif.
- Username tak dikenal di test: isi dengan prior global, dan beri flag `is_unseen_user`.
- Pertimbangkan juga CatBoost dengan `username` sebagai fitur kategorikal (ordered target statistics bawaannya mengurangi leakage).

### 4.2 Waktu (`post_date`)
- Parse ke datetime. Ekstrak: tahun, bulan, hari, hari dalam minggu, jam, menit, akhir pekan, kuartal.
- Fitur siklik (sin/cos untuk jam dan hari dalam minggu).
- Timestamp numerik (selisih terhadap tanggal minimum), umur akun relatif terhadap tweet pertamanya di data.
- Per user: selisih waktu dari tweet sebelumnya dan sesudahnya, urutan tweet, jumlah tweet pada hari yang sama. Hitung ini dari train+test digabung (tanpa target), jadi aman dari kebocoran.

### 4.3 Kategorikal
- `media_type` dan `client_device`: one-hot atau kategorikal native (LightGBM/CatBoost). Normalisasi label dulu.
- Interaksi: `username x media_type`, `media_type x jam`.

### 4.4 Teks (`text`)
Fitur statistik:
- Panjang karakter dan jumlah kata, jumlah hashtag, mention, URL, emoji, tanda seru dan tanya, huruf kapital, angka, rasio non-alfabet, apakah dimulai dengan mention (indikasi balasan), apakah ada "RT", jumlah baris.
- Fitur slang atau singkatan sederhana bila bermanfaat.

Representasi teks:
1. **TF-IDF** (word n-gram 1-2 dan char n-gram 2-5), lalu `TruncatedSVD` ke 64-256 dimensi, atau model linear (Ridge) yang menghasilkan prediksi OOF yang dipakai sebagai fitur untuk GBDT (stacking).
2. **Embedding transformer** (ekstrak embedding lalu masukkan ke GBDT atau Ridge):
   - Kandidat model Indonesia: `indobenchmark/indobert-base-p1`, `indolem/indobertweet-base-uncased`, atau multilingual `intfloat/multilingual-e5-base`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
   - **Verifikasi dulu** bahwa repo itu benar-benar ada dan publik di Hugging Face Hub sebelum memakainya. Jangan menebak nama repo.
   - Reduksi dimensi (PCA/SVD) sebelum masuk GBDT bila perlu.
3. Simpan embedding ke file `.npy` di `outputs/`, **bukan** di notebook.

## 5. Fase 5: Modeling

### 5.1 Baseline (kerjakan dulu, target selesai cepat)
- LightGBM pada fitur tabular + statistik teks + username encoding + waktu.
- Catat skor CV sebagai patokan. Setiap perubahan berikutnya harus dibandingkan dengan patokan ini.

### 5.2 Model lanjutan
- **LightGBM**, **CatBoost** (kategorikal native), **XGBoost** dengan tuning ringan (Optuna, 30-100 trial, early stopping).
- Variasi: pakai atau tanpa embedding, `log1p` vs Tweedie/Poisson, seed berbeda.
- **Opsional (jika GPU ada dan waktu cukup):** fine-tune IndoBERT/IndoBERTweet untuk regresi `log1p(likes)`, dengan input teks, dan metadata (username, media, device) ditambahkan sebagai token teks atau digabung ke head. Simpan checkpoint ke HF Hub (repo **public**) dan muat saat inference.

### 5.3 Ensemble
- Blending berbobot berdasarkan OOF (optimasi bobot dengan `scipy.optimize` atau regresi non-negatif).
- Atau stacking dengan meta-model sederhana (Ridge) di atas OOF semua model.
- Pakai rata-rata beberapa seed per model untuk menstabilkan.
- Pastikan blending dievaluasi dengan cara yang tidak mengintip (nested atau validasi terpisah) agar skor tidak terlalu optimistis.

### 5.4 Pasca-pemrosesan
- Kembalikan ke skala asli (`expm1`), **clip ke minimal 0**.
- Cek apakah pembulatan ke integer membantu atau merugikan metrik, lalu putuskan berdasarkan OOF.
- Bila RMSE skala asli dipakai, periksa bias kalibrasi setelah `expm1` (smearing correction) dan uji apakah perkalian faktor kecil memperbaiki OOF.

## 6. Fase 6: Pencegahan Kebocoran dan Kesalahan Umum

- Jangan fit scaler, encoder, TF-IDF, atau target encoding pada data validasi. Fit hanya pada fold latih.
- Jangan pakai `likes` dari baris lain sebagai fitur tanpa skema OOF.
- Jangan tuning terlalu banyak terhadap satu skema validasi. Jika CV dan leaderboard publik berbeda jauh, curigai distribution shift, bukan sekadar tuning ulang.
- Pastikan urutan baris `test` sama dengan urutan `id` pada `sample_submission.csv`.

## 7. Fase 7: Submission

1. Buat `submission.csv` dengan kolom persis seperti `sample_submission.csv` (`id`, `likes`).
2. Validasi otomatis sebelum menyimpan:
   - Jumlah baris = 5.969
   - `id` identik dengan `sample_submission.csv` (set dan urutan)
   - Tidak ada NaN atau nilai negatif
   - Nama kolom dan tipe data sesuai
3. Simpan beberapa versi (`submission_baseline.csv`, `submission_blend.csv`) dan catat skor CV untuk masing-masing.

## 8. Fase 8: Notebook Final untuk Dikumpulkan

Susun satu notebook yang bersih, urut, dan dapat dijalankan dari atas ke bawah:

1. Judul dan ringkasan metodologi
2. **Sumber pretrained model, link repo HF, library dan versi** (wajib)
3. Load data dan EDA ringkas
4. Preprocessing dan feature engineering
5. Konfigurasi model
6. Training / fine-tuning (boleh dipisah ke notebook lain bila berat, asalkan checkpoint bisa di-load)
7. Inference dan pembuatan submission
8. Cek akhir: ukuran file < 25 MB, repo model public, seluruh sel berjalan tanpa error

Checklist sebelum kumpul:
- [ ] Notebook < 25 MB (output besar dibersihkan)
- [ ] Repo HF public dan dapat di-load tanpa token (uji di sesi atau akun logout)
- [ ] Sumber model, link, dan dependensi tercantum
- [ ] Preprocessing, konfigurasi, training, inference semuanya ada
- [ ] Notebook dijalankan ulang dari awal dan berhasil
- [ ] Submission lolos validasi format

## 9. Cara Kerja Agent

- Kerjakan per fase. Setelah tiap fase, ringkas temuan dan skor CV dalam 2-3 kalimat.
- Jika ada keputusan yang bergantung pada informasi yang belum diketahui (metrik resmi, bentuk split train/test), **berhenti dan tanyakan ke pengguna**, jangan mengasumsikan diam-diam.
- Jangan mengarang nama model, repo, atau skor. Verifikasi repo ada sebelum dipakai, dan laporkan skor apa adanya dari validasi.
- Prioritas: validasi yang benar > fitur username dan waktu > embedding teks > ensemble > fine-tuning berat.
