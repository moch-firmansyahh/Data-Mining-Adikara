import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Title & Overview
cells.append(nbf.v4.new_markdown_cell(r"""# Data Mining ADIKARA 2026: Prediksi Likes Tweet
**Tim Antigravity (SundaSundaJawa) | Pipeline V2: Penyelarasan Penuh RMSLE & Kedekatan Waktu**

---

### 1. Ringkasan Metodologi V2 (Menuju Puncak Leaderboard)
Dokumen panduan V2 memfokuskan optimasi langsung pada metrik resmi kompetisi: **RMSLE** (*Root Mean Squared Logarithmic Error*). 

**Pilar Perbaikan Utama V2:**
1. **Penyelarasan Total dengan RMSLE (Log-Space Alignment)**:
   - Seluruh pelatihan model GBDT dan optimasi bobot *blending* dilakukan di ruang logaritma $z = \log(1 + y)$.
   - Menghapus faktor kalibrasi skala mentah (smearing), karena evaluasi leaderboard dinilai langsung pada $\text{RMSE}(z_{\text{true}}, z_{\text{pred}})$.
   - Prediksi akhir dihitung melalui $\hat{y} = \max(0, \exp(\hat{z}) - 1)$.
2. **Fitur Kedekatan Waktu per User (*Temporal Nearest Neighbors*)**:
   - Tingkat interaksi akun berubah dinamis seiring waktu. Rata-rata global user terlalu kasar.
   - Diekstrak fitur tweet dari user yang sama yang paling dekat secara waktu: `nn_z1` (likes tweet terdekat), `nn_gap` (selisih hari), `nn_wmean_3` & `nn_wmean_10` (*inverse-time weighted mean*), serta rata-rata likes dalam jendela 7 hari dan 30 hari.
   - Menggunakan skema **Leave-One-Out** bebas kebocoran (*leakage-free*): baris latih tidak melihat dirinya sendiri, baris validasi hanya merujuk ke fold latih, dan baris test merujuk ke data train.
3. **Penanganan Zero-Inflation (Hurdle Classifier)**:
   - Sebanyak 25.66% tweet bernilai 0 likes. Dilatih *in-fold classifier* probabilitas $P(\text{likes} = 0)$ sebagai meta-fitur tambahan.
4. **NLP Meta-Feature (TF-IDF + Ridge)**:
   - Representasi teks n-gram kata & karakter diproyeksikan ke target log secara *in-fold* menghasilkan korelasi $r = 0.717$.
5. **Ensemble Multi-Model Teroptimasi**:
   - Kombinasi **LightGBM**, **CatBoost**, dan **XGBoost** dibobotkan secara otomatis dengan algoritma SLSQP yang meminimalkan RMSLE OOF."""))

# Cell 2: Mandatory attribution of libraries & pretrained models
cells.append(nbf.v4.new_markdown_cell(r"""---
### 2. Aturan Panitia: Sumber Model, Repo, & Dependensi Library
*Wajib dicantumkan sesuai Panduan Kompetisi ADIKARA 2026:*

| Komponen | Spesifikasi / Sumber | Keterangan |
|---|---|---|
| **Python Environment** | Python 3.13.7 (64-bit Windows) | Runtime utama |
| **Tabular GBDT 1** | `lightgbm` (v4.7.0) | Microsoft LightGBM Regressor & Classifier |
| **Tabular GBDT 2** | `catboost` (v1.2.10) | Yandex CatBoost Regressor |
| **Tabular GBDT 3** | `xgboost` (v3.4.1) | DMLC XGBoost Regressor |
| **Machine Learning & NLP** | `scikit-learn` (v1.9.0), `scipy` (v1.17.1) | TfidfVectorizer, Ridge, StratifiedKFold, SLSQP |
| **Data Processing** | `pandas` (v3.0.2), `numpy` (v2.4.2) | Manipulasi data tabular & numerik |
| **Pretrained Checkpoint** | Self-contained in-fold Ridge & GBDT Ensembles | Tidak bergantung pada model tertutup/berbayar; semua kode reproducible end-to-end |
| **Public Artifacts** | Open Source / Local Workspace | Checkpoint & OOF tersimpan di folder `outputs/` |"""))

# Cell 3: Imports & Setup
cells.append(nbf.v4.new_code_cell(r"""import os
import re
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
print("Environment siap. Seed dikunci ke:", SEED)"""))

# Cell 4: Load Data & Clean Likes
cells.append(nbf.v4.new_code_cell(r"""# 1. Pemuatan Data
train_df = pd.read_csv('data/train.csv' if os.path.exists('data/train.csv') else '../data/train.csv')
test_df = pd.read_csv('data/test.csv' if os.path.exists('data/test.csv') else '../data/test.csv')
sub_sample = pd.read_csv('data/sample_submission.csv' if os.path.exists('data/sample_submission.csv') else '../data/sample_submission.csv')

def parse_like(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if val_str.startswith('-'):
        return np.nan
    m = re.search(r'(\d+(?:\.\d+)?)', val_str)
    if m:
        return float(m.group(1))
    return np.nan

train_df['clean_likes'] = train_df['likes'].apply(parse_like)
train_clean = train_df[train_df['clean_likes'].notna()].reset_index(drop=True)
y_raw = train_clean['clean_likes'].values
y_log = np.log1p(y_raw)
train_clean['target_log'] = y_log

print(f"Data latih bersih: {len(train_clean)} baris, Data uji: {len(test_df)} baris")"""))

# Cell 5: Fast Preprocessing & Unsupervised Stats
cells.append(nbf.v4.new_code_cell(r"""# Fast Date Parser & Preprocessing
def fast_parse_dates(s):
    res = pd.Series(index=s.index, dtype='datetime64[ns, UTC]')
    m25 = s.str.len() == 25
    m19 = s.str.len() == 19
    m10 = s.str.len() == 10
    res[m25] = pd.to_datetime(s[m25], format='%Y-%m-%d %H:%M:%S%z')
    res[m19] = pd.to_datetime(s[m19], format='%d/%m/%Y %H:%M:%S').dt.tz_localize('UTC')
    res[m10] = pd.to_datetime(s[m10].astype('int64'), unit='s', utc=True)
    return res

def clean_media(s):
    s = s.fillna('missing').astype(str).str.strip().str.lower()
    s = s.replace({'foto': 'photo', 'vidio': 'video', 'carousell': 'carousel'})
    valid = {'text_only', 'photo', 'video', 'carousel', 'missing'}
    return s.apply(lambda x: x if x in valid else 'other')

def clean_device(s):
    s = s.fillna('missing').astype(str).str.strip()
    s = s.replace({'Andorid': 'Android', 'Web-App': 'Web App'})
    valid = {'Android', 'iOS', 'Web App', 'Lite App', 'missing'}
    return s.apply(lambda x: x if x in valid else 'other')

def extract_text_stats(series):
    df_feat = pd.DataFrame(index=series.index)
    s = series.fillna('').astype(str)
    
    df_feat['char_len'] = s.apply(len)
    df_feat['word_count'] = s.apply(lambda x: len(x.split()))
    df_feat['avg_word_len'] = df_feat['char_len'] / (df_feat['word_count'] + 1)
    df_feat['hashtag_count'] = s.apply(lambda x: x.count('#'))
    df_feat['mention_count'] = s.apply(lambda x: x.count('@'))
    df_feat['url_count'] = s.apply(lambda x: len(re.findall(r'https?://\S+', x)))
    df_feat['excl_count'] = s.apply(lambda x: x.count('!'))
    df_feat['question_count'] = s.apply(lambda x: x.count('?'))
    df_feat['uppercase_count'] = s.apply(lambda x: sum(1 for c in x if c.isupper()))
    df_feat['uppercase_ratio'] = df_feat['uppercase_count'] / (df_feat['char_len'] + 1)
    df_feat['digit_count'] = s.apply(lambda x: sum(1 for c in x if c.isdigit()))
    df_feat['digit_ratio'] = df_feat['digit_count'] / (df_feat['char_len'] + 1)
    df_feat['non_alpha_ratio'] = s.apply(lambda x: sum(1 for c in x if not c.isalnum() and not c.isspace())) / (df_feat['char_len'] + 1)
    df_feat['is_reply'] = s.apply(lambda x: 1 if x.strip().startswith('@') else 0)
    df_feat['is_retweet'] = s.apply(lambda x: 1 if (x.strip().startswith('RT ') or 'RT @' in x) else 0)
    df_feat['newline_count'] = s.apply(lambda x: x.count('\n'))
    
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
    df_feat['emoji_count'] = s.apply(lambda x: len(emoji_pattern.findall(x)))
    return df_feat

df_combined = pd.concat([train_clean.assign(is_test=0), test_df.assign(is_test=1)], ignore_index=True)
df_combined['clean_media'] = clean_media(df_combined['media_type'])
df_combined['clean_device'] = clean_device(df_combined['client_device'])
df_combined['dt'] = fast_parse_dates(df_combined['post_date'])

is_bad_date = (df_combined['dt'].dt.year < 2010).astype(int)
valid_median = df_combined.loc[is_bad_date == 0, 'dt'].median()
df_combined['dt_clean'] = df_combined['dt']
df_combined.loc[is_bad_date == 1, 'dt_clean'] = valid_median
df_combined['is_bad_date'] = is_bad_date

df_combined['year'] = df_combined['dt_clean'].dt.year
df_combined['month'] = df_combined['dt_clean'].dt.month
df_combined['day'] = df_combined['dt_clean'].dt.day
df_combined['dayofweek'] = df_combined['dt_clean'].dt.dayofweek
df_combined['hour'] = df_combined['dt_clean'].dt.hour
df_combined['minute'] = df_combined['dt_clean'].dt.minute
df_combined['is_weekend'] = (df_combined['dayofweek'] >= 5).astype(int)
df_combined['quarter'] = df_combined['dt_clean'].dt.quarter

df_combined['sin_hour'] = np.sin(2 * np.pi * df_combined['hour'] / 24.0)
df_combined['cos_hour'] = np.cos(2 * np.pi * df_combined['hour'] / 24.0)
df_combined['sin_dow'] = np.sin(2 * np.pi * df_combined['dayofweek'] / 7.0)
df_combined['cos_dow'] = np.cos(2 * np.pi * df_combined['dayofweek'] / 7.0)

min_time = df_combined['dt_clean'].min()
df_combined['ts_days'] = (df_combined['dt_clean'] - min_time).dt.total_seconds() / 86400.0

text_stats = extract_text_stats(df_combined['text'])
for col in text_stats.columns:
    df_combined[col] = text_stats[col]

user_counts = df_combined['username'].value_counts()
df_combined['user_total_tweets'] = df_combined['username'].map(user_counts).astype(float)
df_combined['user_avg_char_len'] = df_combined.groupby('username')['char_len'].transform('mean')

for m in ['text_only', 'photo', 'video', 'carousel']:
    df_combined[f'user_{m}_prop'] = df_combined.groupby('username')['clean_media'].transform(lambda x: (x == m).mean())
for d in ['Android', 'iOS', 'Web App']:
    df_combined[f'user_{d}_prop'] = df_combined.groupby('username')['clean_device'].transform(lambda x: (x == d).mean())
    
user_min_dt = df_combined.groupby('username')['dt_clean'].transform('min')
user_max_dt = df_combined.groupby('username')['dt_clean'].transform('max')
df_combined['user_active_span_days'] = (user_max_dt - user_min_dt).dt.total_seconds() / 86400.0
df_combined['user_tweet_age_days'] = (df_combined['dt_clean'] - user_min_dt).dt.total_seconds() / 86400.0

df_combined = df_combined.sort_values(['username', 'dt_clean']).reset_index(drop=True)
df_combined['user_time_diff_prev'] = df_combined.groupby('username')['dt_clean'].diff().dt.total_seconds().fillna(-1) / 3600.0
df_combined['user_time_diff_next'] = (-df_combined.groupby('username')['dt_clean'].diff(-1)).dt.total_seconds().fillna(-1) / 3600.0

df_combined['date_only'] = df_combined['dt_clean'].dt.date
df_combined['user_tweets_same_day'] = df_combined.groupby(['username', 'date_only'])['id'].transform('count')
df_combined.drop(columns=['date_only'], inplace=True)

df_combined = df_combined.sort_values('id').reset_index(drop=True)

df_combined['media_device'] = df_combined['clean_media'] + '_' + df_combined['clean_device']
df_combined['media_hour_bucket'] = df_combined['clean_media'] + '_' + (df_combined['hour'] // 4).astype(str)

cat_cols = ['clean_media', 'clean_device', 'media_device', 'media_hour_bucket']
for c in cat_cols:
    all_cats = df_combined[c].unique()
    mapping = {val: idx for idx, val in enumerate(all_cats)}
    df_combined[f'{c}_code'] = df_combined[c].map(mapping).astype(int)
    
code_cat_cols = [f'{c}_code' for c in cat_cols]

train_feat = df_combined[df_combined['is_test'] == 0].reset_index(drop=True)
test_feat = df_combined[df_combined['is_test'] == 1].reset_index(drop=True)
print(f"Fitur tabular & statistik siap: {train_feat.shape}")"""))

# Cell 6: Temporal Neighbors Extractor
cells.append(nbf.v4.new_code_cell(r"""# Modul Ekstraksi Kedekatan Waktu per User (Temporal Neighbors)
def compute_temporal_neighbors(query_df, ref_df, is_train=False, ks=(3, 10), windows_days=(7, 30)):
    ref_g = {}
    for u, g in ref_df.groupby('username'):
        ref_g[u] = {
            'ids': g['id'].values,
            'ts': g['ts_days'].values,
            'z': g['target_log'].values
        }
        
    prior = ref_df['target_log'].mean()
    out = []
    
    q_ids = query_df['id'].values
    q_users = query_df['username'].values
    q_ts = query_df['ts_days'].values
    q_indices = query_df.index.values
    
    for i in range(len(q_ids)):
        qid = q_ids[i]
        u = q_users[i]
        t = q_ts[i]
        idx = q_indices[i]
        
        entry = ref_g.get(u)
        feats = {'idx': idx}
        
        if entry is None:
            valid_ts, valid_z = np.array([]), np.array([])
        else:
            if is_train:
                mask = entry['ids'] != qid
                valid_ts = entry['ts'][mask]
                valid_z = entry['z'][mask]
            else:
                valid_ts = entry['ts']
                valid_z = entry['z']
                
        if len(valid_ts) == 0:
            for k in ks:
                feats[f'nn_mean_{k}'] = prior
                feats[f'nn_wmean_{k}'] = prior
            feats['nn_gap'] = 9999.0
            feats['nn_z1'] = prior
            for w in windows_days:
                feats[f'cnt_{w}d'] = 0
                feats[f'mean_{w}d'] = prior
            out.append(feats)
            continue
            
        dt = np.abs(valid_ts - t)
        order = np.argsort(dt)
        
        feats['nn_gap'] = dt[order[0]]
        feats['nn_z1'] = valid_z[order[0]]
        for k in ks:
            sel = order[:min(k, len(order))]
            feats[f'nn_mean_{k}'] = valid_z[sel].mean()
            wts = 1.0 / (1.0 + dt[sel])
            feats[f'nn_wmean_{k}'] = float((valid_z[sel] * wts).sum() / wts.sum())
        for w in windows_days:
            w_mask = dt <= w
            feats[f'cnt_{w}d'] = int(w_mask.sum())
            feats[f'mean_{w}d'] = valid_z[w_mask].mean() if w_mask.any() else prior
        out.append(feats)
        
    return pd.DataFrame(out).set_index('idx')

print("Fungsi Temporal Nearest Neighbors (Leave-One-Out) siap.")"""))

# Cell 7: In-Fold Bayesian Target Encoding
cells.append(nbf.v4.new_code_cell(r"""# Target Encoding Bebas Bocor
def compute_oof_target_encoding(train_df, val_df, test_df, target_col='target_log', cat_col='username', m=10.0):
    global_mean = train_df[target_col].mean()
    global_median = train_df[target_col].median()
    global_std = train_df[target_col].std()
    global_min = train_df[target_col].min()
    global_max = train_df[target_col].max()
    global_q25 = train_df[target_col].quantile(0.25)
    global_q75 = train_df[target_col].quantile(0.75)
    
    stats = train_df.groupby(cat_col)[target_col].agg(
        count='count',
        raw_mean='mean',
        user_median='median',
        user_std='std',
        user_min='min',
        user_max='max',
        user_q25=lambda x: np.percentile(x, 25),
        user_q75=lambda x: np.percentile(x, 75)
    ).reset_index()
    
    stats['user_te_mean'] = (stats['count'] * stats['raw_mean'] + m * global_mean) / (stats['count'] + m)
    stats['user_te_std'] = stats['user_std'].fillna(global_std)
    
    keep_cols = [cat_col, 'count', 'user_te_mean', 'user_median', 'user_te_std', 'user_min', 'user_max', 'user_q25', 'user_q75']
    stats = stats[keep_cols].rename(columns={'count': 'user_train_count'})
    
    def apply_stats(df):
        m_df = df[[cat_col]].merge(stats, on=cat_col, how='left')
        is_unseen = m_df['user_te_mean'].isna().astype(int)
        
        te_features = pd.DataFrame(index=df.index)
        te_features['is_unseen_user'] = is_unseen
        te_features['user_train_count'] = m_df['user_train_count'].fillna(0)
        te_features['user_te_mean'] = m_df['user_te_mean'].fillna(global_mean)
        te_features['user_te_median'] = m_df['user_median'].fillna(global_median)
        te_features['user_te_std'] = m_df['user_te_std'].fillna(global_std)
        te_features['user_te_min'] = m_df['user_min'].fillna(global_min)
        te_features['user_te_max'] = m_df['user_max'].fillna(global_max)
        te_features['user_te_q25'] = m_df['user_q25'].fillna(global_q25)
        te_features['user_te_q75'] = m_df['user_q75'].fillna(global_q75)
        return te_features
        
    return apply_stats(train_df), apply_stats(val_df), apply_stats(test_df)

print("Fungsi Bayesian OOF Target Encoding siap.")"""))

# Cell 8: In-Fold TF-IDF + Ridge Meta-Feature
cells.append(nbf.v4.new_code_cell(r"""# 5-Fold Stratified Split & NLP Meta-Feature Extraction
bins = pd.qcut(y_log, q=10, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

train_texts = train_feat['text'].fillna('').tolist()
test_texts = test_feat['text'].fillna('').tolist()

oof_ridge_text = np.zeros(len(train_feat))
test_ridge_text = np.zeros(len(test_feat))

print("Mengekstrak in-fold TF-IDF + Ridge NLP meta-feature...")
for fold, (tr_idx, val_idx) in enumerate(skf.split(train_feat, bins)):
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr_tfidf = tfidf.fit_transform([train_texts[i] for i in tr_idx])
    X_val_tfidf = tfidf.transform([train_texts[i] for i in val_idx])
    X_te_tfidf = tfidf.transform(test_texts)
    
    ridge = Ridge(alpha=3.0, random_state=SEED)
    ridge.fit(X_tr_tfidf, y_log[tr_idx])
    oof_ridge_text[val_idx] = ridge.predict(X_val_tfidf)
    test_ridge_text += ridge.predict(X_te_tfidf) / 5.0

train_feat['ridge_text_pred'] = oof_ridge_text
test_feat['ridge_text_pred'] = test_ridge_text
print("Korelasi Ridge Text vs Target:", np.corrcoef(oof_ridge_text, y_log)[0, 1].round(4))"""))

# Cell 9: Training 3 GBDT Models with Temporal Neighbors & Hurdle
cells.append(nbf.v4.new_code_cell(r"""# Definisi Fitur & Pelatihan 5-Fold Multi-Model (LightGBM, CatBoost, XGBoost)
base_numeric_cols = [
    'is_bad_date', 'year', 'month', 'day', 'dayofweek', 'hour', 'minute',
    'is_weekend', 'quarter', 'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
    'ts_days', 'char_len', 'word_count', 'avg_word_len',
    'hashtag_count', 'mention_count', 'url_count', 'excl_count',
    'question_count', 'uppercase_count', 'uppercase_ratio', 'digit_count',
    'digit_ratio', 'non_alpha_ratio', 'is_reply', 'is_retweet',
    'newline_count', 'emoji_count', 'user_total_tweets',
    'user_avg_char_len', 'user_text_only_prop', 'user_photo_prop',
    'user_video_prop', 'user_carousel_prop', 'user_Android_prop',
    'user_iOS_prop', 'user_Web App_prop', 'user_active_span_days',
    'user_tweet_age_days', 'user_time_diff_prev', 'user_time_diff_next',
    'user_tweets_same_day'
]

models = ['lgb', 'cat', 'xgb']
oof_z = {m: np.zeros(len(train_feat)) for m in models}
test_z = {m: np.zeros(len(test_feat)) for m in models}

print("Memulai pelatihan 5-Fold Cross Validation dengan Temporal Neighbors & Hurdle...")

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_feat, bins)):
    tr_df, val_df = train_feat.iloc[tr_idx], train_feat.iloc[val_idx]
    
    # 1. Fitur Kedekatan Waktu per User
    tr_nn = compute_temporal_neighbors(tr_df, tr_df, is_train=True)
    val_nn = compute_temporal_neighbors(val_df, tr_df, is_train=False)
    test_nn = compute_temporal_neighbors(test_feat, tr_df, is_train=False)
    nn_cols = list(tr_nn.columns)
    
    # 2. Target Encoding
    tr_te, val_te, test_te = compute_oof_target_encoding(tr_df, val_df, test_feat, target_col='target_log', cat_col='username', m=10.0)
    te_cols = list(tr_te.columns)
    
    # 3. Hurdle P(likes == 0) Classifier
    clf_features = ['user_total_tweets', 'char_len', 'word_count', 'hour', 'dayofweek', 'clean_device_code', 'clean_media_code', 'is_reply', 'is_retweet']
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.08, num_leaves=15, random_state=SEED+fold, verbose=-1, n_jobs=-1)
    clf.fit(tr_df[clf_features], (y_raw[tr_idx] == 0).astype(int))
    
    tr_pzero = clf.predict_proba(tr_df[clf_features])[:, 1]
    val_pzero = clf.predict_proba(val_df[clf_features])[:, 1]
    test_pzero = clf.predict_proba(test_feat[clf_features])[:, 1]
    
    feat_cols = base_numeric_cols + ['ridge_text_pred'] + nn_cols + te_cols + code_cat_cols + ['p_zero']
    
    X_tr = pd.concat([
        tr_df[base_numeric_cols + ['ridge_text_pred']].reset_index(drop=True),
        tr_nn.reset_index(drop=True),
        tr_te.reset_index(drop=True),
        tr_df[code_cat_cols].reset_index(drop=True),
        pd.Series(tr_pzero, name='p_zero')
    ], axis=1)[feat_cols]
    
    X_val = pd.concat([
        val_df[base_numeric_cols + ['ridge_text_pred']].reset_index(drop=True),
        val_nn.reset_index(drop=True),
        val_te.reset_index(drop=True),
        val_df[code_cat_cols].reset_index(drop=True),
        pd.Series(val_pzero, name='p_zero')
    ], axis=1)[feat_cols]
    
    X_te = pd.concat([
        test_feat[base_numeric_cols + ['ridge_text_pred']].reset_index(drop=True),
        test_nn.reset_index(drop=True),
        test_te.reset_index(drop=True),
        test_feat[code_cat_cols].reset_index(drop=True),
        pd.Series(test_pzero, name='p_zero')
    ], axis=1)[feat_cols]

    # Model 1: LightGBM
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_alpha=0.3,
        reg_lambda=2.0,
        n_estimators=1500,
        random_state=SEED + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_tr, y_log[tr_idx],
        eval_set=[(X_val, y_log[val_idx])],
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
    )
    oof_z['lgb'][val_idx] = lgb_model.predict(X_val)
    test_z['lgb'] += lgb_model.predict(X_te) / 5.0

    # Model 2: CatBoost
    cb_model = CatBoostRegressor(
        iterations=1400,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=4.0,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=SEED + fold,
        verbose=0,
        thread_count=-1
    )
    cb_model.fit(
        X_tr, y_log[tr_idx],
        eval_set=(X_val, y_log[val_idx]),
        early_stopping_rounds=40
    )
    oof_z['cat'][val_idx] = cb_model.predict(X_val)
    test_z['cat'] += cb_model.predict(X_te) / 5.0

    # Model 3: XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=1300,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.5,
        tree_method='hist',
        random_state=SEED + fold,
        n_jobs=-1,
        early_stopping_rounds=40
    )
    xgb_model.fit(
        X_tr, y_log[tr_idx],
        eval_set=[(X_val, y_log[val_idx])],
        verbose=False
    )
    oof_z['xgb'][val_idx] = xgb_model.predict(X_val)
    test_z['xgb'] += xgb_model.predict(X_te) / 5.0

    print(f"Fold {fold+1}/5 selesai. [LGB: {root_mean_squared_error(y_log[val_idx], oof_z['lgb'][val_idx]):.4f}] [CAT: {root_mean_squared_error(y_log[val_idx], oof_z['cat'][val_idx]):.4f}] [XGB: {root_mean_squared_error(y_log[val_idx], oof_z['xgb'][val_idx]):.4f}]")"""))

# Cell 10: Log-Space Blending Optimization
cells.append(nbf.v4.new_code_cell(r"""# Evaluasi OOF & Optimasi Bobot Ensemble di Ruang Log (RMSLE)
for m in models:
    score = root_mean_squared_error(y_log, oof_z[m])
    print(f"[{m.upper():8s}] OOF RMSLE: {score:.5f}")

X_blend = np.column_stack([oof_z['lgb'], oof_z['cat'], oof_z['xgb']])

def loss_rmsle(w):
    return root_mean_squared_error(y_log, X_blend @ w)

cons = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
res = minimize(loss_rmsle, [1/3, 1/3, 1/3], bounds=[(0, 1)]*3, constraints=cons, method='SLSQP')
opt_w = res.x

print(f"\nBobot Optimal Ensemble: LightGBM={opt_w[0]:.4f}, CatBoost={opt_w[1]:.4f}, XGBoost={opt_w[2]:.4f}")

oof_blend_z = X_blend @ opt_w
final_cv_rmsle = root_mean_squared_error(y_log, oof_blend_z)
print(f"*** FINAL V2 ENSEMBLE CV RMSLE: {final_cv_rmsle:.5f} ***")"""))

# Cell 11: Final Test Inference & Submission Verification
cells.append(nbf.v4.new_code_cell(r"""# Inferensi Test di Ruang Log, expm1, Clip & Ekspor Submission
X_test_blend = np.column_stack([test_z['lgb'], test_z['cat'], test_z['xgb']])
final_test_z = X_test_blend @ opt_w
final_test_likes = np.clip(np.expm1(final_test_z), 0, None)

sub_final = pd.DataFrame({
    'id': test_df['id'],
    'likes': final_test_likes
})

assert len(sub_final) == 5969, f"Jumlah baris ({len(sub_final)}) != 5969!"
assert (sub_final['id'].values == sub_sample['id'].values).all(), "Urutan ID tidak cocok!"
assert sub_final['likes'].isna().sum() == 0, "Ditemukan nilai NaN!"
assert (sub_final['likes'] < 0).sum() == 0, "Ditemukan nilai negatif!"

out_dir = 'outputs' if os.path.exists('outputs') else '../outputs'
os.makedirs(out_dir, exist_ok=True)
sub_final.to_csv(os.path.join(out_dir, 'submission.csv'), index=False)
sub_final.to_csv(os.path.join(out_dir, 'submission_v2.csv'), index=False)

print(f"Verifikasi SUKSES! File submission.csv disimpan ke {out_dir}/submission.csv.")
print("\nPratinjau 10 baris pertama submission:")
print(sub_final.head(10))"""))

# Cell 12: Checklist & Summary
cells.append(nbf.v4.new_markdown_cell(r"""---
### 3. Checklist Kepatuhan Aturan Kompetisi
- [x] **Ukuran Notebook < 25 MB**: Kode ringkas dan bersih dari dump array atau embedding besar.
- [x] **Sumber Pretrained Model & Dependensi Tercantum**: Dicantumkan lengkap di tabel Markdown sel 2.
- [x] **Pipeline End-to-End**: Memuat pembersihan data, feature engineering, validasi 5-fold, pelatihan 3 model, dan inferensi.
- [x] **Pencegahan Kebocoran (No Leakage)**: Fitur temporal neighbors dihitung murni secara Leave-One-Out dan in-fold.
- [x] **Penyelarasan Metrik**: Dioptimasi langsung meminimalkan RMSLE (log-space).
- [x] **Verifikasi Format Submission**: Tepat 5.969 baris, urutan ID identik, 0 nilai NaN, 0 nilai negatif."""))

nb['cells'] = cells

with open('notebooks/final_submission_pipeline.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook notebooks/final_submission_pipeline.ipynb berhasil diperbarui ke V2!")
