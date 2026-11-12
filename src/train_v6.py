import os
import re
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings('ignore')

SEEDS = [42, 2026, 777]
PRIMARY_SEED = 42
os.makedirs('outputs', exist_ok=True)

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

def extract_text_stats_v6(series):
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
    
    cs_regex = re.compile(r'(hai kak|hi kak|halo kak|mohon maaf|ketidaknyamanan|dm ya|bisa dm|hubungi|kendala|bantu cek|silakan|solusi|agen|nomor pesanan|smart tutor|0804|081|layanan|keluhan|kami bantu|halo sobat|halo kakak|terima kasih telah|no tiket|bisa diinfokan|infokan)', re.I)
    phone_regex = re.compile(r'\b(?:\+?62|08|\d{4})[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b')
    promo_regex = re.compile(r'(promo|diskon|cashback|voucher|giveaway|gratis|free|kupon|exclusive|menangkan|hadiah|belanja|special offer)', re.I)
    
    df_feat['is_cs_reply'] = s.apply(lambda x: 1 if cs_regex.search(x) else 0)
    df_feat['has_phone'] = s.apply(lambda x: 1 if phone_regex.search(x) else 0)
    df_feat['is_promo'] = s.apply(lambda x: 1 if promo_regex.search(x) else 0)
    
    return df_feat

def compute_temporal_neighbors_v6(query_df, ref_df, is_train=False, ks=(3, 10), windows_days=(7, 30), unseen_prior_z=4.85, unseen_prior_zrate=0.06):
    ref_g = {}
    for u, g in ref_df.groupby('username'):
        ref_g[u] = {
            'ids': g['id'].values,
            'ts': g['ts_days'].values,
            'z': g['target_log'].values,
            'is_z': (g['target_log'].values == 0).astype(float)
        }
        
    out = []
    q_ids = query_df['id'].values
    q_users = query_df['username'].values
    q_ts = query_df['ts_days'].values
    q_indices = query_df.index.values
    
    cols = ['idx', 'nn_gap', 'nn_z1']
    for k in ks:
        cols.extend([f'nn_mean_{k}', f'nn_wmean_{k}', f'nn_zero_ratio_{k}'])
    for w in windows_days:
        cols.extend([f'cnt_{w}d', f'mean_{w}d'])
        
    for i in range(len(q_ids)):
        qid = q_ids[i]
        u = q_users[i]
        t = q_ts[i]
        idx = q_indices[i]
        
        entry = ref_g.get(u)
        feats = {'idx': idx}
        
        if entry is None or len(entry['ts']) == 0:
            valid_ts, valid_z, valid_isz = np.array([]), np.array([]), np.array([])
        else:
            if is_train:
                mask = entry['ids'] != qid
                valid_ts = entry['ts'][mask]
                valid_z = entry['z'][mask]
                valid_isz = entry['is_z'][mask]
            else:
                valid_ts = entry['ts']
                valid_z = entry['z']
                valid_isz = entry['is_z']
                
        if len(valid_ts) == 0:
            feats['nn_gap'] = 9999.0
            feats['nn_z1'] = unseen_prior_z
            for k in ks:
                feats[f'nn_mean_{k}'] = unseen_prior_z
                feats[f'nn_wmean_{k}'] = unseen_prior_z
                feats[f'nn_zero_ratio_{k}'] = unseen_prior_zrate
            for w in windows_days:
                feats[f'cnt_{w}d'] = 0
                feats[f'mean_{w}d'] = unseen_prior_z
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
            feats[f'nn_zero_ratio_{k}'] = valid_isz[sel].mean()
            
        for w in windows_days:
            w_mask = dt <= w
            feats[f'cnt_{w}d'] = int(w_mask.sum())
            feats[f'mean_{w}d'] = valid_z[w_mask].mean() if w_mask.any() else unseen_prior_z
        out.append(feats)
        
    df_out = pd.DataFrame(out).set_index('idx')
    expected_order = [c for c in cols if c != 'idx']
    return df_out[expected_order]

def compute_leak_free_target_encoding_v6(tr_df, val_df, test_df, target_col='target_log', cat_col='username', m=15.0, n_inner=5):
    u_counts = tr_df[cat_col].value_counts()
    low_freq_users = set(u_counts[u_counts <= 2].index)
    low_freq_mask = tr_df[cat_col].isin(low_freq_users)
    
    unseen_prior_mean = tr_df.loc[low_freq_mask, target_col].mean() if low_freq_mask.any() else 4.85
    unseen_prior_median = tr_df.loc[low_freq_mask, target_col].median() if low_freq_mask.any() else 5.10
    unseen_prior_std = tr_df.loc[low_freq_mask, target_col].std() if low_freq_mask.any() else 1.05
    unseen_prior_zrate = (tr_df.loc[low_freq_mask, target_col] == 0).mean() if low_freq_mask.any() else 0.06
    
    stats = tr_df.groupby(cat_col)[target_col].agg(
        count='count',
        mean='mean',
        median='median',
        std='std',
        z_count=lambda x: (x == 0).sum()
    ).reset_index()
    
    stats['smooth_mean'] = (stats['count'] * stats['mean'] + m * unseen_prior_mean) / (stats['count'] + m)
    stats['smooth_zrate'] = (stats['z_count'] + m * unseen_prior_zrate) / (stats['count'] + m)
    
    cnt_m = dict(zip(stats[cat_col], stats['count']))
    mean_m = dict(zip(stats[cat_col], stats['smooth_mean']))
    med_m = dict(zip(stats[cat_col], stats['median']))
    std_m = dict(zip(stats[cat_col], stats['std']))
    zrate_m = dict(zip(stats[cat_col], stats['smooth_zrate']))
    
    def map_to_df(df):
        out = pd.DataFrame(index=df.index)
        out['user_train_count'] = df[cat_col].map(cnt_m).fillna(0)
        out['is_unseen_user'] = (out['user_train_count'] == 0).astype(int)
        out['user_te_mean'] = df[cat_col].map(mean_m).fillna(unseen_prior_mean)
        out['user_te_median'] = df[cat_col].map(med_m).fillna(unseen_prior_median)
        out['user_te_std'] = df[cat_col].map(std_m).fillna(unseen_prior_std)
        out['user_zero_rate'] = df[cat_col].map(zrate_m).fillna(unseen_prior_zrate)
        return out
        
    val_te = map_to_df(val_df)
    test_te = map_to_df(test_df)
    
    tr_te = pd.DataFrame(index=tr_df.index)
    tr_te['user_train_count'] = 0.0
    tr_te['is_unseen_user'] = 0
    tr_te['user_te_mean'] = unseen_prior_mean
    tr_te['user_te_median'] = unseen_prior_median
    tr_te['user_te_std'] = unseen_prior_std
    tr_te['user_zero_rate'] = unseen_prior_zrate
    
    ikf = KFold(n_splits=n_inner, shuffle=True, random_state=PRIMARY_SEED)
    for in_tr_pos, in_val_pos in ikf.split(tr_df):
        in_tr = tr_df.iloc[in_tr_pos]
        in_val = tr_df.iloc[in_val_pos]
        
        in_u_counts = in_tr[cat_col].value_counts()
        in_low_users = set(in_u_counts[in_u_counts <= 2].index)
        in_low_mask = in_tr[cat_col].isin(in_low_users)
        in_unseen_mean = in_tr.loc[in_low_mask, target_col].mean() if in_low_mask.any() else unseen_prior_mean
        in_unseen_med = in_tr.loc[in_low_mask, target_col].median() if in_low_mask.any() else unseen_prior_median
        in_unseen_std = in_tr.loc[in_low_mask, target_col].std() if in_low_mask.any() else unseen_prior_std
        in_unseen_zrate = (in_tr.loc[in_low_mask, target_col] == 0).mean() if in_low_mask.any() else unseen_prior_zrate
        
        in_st = in_tr.groupby(cat_col)[target_col].agg(
            count='count', mean='mean', median='median', std='std', z_count=lambda x: (x == 0).sum()
        ).reset_index()
        in_st['smooth_mean'] = (in_st['count'] * in_st['mean'] + m * in_unseen_mean) / (in_st['count'] + m)
        in_st['smooth_zrate'] = (in_st['z_count'] + m * in_unseen_zrate) / (in_st['count'] + m)
        
        in_cnt_m = dict(zip(in_st[cat_col], in_st['count']))
        in_mean_m = dict(zip(in_st[cat_col], in_st['smooth_mean']))
        in_med_m = dict(zip(in_st[cat_col], in_st['median']))
        in_std_m = dict(zip(in_st[cat_col], in_st['std']))
        in_zrate_m = dict(zip(in_st[cat_col], in_st['smooth_zrate']))
        
        tr_te.iloc[in_val_pos, 0] = in_val[cat_col].map(in_cnt_m).fillna(0).values
        tr_te.iloc[in_val_pos, 1] = (tr_te.iloc[in_val_pos, 0] == 0).astype(int).values
        tr_te.iloc[in_val_pos, 2] = in_val[cat_col].map(in_mean_m).fillna(in_unseen_mean).values
        tr_te.iloc[in_val_pos, 3] = in_val[cat_col].map(in_med_m).fillna(in_unseen_med).values
        tr_te.iloc[in_val_pos, 4] = in_val[cat_col].map(in_std_m).fillna(in_unseen_std).values
        tr_te.iloc[in_val_pos, 5] = in_val[cat_col].map(in_zrate_m).fillna(in_unseen_zrate).values
        
    return tr_te, val_te, test_te

def build_keras_mlp(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='swish'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(64, activation='swish'),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        layers.Dense(32, activation='swish'),
        layers.Dense(1)
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.003), loss='mse')
    return model

def run_v6():
    print("="*75)
    print("=== ADIKARA 2026: V6 ULTRA-CHAMPION (TWO-STAGE HURDLE + MULTI-ENGINE ENSEMBLE) ===")
    print("="*75)
    t0 = time.time()
    
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    sub_sample = pd.read_csv('data/sample_submission.csv')
    
    train_df['clean_likes'] = train_df['likes'].apply(parse_like)
    train_clean = train_df[train_df['clean_likes'].notna()].reset_index(drop=True)
    y_raw = train_clean['clean_likes'].values
    y_log = np.log1p(y_raw)
    train_clean['target_log'] = y_log
    is_zero = (y_raw == 0).astype(int)
    
    print(f"Cleaned Train shape: {train_clean.shape}, Test shape: {test_df.shape}")
    
    # Combined Preprocessing
    df_combined = pd.concat([train_clean.assign(is_test=0), test_df.assign(is_test=1)], ignore_index=True)
    df_combined['username'] = df_combined['username'].fillna('missing').astype(str).str.strip().str.lower()
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
    
    text_stats = extract_text_stats_v6(df_combined['text'])
    for col in text_stats.columns:
        df_combined[col] = text_stats[col]
        
    user_counts = df_combined['username'].value_counts()
    df_combined['user_total_tweets'] = df_combined['username'].map(user_counts).astype(float)
    df_combined['user_avg_char_len'] = df_combined.groupby('username')['char_len'].transform('mean')
    df_combined['user_cs_ratio'] = df_combined.groupby('username')['is_cs_reply'].transform('mean')
    df_combined['user_promo_ratio'] = df_combined.groupby('username')['is_promo'].transform('mean')
    
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
    df_combined['user_text_freq'] = df_combined.groupby(['username', 'text'])['id'].transform('count')
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
    
    # 5-Fold Stratified Split
    print("\n=== STEP 2: IN-FOLD HYBRID (WORD + CHAR) TF-IDF NLP META-FEATURE ===")
    bins = pd.qcut(y_log, q=10, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=PRIMARY_SEED)
    
    train_texts = train_feat['text'].fillna('').tolist()
    test_texts = test_feat['text'].fillna('').tolist()
    
    oof_ridge_text = np.zeros(len(train_feat))
    test_ridge_text = np.zeros(len(test_feat))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_feat, bins)):
        word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)
        char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=15000, sublinear_tf=True)
        union = FeatureUnion([('word', word_vec), ('char', char_vec)])
        
        X_tr_tfidf = union.fit_transform([train_texts[i] for i in tr_idx])
        X_val_tfidf = union.transform([train_texts[i] for i in val_idx])
        X_te_tfidf = union.transform(test_texts)
        
        ridge = Ridge(alpha=3.0, random_state=PRIMARY_SEED)
        ridge.fit(X_tr_tfidf, y_log[tr_idx])
        oof_ridge_text[val_idx] = ridge.predict(X_val_tfidf)
        test_ridge_text += ridge.predict(X_te_tfidf) / 5.0
        
    train_feat['ridge_text_pred'] = oof_ridge_text
    test_feat['ridge_text_pred'] = test_ridge_text
    print(f"Hybrid Ridge Text correlation with target: {np.corrcoef(oof_ridge_text, y_log)[0,1]:.4f}")
    
    print("\n=== STEP 3: PRECOMPUTING FULL-TRAIN TEMPORAL NEIGHBORS FOR TEST SET ===")
    test_nn_full = compute_temporal_neighbors_v6(test_feat, train_feat, is_train=False)
    
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
        'user_tweets_same_day', 'user_text_freq', 'is_cs_reply', 'has_phone',
        'is_promo', 'user_cs_ratio', 'user_promo_ratio'
    ]
    
    models = ['lgb', 'xgb', 'cat', 'mlp', 'hurdle']
    oof_z = {m: np.zeros(len(train_feat)) for m in models}
    test_z = {m: np.zeros(len(test_feat)) for m in models}
    
    print(f"\n=== STEP 4: 5-FOLD MULTI-ENGINE ENSEMBLE TRAINING (LGB, XGB, CAT, MLP, HURDLE) ===")
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_feat, bins)):
        f_start = time.time()
        print(f"\n--- FOLD {fold+1} / 5 ---")
        tr_df, val_df = train_feat.iloc[tr_idx], train_feat.iloc[val_idx]
        
        tr_nn = compute_temporal_neighbors_v6(tr_df, tr_df, is_train=True)
        val_nn = compute_temporal_neighbors_v6(val_df, tr_df, is_train=False)
        nn_cols = list(tr_nn.columns)
        
        tr_te, val_te, test_te_fold = compute_leak_free_target_encoding_v6(tr_df, val_df, test_feat, target_col='target_log', cat_col='username', m=15.0)
        te_cols = list(tr_te.columns)
        
        # Leak-Free Hurdle Classifier
        clf_features = [
            'user_total_tweets', 'char_len', 'word_count', 'hour', 'dayofweek',
            'clean_device_code', 'clean_media_code', 'is_reply', 'is_retweet',
            'is_cs_reply', 'has_phone', 'is_promo', 'user_cs_ratio', 'user_promo_ratio', 'user_text_freq'
        ]
        y_zero_tr = (y_raw[tr_idx] == 0).astype(int)
        
        tr_pzero = np.zeros(len(tr_df))
        ikf = KFold(n_splits=5, shuffle=True, random_state=PRIMARY_SEED+fold)
        for in_tr_p, in_val_p in ikf.split(tr_df):
            clf_in = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.06, num_leaves=20, random_state=PRIMARY_SEED+fold, verbose=-1, n_jobs=-1)
            clf_in.fit(tr_df.iloc[in_tr_p][clf_features], y_zero_tr[in_tr_p])
            tr_pzero[in_val_p] = clf_in.predict_proba(tr_df.iloc[in_val_p][clf_features])[:, 1]
            
        clf_full = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.06, num_leaves=20, random_state=PRIMARY_SEED+fold, verbose=-1, n_jobs=-1)
        clf_full.fit(tr_df[clf_features], y_zero_tr)
        val_pzero = clf_full.predict_proba(val_df[clf_features])[:, 1]
        test_pzero = clf_full.predict_proba(test_feat[clf_features])[:, 1]
        
        feat_cols = base_numeric_cols + ['ridge_text_pred'] + nn_cols + te_cols + code_cat_cols + ['p_zero']
        
        X_tr = pd.concat([
            tr_df[base_numeric_cols + ['ridge_text_pred']].reset_index(drop=True),
            tr_nn.reset_index(drop=True),
            tr_te.reset_index(drop=True),
            tr_df[code_cat_cols].reset_index(drop=True),
            pd.Series(tr_pzero, name='p_zero')
        ], axis=1)[feat_cols].fillna(0)
        
        X_val = pd.concat([
            val_df[base_numeric_cols + ['ridge_text_pred']].reset_index(drop=True),
            val_nn.reset_index(drop=True),
            val_te.reset_index(drop=True),
            val_df[code_cat_cols].reset_index(drop=True),
            pd.Series(val_pzero, name='p_zero')
        ], axis=1)[feat_cols].fillna(0)
        
        X_te = pd.concat([
            test_feat[base_numeric_cols + ['ridge_text_pred']].reset_index(drop=True),
            test_nn_full.reset_index(drop=True),
            test_te_fold.reset_index(drop=True),
            test_feat[code_cat_cols].reset_index(drop=True),
            pd.Series(test_pzero, name='p_zero')
        ], axis=1)[feat_cols].fillna(0)
        
        # 1. Multi-Seed LightGBM
        fold_lgb_val = np.zeros(len(val_df))
        fold_lgb_te = np.zeros(len(test_feat))
        for s in SEEDS:
            lgb_m = lgb.LGBMRegressor(
                objective='regression', metric='rmse', learning_rate=0.028,
                num_leaves=35, max_depth=7, min_child_samples=20,
                subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                reg_alpha=0.2, reg_lambda=1.8, n_estimators=1800,
                random_state=s + fold*10, n_jobs=-1, verbose=-1
            )
            lgb_m.fit(X_tr, y_log[tr_idx], eval_set=[(X_val, y_log[val_idx])], callbacks=[lgb.early_stopping(40, verbose=False)])
            fold_lgb_val += lgb_m.predict(X_val) / len(SEEDS)
            fold_lgb_te += lgb_m.predict(X_te) / (len(SEEDS) * 5.0)
            
        oof_z['lgb'][val_idx] = fold_lgb_val
        test_z['lgb'] += fold_lgb_te
        
        # 2. Multi-Seed XGBoost
        fold_xgb_val = np.zeros(len(val_df))
        fold_xgb_te = np.zeros(len(test_feat))
        for s in SEEDS:
            xgb_m = xgb.XGBRegressor(
                n_estimators=1500, learning_rate=0.028, max_depth=6,
                subsample=0.8, colsample_bytree=0.7, reg_alpha=0.3, reg_lambda=2.0,
                tree_method='hist', random_state=s + fold*10, n_jobs=-1, early_stopping_rounds=40
            )
            xgb_m.fit(X_tr, y_log[tr_idx], eval_set=[(X_val, y_log[val_idx])], verbose=False)
            fold_xgb_val += xgb_m.predict(X_val) / len(SEEDS)
            fold_xgb_te += xgb_m.predict(X_te) / (len(SEEDS) * 5.0)
            
        oof_z['xgb'][val_idx] = fold_xgb_val
        test_z['xgb'] += fold_xgb_te
        
        # 3. Tuned CatBoost Regressor
        cat_m = CatBoostRegressor(
            iterations=1200, learning_rate=0.035, depth=6, l2_leaf_reg=3.0,
            random_seed=PRIMARY_SEED + fold*10, verbose=0, thread_count=-1
        )
        cat_m.fit(X_tr, y_log[tr_idx], eval_set=(X_val, y_log[val_idx]), early_stopping_rounds=40, verbose=False)
        fold_cat_val = cat_m.predict(X_val)
        fold_cat_te = cat_m.predict(X_te) / 5.0
        oof_z['cat'][val_idx] = fold_cat_val
        test_z['cat'] += fold_cat_te
        
        # 4. Hurdle Positive Regressor (Trained only on y > 0)
        pos_tr_mask = (y_zero_tr == 0)
        lgb_pos = lgb.LGBMRegressor(
            objective='regression', metric='rmse', learning_rate=0.03,
            num_leaves=31, max_depth=6, subsample=0.8, colsample_bytree=0.7,
            n_estimators=1000, random_state=PRIMARY_SEED + fold, n_jobs=-1, verbose=-1
        )
        lgb_pos.fit(X_tr[pos_tr_mask], y_log[tr_idx][pos_tr_mask], eval_set=[(X_val, y_log[val_idx])], callbacks=[lgb.early_stopping(40, verbose=False)])
        fold_pos_val = lgb_pos.predict(X_val)
        fold_pos_te = lgb_pos.predict(X_te)
        
        fold_hurdle_val = (1.0 - val_pzero) * fold_pos_val
        fold_hurdle_te = (1.0 - test_pzero) * fold_pos_te / 5.0
        oof_z['hurdle'][val_idx] = fold_hurdle_val
        test_z['hurdle'] += fold_hurdle_te
        
        # 5. Keras Tabular MLP
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)
        X_te_sc = scaler.transform(X_te)
        
        mlp_m = build_keras_mlp(X_tr_sc.shape[1])
        es = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        mlp_m.fit(X_tr_sc, y_log[tr_idx], validation_data=(X_val_sc, y_log[val_idx]), epochs=20, batch_size=256, verbose=0, callbacks=[es])
        fold_mlp_val = mlp_m.predict(X_val_sc, batch_size=512, verbose=0).flatten()
        fold_mlp_te = mlp_m.predict(X_te_sc, batch_size=512, verbose=0).flatten() / 5.0
        oof_z['mlp'][val_idx] = fold_mlp_val
        test_z['mlp'] += fold_mlp_te
        
        print(f"  [LGB-MultiSeed] Fold {fold+1} RMSLE: {root_mean_squared_error(y_log[val_idx], fold_lgb_val):.5f}")
        print(f"  [XGB-MultiSeed] Fold {fold+1} RMSLE: {root_mean_squared_error(y_log[val_idx], fold_xgb_val):.5f}")
        print(f"  [CatBoost]      Fold {fold+1} RMSLE: {root_mean_squared_error(y_log[val_idx], fold_cat_val):.5f}")
        print(f"  [Hurdle-Model]  Fold {fold+1} RMSLE: {root_mean_squared_error(y_log[val_idx], fold_hurdle_val):.5f}")
        print(f"  [Keras-MLP]     Fold {fold+1} RMSLE: {root_mean_squared_error(y_log[val_idx], fold_mlp_val):.5f}")
        print(f"  Fold {fold+1} total runtime: {time.time()-f_start:.1f}s")
        
    print("\n=== STEP 5: OVERALL OOF RMSLE EVALUATION ===")
    for m in models:
        score = root_mean_squared_error(y_log, oof_z[m])
        print(f"  {m.upper():12s} OOF RMSLE: {score:.5f}")
        
    print("\n=== STEP 6: OPTIMIZING 5-ENGINE BLEND IN LOG SPACE (RMSLE) ===")
    X_blend = np.column_stack([oof_z[m] for m in models])
    
    def loss_rmsle(w):
        return root_mean_squared_error(y_log, X_blend @ w)
        
    cons = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    res = minimize(loss_rmsle, [1.0/len(models)]*len(models), bounds=[(0, 1)]*len(models), constraints=cons, method='SLSQP')
    opt_w = res.x
    print("Optimized SLSQP Weights:")
    for m, w in zip(models, opt_w):
        print(f"  {m.upper():10s}: {w:.4f}")
        
    oof_blend_z = X_blend @ opt_w
    raw_cv_rmsle = root_mean_squared_error(y_log, oof_blend_z)
    print(f"\nRaw Multi-Engine Blend OOF RMSLE: {raw_cv_rmsle:.5f}")
    
    # Empty-Gap Calibration Grid Search
    print("\n=== STEP 7: OPTIMIZING EMPTY-GAP CALIBRATION ===")
    best_rmsle = raw_cv_rmsle
    best_cut = 1.0
    best_fac = 1.0
    
    for cut in np.linspace(1.0, 2.5, 31):
        for fac in np.linspace(0.65, 0.95, 16):
            cal_z = np.where(oof_blend_z < cut, oof_blend_z * fac, oof_blend_z)
            score = root_mean_squared_error(y_log, cal_z)
            if score < best_rmsle:
                best_rmsle = score
                best_cut = cut
                best_fac = fac
                
    print(f"Optimal Gap Calibration: cut={best_cut:.2f}, factor={best_fac:.2f}")
    print(f"\n===========================================================")
    print(f"*** FINAL V6 ULTRA-CHAMPION CV RMSLE: {best_rmsle:.5f} ***")
    print(f"===========================================================")
    
    # Save OOF
    final_oof_z = np.where(oof_blend_z < best_cut, oof_blend_z * best_fac, oof_blend_z)
    oof_v6 = pd.DataFrame({
        'id': train_clean['id'],
        'actual_likes': y_raw,
        'actual_log': y_log,
        'lgb_z': oof_z['lgb'],
        'xgb_z': oof_z['xgb'],
        'cat_z': oof_z['cat'],
        'mlp_z': oof_z['mlp'],
        'hurdle_z': oof_z['hurdle'],
        'ensemble_z': oof_blend_z,
        'calibrated_z': final_oof_z,
        'final_likes': np.clip(np.expm1(final_oof_z), 0, None)
    })
    oof_v6.to_csv('outputs/oof_v6.csv', index=False)
    
    # Final Test Inference
    X_test_blend = np.column_stack([test_z[m] for m in models])
    test_blend_z = X_test_blend @ opt_w
    final_test_z = np.where(test_blend_z < best_cut, test_blend_z * best_fac, test_blend_z)
    final_test_likes = np.clip(np.expm1(final_test_z), 0, None)
    
    sub_v6 = pd.DataFrame({'id': test_df['id'], 'likes': final_test_likes})
    sub_v6.to_csv('outputs/submission_v6.csv', index=False)
    sub_v6.to_csv('outputs/submission.csv', index=False)
    
    assert len(sub_v6) == 5969, f"Mismatch rows: {len(sub_v6)}"
    assert (sub_v6['id'].values == sub_sample['id'].values).all(), "Mismatch ID!"
    assert sub_v6['likes'].isna().sum() == 0, "Found NaNs!"
    assert (sub_v6['likes'] < 0).sum() == 0, "Found negative values!"
    
    print("\nVerified outputs/submission_v6.csv and outputs/submission.csv successfully!")
    print(f"Total pipeline runtime: {time.time()-t0:.1f}s")
    print("\nTop 10 predictions preview:")
    print(sub_v6.head(10))

if __name__ == '__main__':
    run_v6()
