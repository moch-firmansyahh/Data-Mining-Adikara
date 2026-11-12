import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

from features import (
    parse_like, build_unsupervised_features, compute_oof_target_encoding
)

SEED = 42
os.makedirs('outputs', exist_ok=True)

def run_baseline():
    print("--- 1. LOADING & PREPROCESSING DATA ---")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    sub_sample = pd.read_csv('data/sample_submission.csv')
    
    # Target parsing
    train['clean_likes'] = train['likes'].apply(parse_like)
    # Drop 25 sentinel rows (-1, -99)
    valid_mask = train['clean_likes'].notna()
    print(f"Dropping {(~valid_mask).sum()} invalid sentinel rows from train.")
    train = train[valid_mask].reset_index(drop=True)
    
    y_raw = train['clean_likes'].values
    y_log = np.log1p(y_raw)
    train['target_log'] = y_log
    
    # Build combined dataframe for unsupervised features
    print("--- 2. BUILDING UNSUPERVISED & TEMPORAL & TEXT STATS FEATURES ---")
    df_all = pd.concat([train.assign(is_test=0), test.assign(is_test=1)], ignore_index=True)
    df_feat = build_unsupervised_features(df_all)
    
    train_feat = df_feat[df_feat['is_test'] == 0].reset_index(drop=True)
    test_feat = df_feat[df_feat['is_test'] == 1].reset_index(drop=True)
    
    # Categoricals to categorical dtype for LightGBM
    cat_cols = ['clean_media', 'clean_device', 'media_device', 'media_hour_bucket']
    for c in cat_cols:
        train_feat[c] = train_feat[c].astype('category')
        test_feat[c] = test_feat[c].astype('category')
        
    base_numeric_cols = [
        'is_bad_date', 'year', 'month', 'day', 'dayofweek', 'hour', 'minute',
        'is_weekend', 'quarter', 'sin_hour', 'cos_hour', 'sin_dow', 'cos_dow',
        'time_elapsed_days', 'char_len', 'word_count', 'avg_word_len',
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
    
    # 3. Stratified 5-Fold setup
    print("--- 3. CREATING 5-FOLD STRATIFIED SPLITS ---")
    # Bin log1p target into 10 quantiles for stratification
    bins = pd.qcut(y_log, q=10, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    oof_preds_log = np.zeros(len(train_feat))
    test_preds_log = np.zeros(len(test_feat))
    
    oof_ridge_text = np.zeros(len(train_feat))
    test_ridge_text = np.zeros(len(test_feat))
    
    # Step A: Ridge on TF-IDF OOF text feature
    print("--- 4. COMPUTING TF-IDF + RIDGE TEXT META-FEATURE ---")
    train_texts = train_feat['text'].fillna('').tolist()
    test_texts = test_feat['text'].fillna('').tolist()
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_feat, bins)):
        tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
        X_tr_tfidf = tfidf.fit_transform([train_texts[i] for i in tr_idx])
        X_val_tfidf = tfidf.transform([train_texts[i] for i in val_idx])
        X_te_tfidf = tfidf.transform(test_texts)
        
        ridge = Ridge(alpha=2.0, random_state=SEED)
        ridge.fit(X_tr_tfidf, y_log[tr_idx])
        
        oof_ridge_text[val_idx] = ridge.predict(X_val_tfidf)
        test_ridge_text += ridge.predict(X_te_tfidf) / 5.0
        
    train_feat['ridge_text_pred'] = oof_ridge_text
    test_feat['ridge_text_pred'] = test_ridge_text
    
    # Step B: LightGBM Training with OOF Target Encoding
    print("--- 5. TRAINING LIGHTGBM BASELINE WITH OOF USER TARGET ENCODING ---")
    feature_names = None
    fold_rmse_raw = []
    fold_rmsle = []
    
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'n_estimators': 1500,
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': -1
    }
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_feat, bins)):
        tr_df, val_df = train_feat.iloc[tr_idx], train_feat.iloc[val_idx]
        
        # Out-of-fold target encoding
        tr_te, val_te, te_test_fold = compute_oof_target_encoding(
            tr_df, val_df, test_feat, target_col='target_log', cat_col='username', m=10.0
        )
        
        te_cols = list(tr_te.columns)
        all_features = base_numeric_cols + ['ridge_text_pred'] + te_cols + cat_cols
        if feature_names is None:
            feature_names = all_features
            
        X_tr = pd.concat([tr_df[base_numeric_cols + ['ridge_text_pred'] + cat_cols].reset_index(drop=True), tr_te.reset_index(drop=True)], axis=1)[all_features]
        X_val = pd.concat([val_df[base_numeric_cols + ['ridge_text_pred'] + cat_cols].reset_index(drop=True), val_te.reset_index(drop=True)], axis=1)[all_features]
        X_te = pd.concat([test_feat[base_numeric_cols + ['ridge_text_pred'] + cat_cols].reset_index(drop=True), te_test_fold.reset_index(drop=True)], axis=1)[all_features]
        
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_tr, y_log[tr_idx],
            eval_set=[(X_val, y_log[val_idx])],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        val_pred = model.predict(X_val)
        oof_preds_log[val_idx] = val_pred
        test_preds_log += model.predict(X_te) / 5.0
        
        val_pred_raw = np.clip(np.expm1(val_pred), 0, None)
        f_rmse = root_mean_squared_error(y_raw[val_idx], val_pred_raw)
        f_rmsle = root_mean_squared_error(y_log[val_idx], np.log1p(val_pred_raw))
        fold_rmse_raw.append(f_rmse)
        fold_rmsle.append(f_rmsle)
        print(f"Fold {fold+1} (best iter {model.best_iteration_}): Raw RMSE = {f_rmse:.4f}, RMSLE = {f_rmsle:.4f}")
        
    oof_raw = np.clip(np.expm1(oof_preds_log), 0, None)
    total_rmse = root_mean_squared_error(y_raw, oof_raw)
    total_rmsle = root_mean_squared_error(y_log, np.log1p(oof_raw))
    print("\n================ BASELINE SUMMARY ================")
    print(f"Overall OOF Raw RMSE : {total_rmse:.4f}")
    print(f"Overall OOF RMSLE    : {total_rmsle:.4f}")
    
    # Optimal calibration factor for raw RMSE
    # c = sum(y * y_pred) / sum(y_pred^2)
    c_factor = np.sum(y_raw * oof_raw) / (np.sum(oof_raw ** 2) + 1e-8)
    oof_calibrated = oof_raw * c_factor
    calibrated_rmse = root_mean_squared_error(y_raw, oof_calibrated)
    print(f"Optimal Calibration Multiplier: {c_factor:.4f}")
    print(f"Calibrated OOF Raw RMSE: {calibrated_rmse:.4f} (improvement: {total_rmse - calibrated_rmse:.4f})")
    
    # Save OOF
    oof_df = pd.DataFrame({
        'id': train['id'],
        'actual_likes': y_raw,
        'oof_likes_pred': oof_raw,
        'oof_likes_calibrated': oof_calibrated
    })
    oof_df.to_csv('outputs/oof_baseline.csv', index=False)
    
    # Save Submissions
    test_raw = np.clip(np.expm1(test_preds_log), 0, None)
    sub_baseline = pd.DataFrame({'id': test['id'], 'likes': test_raw})
    sub_baseline.to_csv('outputs/submission_baseline.csv', index=False)
    
    sub_calibrated = pd.DataFrame({'id': test['id'], 'likes': test_raw * c_factor})
    sub_calibrated.to_csv('outputs/submission_baseline_calibrated.csv', index=False)
    
    # Verification checks according to guide
    assert len(sub_baseline) == 5969, "Submission row count mismatch!"
    assert (sub_baseline['id'].values == sub_sample['id'].values).all(), "Submission ID mismatch!"
    assert sub_baseline['likes'].isna().sum() == 0, "NaN found in predictions!"
    assert (sub_baseline['likes'] < 0).sum() == 0, "Negative values found!"
    print("Submission format verified successfully! (5969 rows, exact IDs, no NaN/negatives)")
    
    # Feature importances
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    print("\nTop 15 Feature Importances:")
    print(feat_imp.head(15))

if __name__ == '__main__':
    run_baseline()

