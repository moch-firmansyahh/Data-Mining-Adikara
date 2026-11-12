import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_error, root_mean_squared_log_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from features import (
    parse_like, build_unsupervised_features, compute_oof_target_encoding
)

SEED = 42
os.makedirs('outputs', exist_ok=True)

def train_all_models():
    print("=== 1. PREPARING DATA ===")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    sub_sample = pd.read_csv('data/sample_submission.csv')
    
    train['clean_likes'] = train['likes'].apply(parse_like)
    valid_mask = train['clean_likes'].notna()
    train = train[valid_mask].reset_index(drop=True)
    
    y_raw = train['clean_likes'].values
    y_log = np.log1p(y_raw)
    train['target_log'] = y_log
    
    print("=== 2. UNSUPERVISED & TEMPORAL & STATS FEATURES ===")
    df_all = pd.concat([train.assign(is_test=0), test.assign(is_test=1)], ignore_index=True)
    df_feat = build_unsupervised_features(df_all)
    
    train_feat = df_feat[df_feat['is_test'] == 0].reset_index(drop=True)
    test_feat = df_feat[df_feat['is_test'] == 1].reset_index(drop=True)
    
    cat_cols = ['clean_media', 'clean_device', 'media_device', 'media_hour_bucket']
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
    
    bins = pd.qcut(y_log, q=10, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    
    print("=== 3. COMPUTING OOF TF-IDF RIDGE & SVD ===")
    train_texts = train_feat['text'].fillna('').tolist()
    test_texts = test_feat['text'].fillna('').tolist()
    
    n_svd = 16
    svd_col_names = [f'svd_{i}' for i in range(n_svd)]
    oof_ridge_text = np.zeros(len(train_feat))
    test_ridge_text = np.zeros(len(test_feat))
    oof_svd = np.zeros((len(train_feat), n_svd))
    test_svd = np.zeros((len(test_feat), n_svd))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_feat, bins)):
        tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)
        X_tr_tfidf = tfidf.fit_transform([train_texts[i] for i in tr_idx])
        X_val_tfidf = tfidf.transform([train_texts[i] for i in val_idx])
        X_te_tfidf = tfidf.transform(test_texts)
        
        ridge = Ridge(alpha=2.0, random_state=SEED)
        ridge.fit(X_tr_tfidf, y_log[tr_idx])
        oof_ridge_text[val_idx] = ridge.predict(X_val_tfidf)
        test_ridge_text += ridge.predict(X_te_tfidf) / 5.0
        
        svd = TruncatedSVD(n_components=n_svd, random_state=SEED)
        oof_svd[val_idx] = svd.fit_transform(X_val_tfidf)
        test_svd += svd.transform(X_te_tfidf) / 5.0
        
    train_feat['ridge_text_pred'] = oof_ridge_text
    test_feat['ridge_text_pred'] = test_ridge_text
    for i in range(n_svd):
        train_feat[f'svd_{i}'] = oof_svd[:, i]
        test_feat[f'svd_{i}'] = test_svd[:, i]
        
    # Categorical label encoding for XGBoost and CatBoost
    cat_enc_map = {}
    for c in cat_cols:
        all_cats = pd.concat([train_feat[c], test_feat[c]]).unique()
        mapping = {val: idx for idx, val in enumerate(all_cats)}
        cat_enc_map[c] = mapping
        train_feat[f'{c}_code'] = train_feat[c].map(mapping).astype(int)
        test_feat[f'{c}_code'] = test_feat[c].map(mapping).astype(int)
        
    code_cat_cols = [f'{c}_code' for c in cat_cols]
    
    # Store OOF and test predictions for ensemble
    models = ['lgb', 'cat', 'xgb']
    oof_preds = {m: np.zeros(len(train_feat)) for m in models}
    test_preds = {m: np.zeros(len(test_feat)) for m in models}
    
    print("\n=== 4. TRAINING 3 DISTINCT GBDT ARCHITECTURES (5 FOLDS) ===")
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(train_feat, bins)):
        print(f"\n>>> FOLD {fold+1} / 5 <<<")
        tr_df, val_df = train_feat.iloc[tr_idx], train_feat.iloc[val_idx]
        
        tr_te, val_te, te_test_fold = compute_oof_target_encoding(
            tr_df, val_df, test_feat, target_col='target_log', cat_col='username', m=10.0
        )
        te_cols = list(tr_te.columns)
        
        # Features for LGBM (uses category dtype)
        lgb_features = base_numeric_cols + ['ridge_text_pred'] + svd_col_names + te_cols + cat_cols
        X_tr_lgb = pd.concat([tr_df[base_numeric_cols + ['ridge_text_pred'] + svd_col_names].reset_index(drop=True),
                              tr_te.reset_index(drop=True),
                              tr_df[cat_cols].astype('category').reset_index(drop=True)], axis=1)[lgb_features]
        X_val_lgb = pd.concat([val_df[base_numeric_cols + ['ridge_text_pred'] + svd_col_names].reset_index(drop=True),
                               val_te.reset_index(drop=True),
                               val_df[cat_cols].astype('category').reset_index(drop=True)], axis=1)[lgb_features]
        X_te_lgb = pd.concat([test_feat[base_numeric_cols + ['ridge_text_pred'] + svd_col_names].reset_index(drop=True),
                              te_test_fold.reset_index(drop=True),
                              test_feat[cat_cols].astype('category').reset_index(drop=True)], axis=1)[lgb_features]
        
        # Features for CatBoost & XGBoost (uses integer codes)
        num_features = base_numeric_cols + ['ridge_text_pred'] + svd_col_names + te_cols + code_cat_cols
        X_tr_num = pd.concat([tr_df[base_numeric_cols + ['ridge_text_pred'] + svd_col_names].reset_index(drop=True),
                              tr_te.reset_index(drop=True),
                              tr_df[code_cat_cols].reset_index(drop=True)], axis=1)[num_features]
        X_val_num = pd.concat([val_df[base_numeric_cols + ['ridge_text_pred'] + svd_col_names].reset_index(drop=True),
                               val_te.reset_index(drop=True),
                               val_df[code_cat_cols].reset_index(drop=True)], axis=1)[num_features]
        X_te_num = pd.concat([test_feat[base_numeric_cols + ['ridge_text_pred'] + svd_col_names].reset_index(drop=True),
                              te_test_fold.reset_index(drop=True),
                              test_feat[code_cat_cols].reset_index(drop=True)], axis=1)[num_features]

        # --- MODEL 1: LightGBM ---
        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            metric='rmse',
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=25,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=1.5,
            n_estimators=1500,
            random_state=SEED + fold,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(
            X_tr_lgb, y_log[tr_idx],
            eval_set=[(X_val_lgb, y_log[val_idx])],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)]
        )
        pred_lgb = lgb_model.predict(X_val_lgb)
        oof_preds['lgb'][val_idx] = pred_lgb
        test_preds['lgb'] += lgb_model.predict(X_te_lgb) / 5.0
        rmse_lgb = root_mean_squared_error(y_raw[val_idx], np.clip(np.expm1(pred_lgb), 0, None))
        print(f"  [LightGBM] Fold {fold+1} Raw RMSE: {rmse_lgb:.4f}")

        # --- MODEL 2: CatBoost ---
        cb_model = CatBoostRegressor(
            iterations=1200,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=3.0,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=SEED + fold,
            verbose=0,
            thread_count=-1
        )
        cb_model.fit(
            X_tr_num, y_log[tr_idx],
            eval_set=(X_val_num, y_log[val_idx]),
            early_stopping_rounds=40
        )
        pred_cat = cb_model.predict(X_val_num)
        oof_preds['cat'][val_idx] = pred_cat
        test_preds['cat'] += cb_model.predict(X_te_num) / 5.0
        rmse_cat = root_mean_squared_error(y_raw[val_idx], np.clip(np.expm1(pred_cat), 0, None))
        print(f"  [CatBoost] Fold {fold+1} Raw RMSE: {rmse_cat:.4f}")

        # --- MODEL 3: XGBoost ---
        xgb_model = xgb.XGBRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.5,
            reg_lambda=2.0,
            tree_method='hist',
            random_state=SEED + fold,
            n_jobs=-1,
            early_stopping_rounds=40
        )
        xgb_model.fit(
            X_tr_num, y_log[tr_idx],
            eval_set=[(X_val_num, y_log[val_idx])],
            verbose=False
        )
        pred_xgb = xgb_model.predict(X_val_num)
        oof_preds['xgb'][val_idx] = pred_xgb
        test_preds['xgb'] += xgb_model.predict(X_te_num) / 5.0
        rmse_xgb = root_mean_squared_error(y_raw[val_idx], np.clip(np.expm1(pred_xgb), 0, None))
        print(f"  [XGBoost]  Fold {fold+1} Raw RMSE: {rmse_xgb:.4f}")

    print("\n=== 5. EVALUATION OF INDIVIDUAL MODELS ===")
    oof_raw = {}
    test_raw = {}
    for m in models:
        oof_raw[m] = np.clip(np.expm1(oof_preds[m]), 0, None)
        test_raw[m] = np.clip(np.expm1(test_preds[m]), 0, None)
        m_rmse = root_mean_squared_error(y_raw, oof_raw[m])
        m_rmsle = root_mean_squared_error(y_log, np.log1p(oof_raw[m]))
        print(f"{m.upper():8s} OOF Raw RMSE: {m_rmse:.4f} | RMSLE: {m_rmsle:.4f}")
        
    print("\n=== 6. BLENDING OPTIMIZATION (ENSEMBLE) ===")
    # Optimize weights w0, w1, w2 to minimize RMSE(y_raw, w0*pred0 + w1*pred1 + w2*pred2)
    def loss_func(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred_blend = (w[0] * oof_raw['lgb'] + 
                      w[1] * oof_raw['cat'] + 
                      w[2] * oof_raw['xgb'])
        return root_mean_squared_error(y_raw, pred_blend)
        
    init_weights = [1.0/3, 1.0/3, 1.0/3]
    bounds = [(0, 1), (0, 1), (0, 1)]
    res = minimize(loss_func, init_weights, bounds=bounds, method='SLSQP')
    opt_w = res.x / np.sum(res.x)
    print(f"Optimized Weights: LGB={opt_w[0]:.4f}, CAT={opt_w[1]:.4f}, XGB={opt_w[2]:.4f}")
    
    oof_blend = (opt_w[0] * oof_raw['lgb'] + 
                 opt_w[1] * oof_raw['cat'] + 
                 opt_w[2] * oof_raw['xgb'])
    blend_rmse = root_mean_squared_error(y_raw, oof_blend)
    blend_rmsle = root_mean_squared_error(y_log, np.log1p(oof_blend))
    print(f"Ensemble OOF Raw RMSE: {blend_rmse:.4f} | RMSLE: {blend_rmsle:.4f}")
    
    # Smearing / Calibration Multiplier on Ensemble
    c_factor = np.sum(y_raw * oof_blend) / (np.sum(oof_blend ** 2) + 1e-8)
    oof_blend_cal = oof_blend * c_factor
    cal_rmse = root_mean_squared_error(y_raw, oof_blend_cal)
    cal_rmsle = root_mean_squared_error(y_log, np.log1p(oof_blend_cal))
    print(f"Optimal Calibration Multiplier: {c_factor:.4f}")
    print(f"Calibrated Ensemble OOF Raw RMSE: {cal_rmse:.4f} | RMSLE: {cal_rmsle:.4f}")
    
    # Check integer rounding
    oof_rounded = np.round(oof_blend_cal)
    round_rmse = root_mean_squared_error(y_raw, oof_rounded)
    print(f"Rounded Calibrated OOF Raw RMSE : {round_rmse:.4f}")
    
    # Save OOF Predictions
    oof_all_df = pd.DataFrame({
        'id': train['id'],
        'actual_likes': y_raw,
        'lgb_pred': oof_raw['lgb'],
        'cat_pred': oof_raw['cat'],
        'xgb_pred': oof_raw['xgb'],
        'ensemble_pred': oof_blend,
        'ensemble_calibrated': oof_blend_cal
    })
    oof_all_df.to_csv('outputs/oof_ensemble.csv', index=False)
    
    # Compute Test Predictions for Ensemble
    test_blend = (opt_w[0] * test_raw['lgb'] + 
                  opt_w[1] * test_raw['cat'] + 
                  opt_w[2] * test_raw['xgb'])
    test_blend_cal = test_blend * c_factor
    
    # Save Submissions
    sub_lgb = pd.DataFrame({'id': test['id'], 'likes': test_raw['lgb']})
    sub_lgb.to_csv('outputs/submission_lgb.csv', index=False)
    
    sub_cat = pd.DataFrame({'id': test['id'], 'likes': test_raw['cat']})
    sub_cat.to_csv('outputs/submission_cat.csv', index=False)
    
    sub_xgb = pd.DataFrame({'id': test['id'], 'likes': test_raw['xgb']})
    sub_xgb.to_csv('outputs/submission_xgb.csv', index=False)
    
    sub_blend = pd.DataFrame({'id': test['id'], 'likes': test_blend})
    sub_blend.to_csv('outputs/submission_blend.csv', index=False)
    
    # Use rounded or continuous based on validation
    final_sub_values = test_blend_cal if cal_rmse <= round_rmse else np.round(test_blend_cal)
    sub_final = pd.DataFrame({'id': test['id'], 'likes': final_sub_values})
    sub_final.to_csv('outputs/submission.csv', index=False)
    
    # Verification checks
    for path, df in [('submission_blend.csv', sub_blend), ('submission.csv', sub_final)]:
        assert len(df) == 5969, f"{path}: length mismatch {len(df)} != 5969"
        assert (df['id'].values == sub_sample['id'].values).all(), f"{path}: id mismatch"
        assert df['likes'].isna().sum() == 0, f"{path}: NaN values found"
        assert (df['likes'] < 0).sum() == 0, f"{path}: negative values found"
    print("\nAll submissions saved and verified according to Phase 7 guidelines!")
    print(f"Final submission written to: outputs/submission.csv")

if __name__ == '__main__':
    train_all_models()

