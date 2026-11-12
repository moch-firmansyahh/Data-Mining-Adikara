import pandas as pd
import numpy as np
import re
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

def parse_dates(s):
    is_num = s.str.match(r'^\d{9,11}$')
    res = pd.Series(index=s.index, dtype='datetime64[ns, UTC]')
    if is_num.any():
        res[is_num] = pd.to_datetime(s[is_num].astype('int64'), unit='s', utc=True)
    if (~is_num).any():
        res[~is_num] = pd.to_datetime(s[~is_num], format='mixed', utc=True)
    return res

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

def analyze():
    print("=== LOADING DATA ===")
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    sub = pd.read_csv('data/sample_submission.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Sample sub shape: {sub.shape}")

    print("\n=== ANALYZING LIKES COLUMN ===")
    likes_raw = train['likes'].astype(str)
    non_pure_digits = likes_raw[~likes_raw.str.match(r'^\d+$')]
    print(f"Total non-pure-digits rows: {len(non_pure_digits)} ({len(non_pure_digits)/len(train)*100:.2f}%)")
    print(non_pure_digits.value_counts().head(15))
    
    negatives = likes_raw[likes_raw.str.match(r'^-\d+')]
    print(f"Negative values count: {len(negatives)}: {negatives.value_counts().to_dict()}")

    train['clean_likes'] = train['likes'].apply(parse_like)
    print(f"Rows with NaN/negative likes: {train['clean_likes'].isna().sum()}")
    
    valid_likes = train['clean_likes'].dropna()
    print("Likes stats:")
    print(valid_likes.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]))
    print(f"Zero likes: {(valid_likes == 0).sum()} ({(valid_likes == 0).mean()*100:.2f}%)")
    print(f"Skewness raw: {valid_likes.skew():.2f}")
    print(f"Skewness log1p: {np.log1p(valid_likes).skew():.2f}")

    print("\n=== ANALYZING POST_DATE & TIME DISTRIBUTION ===")
    train['datetime'] = parse_dates(train['post_date'])
    test['datetime'] = parse_dates(test['post_date'])
    
    # Check 1970 epoch nulls
    train_1970 = (train['datetime'].dt.year < 2010).sum()
    test_1970 = (test['datetime'].dt.year < 2010).sum()
    print(f"Train 1970 dates: {train_1970}")
    print(f"Test 1970 dates: {test_1970}")
    
    train_valid_dt = train[train['datetime'].dt.year >= 2010]['datetime']
    test_valid_dt = test[test['datetime'].dt.year >= 2010]['datetime']
    print(f"Train date range: {train_valid_dt.min()} to {train_valid_dt.max()}")
    print(f"Test date range: {test_valid_dt.min()} to {test_valid_dt.max()}")
    
    # Year distribution comparison
    train_years = train_valid_dt.dt.year.value_counts(normalize=True).sort_index()
    test_years = test_valid_dt.dt.year.value_counts(normalize=True).sort_index()
    comp_years = pd.DataFrame({'train_%': train_years*100, 'test_%': test_years*100})
    print("\nYear distribution (%):")
    print(comp_years.round(2))

    print("\n=== ANALYZING USERNAME ===")
    train_users = set(train['username'].unique())
    test_users = set(test['username'].unique())
    overlap = train_users.intersection(test_users)
    print(f"Train unique users: {len(train_users)}")
    print(f"Test unique users: {len(test_users)}")
    print(f"Overlap users: {len(overlap)}")
    print(f"% of test users in train: {len(overlap)/len(test_users)*100:.2f}%")
    test_rows_in_train = test['username'].isin(train_users).sum()
    print(f"% of test ROWS with user in train: {test_rows_in_train / len(test) * 100:.2f}%")
    
    print("\n=== CATEGORICALS & MISSING VALUES ===")
    print("Train missing:")
    print(train.isna().sum())
    print("\nTest missing:")
    print(test.isna().sum())

    print("\nTrain media_type value_counts:")
    print(train['media_type'].value_counts(dropna=False, normalize=True)*100)
    print("\nTest media_type value_counts:")
    print(test['media_type'].value_counts(dropna=False, normalize=True)*100)
    
    print("\nTrain client_device value_counts:")
    print(train['client_device'].value_counts(dropna=False, normalize=True)*100)
    print("\nTest client_device value_counts:")
    print(test['client_device'].value_counts(dropna=False, normalize=True)*100)

    print("\n=== TEXT CHARACTERISTICS ===")
    print(f"Train text missing: {train['text'].isna().sum()}")
    print(f"Test text missing: {test['text'].isna().sum()}")
    print(f"Train duplicate text: {train['text'].duplicated().sum()}")
    print(f"Test duplicate text: {test['text'].duplicated().sum()}")

    print("\n=== ID CORRELATIONS ===")
    train['timestamp_s'] = train_valid_dt.astype('int64') // 10**9
    test['timestamp_s'] = test_valid_dt.astype('int64') // 10**9
    print(f"Corr(Train ID, Likes): {train[['id', 'clean_likes']].dropna().corr().iloc[0,1]:.4f}")
    print(f"Corr(Train ID, Timestamp): {train[['id', 'timestamp_s']].dropna().corr().iloc[0,1]:.4f}")
    print(f"Corr(Test ID, Timestamp): {test[['id', 'timestamp_s']].dropna().corr().iloc[0,1]:.4f}")
    print(f"Train ID min-max: {train['id'].min()} - {train['id'].max()}")
    print(f"Test ID min-max: {test['id'].min()} - {test['id'].max()}")

    print("\n=== ADVERSARIAL VALIDATION ===")
    df_all = pd.concat([
        train[['username', 'media_type', 'client_device']].assign(is_test=0),
        test[['username', 'media_type', 'client_device']].assign(is_test=1)
    ], ignore_index=True)
    
    df_all['datetime'] = pd.concat([train['datetime'], test['datetime']], ignore_index=True)
    df_all['year'] = df_all['datetime'].dt.year
    df_all['month'] = df_all['datetime'].dt.month
    df_all['day'] = df_all['datetime'].dt.day
    df_all['hour'] = df_all['datetime'].dt.hour
    df_all['dayofweek'] = df_all['datetime'].dt.dayofweek
    
    for col in ['username', 'media_type', 'client_device']:
        df_all[col] = df_all[col].astype('category').cat.codes
        
    features = ['username', 'year', 'month', 'day', 'hour', 'dayofweek', 'media_type', 'client_device']
    X = df_all[features]
    y = df_all['is_test']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    for tr_idx, val_idx in kf.split(X, y):
        clf = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        preds = clf.predict_proba(X.iloc[val_idx])[:, 1]
        auc_scores.append(roc_auc_score(y.iloc[val_idx], preds))
        
    print(f"Adversarial Validation AUC (with time & metadata): {np.mean(auc_scores):.4f} +/- {np.std(auc_scores):.4f}")

    # Adversarial validation WITHOUT time (only username + metadata)
    features_notime = ['username', 'media_type', 'client_device']
    X_notime = df_all[features_notime]
    auc_notime = []
    for tr_idx, val_idx in kf.split(X_notime, y):
        clf = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
        clf.fit(X_notime.iloc[tr_idx], y.iloc[tr_idx])
        preds = clf.predict_proba(X_notime.iloc[val_idx])[:, 1]
        auc_notime.append(roc_auc_score(y.iloc[val_idx], preds))
    print(f"Adversarial Validation AUC (without time): {np.mean(auc_notime):.4f} +/- {np.std(auc_notime):.4f}")

if __name__ == '__main__':
    analyze()
