import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

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

def clean_media(s):
    s = s.fillna('missing').astype(str).str.strip().str.lower()
    s = s.replace({
        'foto': 'photo',
        'vidio': 'video',
        'carousell': 'carousel'
    })
    valid = {'text_only', 'photo', 'video', 'carousel', 'missing'}
    return s.apply(lambda x: x if x in valid else 'other')

def clean_device(s):
    s = s.fillna('missing').astype(str).str.strip()
    s = s.replace({
        'Andorid': 'Android',
        'Web-App': 'Web App'
    })
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
    
    # Simple regex for emojis
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
    df_feat['emoji_count'] = s.apply(lambda x: len(emoji_pattern.findall(x)))
    return df_feat

def build_unsupervised_features(df_all):
    """
    Features computed across train + test without using the target 'likes'.
    Leakage-free.
    """
    df = df_all.copy()
    
    # 1. Clean categoricals
    df['clean_media'] = clean_media(df['media_type'])
    df['clean_device'] = clean_device(df['client_device'])
    
    # 2. Datetime parsing
    df['dt'] = parse_dates(df['post_date'])
    # Fix 1970 bad dates
    is_bad_date = (df['dt'].dt.year < 2010).astype(int)
    valid_median = df.loc[is_bad_date == 0, 'dt'].median()
    df['dt_clean'] = df['dt']
    df.loc[is_bad_date == 1, 'dt_clean'] = valid_median
    df['is_bad_date'] = is_bad_date
    
    # Temporal features
    df['year'] = df['dt_clean'].dt.year
    df['month'] = df['dt_clean'].dt.month
    df['day'] = df['dt_clean'].dt.day
    df['dayofweek'] = df['dt_clean'].dt.dayofweek
    df['hour'] = df['dt_clean'].dt.hour
    df['minute'] = df['dt_clean'].dt.minute
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['quarter'] = df['dt_clean'].dt.quarter
    
    # Cyclic encoding
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['sin_dow'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['cos_dow'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    
    # Numeric timestamp
    min_time = df['dt_clean'].min()
    df['time_elapsed_days'] = (df['dt_clean'] - min_time).dt.total_seconds() / (24 * 3600.0)
    
    # 3. Text stats
    text_stats = extract_text_stats(df['text'])
    for col in text_stats.columns:
        df[col] = text_stats[col]
        
    # 4. User unsupervised behavior
    user_counts = df['username'].value_counts()
    df['user_total_tweets'] = df['username'].map(user_counts).astype(float)
    
    # User text length avg
    user_avg_len = df.groupby('username')['char_len'].transform('mean')
    df['user_avg_char_len'] = user_avg_len
    
    # User media ratios
    for m in ['text_only', 'photo', 'video', 'carousel']:
        df[f'user_{m}_prop'] = df.groupby('username')['clean_media'].transform(lambda x: (x == m).mean())
        
    # User device ratios
    for d in ['Android', 'iOS', 'Web App']:
        df[f'user_{d}_prop'] = df.groupby('username')['clean_device'].transform(lambda x: (x == d).mean())

    # User active span
    user_min_dt = df.groupby('username')['dt_clean'].transform('min')
    user_max_dt = df.groupby('username')['dt_clean'].transform('max')
    df['user_active_span_days'] = (user_max_dt - user_min_dt).dt.total_seconds() / (24 * 3600.0)
    df['user_tweet_age_days'] = (df['dt_clean'] - user_min_dt).dt.total_seconds() / (24 * 3600.0)
    
    # Sorted by user & time for intervals
    df = df.sort_values(['username', 'dt_clean']).reset_index(drop=True)
    df['user_time_diff_prev'] = df.groupby('username')['dt_clean'].diff().dt.total_seconds().fillna(-1) / 3600.0
    df['user_time_diff_next'] = (-df.groupby('username')['dt_clean'].diff(-1)).dt.total_seconds().fillna(-1) / 3600.0
    
    # Tweets on same day
    df['date_only'] = df['dt_clean'].dt.date
    df['user_tweets_same_day'] = df.groupby(['username', 'date_only'])['id'].transform('count')
    df.drop(columns=['date_only'], inplace=True)
    
    # Re-sort to original order by ID
    df = df.sort_values('id').reset_index(drop=True)
    
    # Categorical interactions
    df['media_device'] = df['clean_media'] + '_' + df['clean_device']
    df['media_hour_bucket'] = df['clean_media'] + '_' + (df['hour'] // 4).astype(str)
    
    return df


def compute_oof_target_encoding(train_df, val_df, test_df, target_col='target_log', cat_col='username', m=10.0):
    """
    Strictly out-of-fold target encoding with Bayesian smoothing.
    Fit ONLY on train_df, transform val_df and test_df.
    """
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
    
    # Bayesian smoothed mean
    stats['user_te_mean'] = (stats['count'] * stats['raw_mean'] + m * global_mean) / (stats['count'] + m)
    # Fill std for single-item users
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
