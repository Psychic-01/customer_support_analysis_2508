"""XGBoostモデル用の特徴量エンジニアリング関数群。"""
import numpy as np
import pandas as pd

# 振込予定日（毎月）: 対象サービスの決済入金が発生し、関連する問合せが増えやすい日
TRANSFER_DAYS = [5, 10, 15, 20, 25]

# 確定申告期間（月日のMMDD表現、会計関連の問合せが増えやすい時期）
TAX_FILING_PERIOD_MMDD = (216, 315)

# サービス障害が発生し、問合せ増加が見込まれる期間
INCIDENT_PERIODS = [
    ("2018-10-25", "2018-10-25"),
    ("2019-06-15", "2019-06-16"),
    ("2019-09-26", "2019-09-26"),
    ("2019-12-18", "2019-12-18"),
    ("2020-02-26", "2020-02-29"),
]


def add_dow_mean_features(
    df: pd.DataFrame,
    base_df: pd.DataFrame,
    call_col: str = "call_num",
    dow_col: str = "dow_name",
    holiday_col: str = "holiday_flag",
) -> pd.DataFrame:
    """曜日別の平均入電数（営業日のみ）を特徴量として付与する。

    df: 特徴量付与対象
    base_df: 平均算出に使う元データ（例: regi_call_df）
    """
    df = df.copy()
    dow_mean_weekday = base_df[base_df[holiday_col] == False].groupby(dow_col)[call_col].mean().to_dict()
    df["dow_callnum_mean_weekday"] = df[dow_col].map(dow_mean_weekday)
    return df


def add_last7d_log1p_avg(
    df: pd.DataFrame,
    target_col: str = "acc_get_cnt",
    date_col: str = "cdr_date",
    output_col: str = "acc_cnt_log1p_last7d_avg",
) -> pd.DataFrame:
    """予測日を含む過去7日間（当日〜6日前）のlog1p平均を特徴量として追加する。"""
    df = df.copy()
    df = df.sort_values(date_col).reset_index(drop=True)
    df[output_col] = np.nan
    df = df.set_index(date_col)

    for current_date in df.index:
        start = current_date - pd.Timedelta(days=6)
        window = df.loc[start:current_date, target_col]
        df.at[current_date, output_col] = np.log1p(window).mean(skipna=True)

    return df.reset_index()


def add_holiday_adjacency_features(df: pd.DataFrame, date_col: str = "cdr_date") -> pd.DataFrame:
    """前日が休日かどうか、連休明け初営業日かどうかの特徴量を追加する。"""
    df = df.sort_values(date_col).reset_index(drop=True)
    df["day_after_holiday_flag"] = df["holiday_flag"].shift(1).fillna(0).astype(int)
    df["is_first_business_day_after_holidays"] = (
        (df["holiday_flag"] == 0) & (df["dow"] < 5) & (df["holiday_flag"].shift(1) == 1)
    ).astype(int)
    return df


def add_days_since_last_holiday(df: pd.DataFrame, date_col: str = "cdr_date") -> pd.DataFrame:
    """直近の休日からの経過日数を特徴量として追加する（連休明けの需要増を捉える）。"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    last_holiday_marker = np.where(df["holiday_flag"] == 1, df[date_col], pd.NaT)
    last_holiday_date = pd.Series(last_holiday_marker, index=df.index).ffill()
    df["days_since_last_holiday"] = (df[date_col] - pd.to_datetime(last_holiday_date)).dt.days
    return df


def add_conditional_group_call_stats(
    df: pd.DataFrame,
    train_cutoff_date: str,
    group_cols=("holiday_flag", "day_before_holiday_flag", "day_after_holiday_flag", "dow"),
    min_group_size: int = 10,
    date_col: str = "cdr_date",
    target_col: str = "call_num",
) -> pd.DataFrame:
    """休日属性の組み合わせ（条件グループ）ごとのtrain期間内call_num平均・中央値を特徴量として付与する。

    件数が min_group_size 以下のグループは全体平均・中央値にフォールバックする。
    """
    df = df.copy()
    df["cond_group"] = df[list(group_cols)].astype(str).agg("_".join, axis=1)

    train_df = df[df[date_col] < train_cutoff_date]
    group_stats = train_df.groupby("cond_group")[target_col].agg(count="count", mean="mean", median="median").reset_index()

    global_mean = train_df[target_col].mean()
    global_median = train_df[target_col].median()
    group_stats["mean"] = group_stats.apply(lambda row: global_mean if row["count"] <= min_group_size else row["mean"], axis=1)
    group_stats["median"] = group_stats.apply(
        lambda row: global_median if row["count"] <= min_group_size else row["median"], axis=1
    )

    group_stats = group_stats[["cond_group", "mean", "median"]].rename(
        columns={"mean": "call_num_mean_group", "median": "call_num_median_group"}
    )
    df = df.merge(group_stats, on="cond_group", how="left").drop(columns=["cond_group"])
    return df


def add_tax_filing_period_flag(df: pd.DataFrame, date_col: str = "cdr_date") -> pd.DataFrame:
    """確定申告期間（2/16〜3/15）フラグを追加する。"""
    df = df.copy()
    md = df[date_col].dt.month * 100 + df[date_col].dt.day
    df["is_tax_filing_period"] = md.between(*TAX_FILING_PERIOD_MMDD).astype(int)
    return df


def add_transfer_day_flag(
    df: pd.DataFrame, calender_df: pd.DataFrame, date_col: str = "cdr_date"
) -> pd.DataFrame:
    """関連決済サービスの振込予定日（毎月5/10/15/20/25日、休日は翌営業日にシフト）フラグを追加する。"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    calender_df = calender_df.copy()
    calender_df[date_col] = pd.to_datetime(calender_df[date_col])
    calender_df["is_holiday_or_weekend"] = (calender_df["holiday_flag"] == 1) | (
        calender_df[date_col].dt.weekday >= 5
    )
    calender_df = calender_df.set_index(date_col)

    transfer_base = calender_df[calender_df.index.day.isin(TRANSFER_DAYS)]

    adjusted_dates = []
    for date in transfer_base.index:
        d = date
        while calender_df.loc[d, "is_holiday_or_weekend"]:
            d += pd.Timedelta(days=1)
        adjusted_dates.append(d)

    transfer_dates = pd.Series(adjusted_dates).drop_duplicates().sort_values().reset_index(drop=True)
    df["is_transfer_day"] = df[date_col].isin(transfer_dates).astype(int)
    return df


def add_incident_flag(df: pd.DataFrame, date_col: str = "cdr_date") -> pd.DataFrame:
    """既知のサービス障害期間フラグを追加する（障害による問合せ急増を捉える）。"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["incident_flag"] = 0
    for start, end in INCIDENT_PERIODS:
        mask = (df[date_col] >= pd.to_datetime(start)) & (df[date_col] <= pd.to_datetime(end))
        df.loc[mask, "incident_flag"] = 1
    return df


def add_spike_flag(df: pd.DataFrame, base_series: pd.Series, col: str, quantile: float = 0.9) -> pd.DataFrame:
    """base_seriesの指定分位点を閾値に、突発的な増加（スパイク）フラグを追加する。"""
    df = df.copy()
    threshold = base_series.quantile(quantile)
    df[f"{col}_spike_flag"] = (df[col] > threshold).astype(int)
    return df
