"""データ読込・カレンダーマージ・入電数の空白期間補完。"""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# 全テーブルの分析対象開始日（それ以前はデータが揃っていないため除外）
START_DATE = pd.to_datetime("2018-06-01")

# 休業日として扱う日付（休日フラグが立っていないが実際は営業していない日）
CLOSED_DATES = pd.to_datetime([
    "2018-09-06", "2018-08-31", "2018-12-31",
    "2019-10-15", "2019-12-31", "2020-01-02",
])

RAW_FILENAMES = {
    "calender": "calender_data .csv",
    "cm": "cm_data .csv",
    "gt": "gt_service_name .csv",
    "regi_acc": "regi_acc_get_data_transform .csv",
    "regi_call": "regi_call_data_transform .csv",
}


def load_raw_tables(raw_dir: Path = RAW_DATA_DIR) -> Dict[str, pd.DataFrame]:
    """5つの生データテーブルを読み込み、日付型変換と開始日フィルタを適用する。"""
    calender_df = pd.read_csv(raw_dir / RAW_FILENAMES["calender"])
    cm_df = pd.read_csv(raw_dir / RAW_FILENAMES["cm"])
    gt_df = pd.read_csv(raw_dir / RAW_FILENAMES["gt"])
    regi_acc_df = pd.read_csv(raw_dir / RAW_FILENAMES["regi_acc"])
    regi_call_df = pd.read_csv(raw_dir / RAW_FILENAMES["regi_call"])

    calender_df["cdr_date"] = pd.to_datetime(calender_df["cdr_date"], format="%Y-%m-%d")
    cm_df["cdr_date"] = pd.to_datetime(cm_df["cdr_date"], format="%Y-%m-%d")
    gt_df["week"] = pd.to_datetime(gt_df["week"], format="%Y-%m-%d")
    regi_acc_df["cdr_date"] = pd.to_datetime(regi_acc_df["cdr_date"], format="%Y-%m-%d")
    regi_call_df["cdr_date"] = pd.to_datetime(regi_call_df["cdr_date"], format="%Y-%m-%d")

    regi_acc_df = regi_acc_df[regi_acc_df["cdr_date"] >= START_DATE]
    regi_call_df = regi_call_df[regi_call_df["cdr_date"] >= START_DATE]
    cm_df = cm_df[cm_df["cdr_date"] >= START_DATE]
    calender_df = calender_df[calender_df["cdr_date"] >= START_DATE]
    gt_df = gt_df[gt_df["week"] >= START_DATE]

    return {
        "calender": calender_df,
        "cm": cm_df,
        "gt": gt_df,
        "regi_acc": regi_acc_df,
        "regi_call": regi_call_df,
    }


def merge_calendar(df: pd.DataFrame, calender_df: pd.DataFrame) -> pd.DataFrame:
    """指定したDataFrameにカレンダー情報（休日フラグ等）を左結合する。"""
    return pd.merge(df, calender_df, on="cdr_date", how="left")


def fill_missing_call_days(regi_call_df: pd.DataFrame) -> pd.DataFrame:
    """休日ではないのに入電数が0になっている空白期間を補完する。

    単発の欠測日は closed_dates として休業日扱いにし、
    連続した欠測期間（2019-01-09〜31, 2019-11-18〜22）は
    曜日別の営業日中央値で埋める。
    """
    df = regi_call_df.copy()

    zero_on_business_day = df[(df["call_num"] == 0) & (df["holiday_flag"] == False)]
    to_fill = zero_on_business_day[~zero_on_business_day["cdr_date"].isin(CLOSED_DATES)].copy()

    weekday_medians = df[df["holiday_flag"] == False].groupby("dow_name")["call_num"].median()
    to_fill["call_num_median"] = to_fill["dow_name"].map(weekday_medians)
    to_fill["call_num_filled"] = to_fill.apply(
        lambda row: row["call_num_median"] if row["call_num"] == 0 else row["call_num"],
        axis=1,
    )

    df.loc[to_fill.index, "call_num"] = to_fill["call_num_filled"]
    df["business_day_flag"] = (df["call_num"] > 0) & (df["holiday_flag"] == False)
    return df


def make_weekly(df: pd.DataFrame, date_col: str = "cdr_date") -> pd.DataFrame:
    """日次DataFrameを週次（週末=日曜起点）に合計してリサンプリングする。"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).resample("W-SUN").sum().reset_index()


def save_processed(df: pd.DataFrame, filename: str, processed_dir: Path = PROCESSED_DATA_DIR) -> Path:
    """前処理済みDataFrameを data/processed/ にCSVで保存する。"""
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / filename
    df.to_csv(out_path, index=False)
    return out_path
