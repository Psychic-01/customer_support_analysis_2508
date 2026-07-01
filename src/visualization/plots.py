"""EDAで使用する可視化関数群。"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.tseries.frequencies import to_offset
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

FIGURES_DIR = Path("outputs/figures")


def _save_or_show(save_as: Optional[str] = None) -> None:
    if save_as is not None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(FIGURES_DIR / save_as, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def fill_flag_area(ax, flags: pd.Series, label=None, freq=None, **kwargs):
    """フラグが立っている領域を塗りつぶす。

    params:
        ax: Matplotlib の Axes オブジェクト
        flags: index が DatetimeIndex で dtype が bool な pandas.Series オブジェクト
        freq: 時系列データの1単位時間, 指定しない場合は flags.index.freq が使われる
              flags.index.freq が None の場合には必ず指定しなければならない
              (例: 1日単位のデータの場合) pandas.tseries.frequencies.Day(1)
    return:
        Matplotlib の Axes オブジェクト
    """
    assert flags.dtype == bool
    assert type(flags.index) == pd.DatetimeIndex
    freq = freq or flags.index.freq
    assert freq is not None
    diff = pd.Series([0] + list(flags.astype(int)) + [0]).diff().dropna()
    for start, end in zip(flags.index[diff.iloc[:-1] == 1], flags.index[diff.iloc[1:] == -1]):
        ax.axvspan(start, end + freq, label=label, **kwargs)
        label = None  # 凡例が複数表示されないようにする
    return ax


def lineplot(df: pd.DataFrame, y: str, title: str, x: str = "cdr_date", save_as: Optional[str] = None) -> None:
    plt.figure(figsize=(12, 5))
    sns.lineplot(x=x, y=y, data=df)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title(title)
    _save_or_show(save_as)


def lineplot_with_cm(
    df: pd.DataFrame,
    col: str,
    label: str,
    freq_val: str,
    cm_df: pd.DataFrame,
    freq: str = "1W",
    date: str = "cdr_date",
    save_as: Optional[str] = None,
) -> None:
    """推移をプロットし、平均・中央値・75%値の水平線とCM放映期間を重ねて表示する。"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[date], df[col], label=label)

    mean_value = df[col].mean()
    ax.axhline(mean_value, color="red", linestyle="--", linewidth=0.5, label=f"平均値：{mean_value:.1f}")

    med_value = df[col].median()
    ax.axhline(med_value, color="green", linestyle="--", linewidth=0.5, label=f"中央値：{med_value:.1f}")

    q3_value = df[col].quantile(0.75)
    ax.axhline(q3_value, color="purple", linestyle="--", linewidth=0.5, label=f"75%値：{q3_value:.1f}")

    cm_series = cm_df.set_index("cdr_date")["cm_flg"].astype(bool)
    fill_flag_area(ax, cm_series, label="CM放映期間", alpha=0.2, freq=to_offset(freq))

    ax.set_title(f"{label}推移（{freq_val}）")
    ax.set_xlabel(freq_val)
    ax.set_ylabel(col)
    ax.legend()
    plt.tight_layout()
    _save_or_show(save_as)


def callnum_plot_with_cm(
    df: pd.DataFrame,
    col: str,
    label: str,
    freq_val: str,
    cm_df: pd.DataFrame,
    freq: str = "1W",
    date: str = "cdr_date",
    save_as: Optional[str] = None,
) -> None:
    """入電数の推移をプロットし、営業日ベースの平均線とCM放映期間を重ねて表示する。"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df[date], df[col], label=label)

    business_days = df[df["holiday_flag"] == False]
    mean_value = business_days[col].mean()
    ax.axhline(mean_value, color="red", linestyle="--", linewidth=0.5, label=f"平均値：{mean_value:.1f}")

    med_value = business_days[col].median()
    ax.axhline(med_value, color="green", linestyle="--", linewidth=0.5, label=f"中央値：{med_value:.1f}")

    q3_value = business_days[col].quantile(0.75)
    ax.axhline(q3_value, color="purple", linestyle="--", linewidth=0.5, label=f"75%値：{q3_value:.1f}")

    cm_series = cm_df.set_index("cdr_date")["cm_flg"].astype(bool)
    fill_flag_area(ax, cm_series, label="CM放映期間", alpha=0.2, freq=to_offset(freq))

    ax.set_title(f"{label}推移（{freq_val}）")
    ax.set_xlabel(freq_val)
    ax.set_ylabel(col)
    ax.legend()
    plt.tight_layout()
    _save_or_show(save_as)


def stl_decomposition(
    df: pd.DataFrame, num_col: str, period: int, date_col: str = "cdr_date", save_as: Optional[str] = None
) -> None:
    """STL分解（トレンド・季節性・残差）をプロットする。"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    stl = STL(df[num_col], period=period)
    res = stl.fit()

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    df[num_col].plot(ax=axes[0])
    axes[0].set_title("Original Data")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    res.trend.plot(ax=axes[1])
    axes[1].set_title("Trend")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    res.seasonal.plot(ax=axes[2])
    axes[2].set_title("Seasonal")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    res.resid.plot(ax=axes[3])
    axes[3].set_title("Residual")
    axes[3].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    _save_or_show(save_as)


def plot_standardized_trends(
    merged_df: pd.DataFrame, cm_df: pd.DataFrame, freq: str = "1W", save_as: Optional[str] = None
) -> None:
    """検索数・アカウント開設数・入電数を標準化して重ねてプロットし、CM放映期間を塗りつぶす。

    merged_df: search_cnt, acc_get_cnt, call_num, cdr_date を含むDataFrame（週次ベース）
    cm_df: cm_flgとcdr_dateを含むDataFrame（日次ベース）
    """
    scaler = StandardScaler()
    cols_to_scale = ["search_cnt", "acc_get_cnt", "call_num"]
    scaled = scaler.fit_transform(merged_df[cols_to_scale])
    merged_df[["search_std", "acc_std", "call_std"]] = scaled

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.lineplot(data=merged_df, x="cdr_date", y="search_std", label="検索数（標準化）", ax=ax)
    sns.lineplot(data=merged_df, x="cdr_date", y="acc_std", label="アカウント開設数（標準化）", ax=ax)
    sns.lineplot(data=merged_df, x="cdr_date", y="call_std", label="入電数（標準化）", ax=ax)

    cm_series = cm_df.set_index("cdr_date")["cm_flg"].astype(bool)
    fill_flag_area(ax, cm_series, label="CM放映期間", alpha=0.2, freq=to_offset(freq))

    ax.set_xlabel("週次")
    ax.set_ylabel("標準化値")
    ax.set_title("関連データのトレンド比較（週次・標準化）")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    _save_or_show(save_as)


def plot_by_dow(df: pd.DataFrame, num_col: str, label: str, save_as: Optional[str] = None) -> None:
    """曜日別の平均値（全日 vs 祝日除く営業日）を棒グラフで比較する。

    df: holiday_flagとdow_nameを含むDataFrame（日次ベース）
    num_col: 集計対象カラム
    label: num_colのラベル名
    """
    dow_order = ["日", "月", "火", "水", "木", "金", "土"]

    overall_mean = df.groupby("dow_name", sort=False)[num_col].mean()
    weekday_mean = (
        df[df["holiday_flag"] == False]
        .groupby("dow_name", sort=False)[num_col]
        .mean()
    )

    plot_df = pd.DataFrame({
        "全日平均": overall_mean,
        "祝日除く営業日平均": weekday_mean,
    }).reindex(dow_order)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(ax=ax, kind="bar")

    ax.set_xlabel("曜日")
    ax.set_ylabel(label)
    ax.set_title(f"曜日別{label}（全日 vs 祝日除く営業日）")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_as)
