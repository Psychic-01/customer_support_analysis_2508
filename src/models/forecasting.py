"""入電数予測モデル（統計・機械学習）の学習・評価・チューニング関数群。"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor

RANDOM_STATE = 42
MODELS_DIR = Path("outputs/models")

GROUP_COLS = ["holiday_flag", "day_before_holiday_flag", "day_after_holiday_flag", "dow"]

DEFAULT_DROP_COLS = [
    "dow_name", "woy", "wom", "doy", "financial_year",
    "holiday_name", "exceed_count", "exceed_rate", "call_num_diff1",
]


def adf_test(ts: pd.Series) -> float:
    """ADF検定を実行し、p値を表示・返却する（p<0.05で定常と判定）。"""
    result = adfuller(ts)
    p_value = result[1]
    print("p値\t: %f" % p_value)
    if p_value < 0.05:
        print("時系列データは定常であると考えられます。")
    else:
        print("時系列データは定常であるとは言えません。")
    return p_value


def evaluate_forecast(y_true, y_pred) -> Dict[str, float]:
    """MAE, RMSE, MAPEを計算して表示・返却する。"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mae / np.mean(y_true) * 100
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def fit_sarima_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str = "call_num",
    order: Tuple[int, int, int] = (6, 1, 6),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 7),
) -> pd.Series:
    """SARIMAモデルを学習し、test期間のフォーキャストを返す。"""
    model = SARIMAX(
        train[target_col],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)
    return result.forecast(steps=len(test))


def run_xgboost_pipeline(train: pd.DataFrame, test: pd.DataFrame, drop_cols: Optional[List[str]] = None):
    """XGBoostによる予測パイプライン（学習・評価・可視化・SHAP分析）。

    Parameters:
    - train: 学習用DataFrame（call_num, cdr_date列を含む）
    - test: テスト用DataFrame（call_num, cdr_date列を含む）
    - drop_cols: 除外するカラム（list）
    """
    if drop_cols is None:
        drop_cols = DEFAULT_DROP_COLS

    X_train = train.drop(columns=drop_cols + ["cdr_date", "call_num"], errors="ignore")
    y_train = train["call_num"]
    X_test = test.drop(columns=drop_cols + ["cdr_date", "call_num"], errors="ignore")
    y_test = test["call_num"]

    model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    evaluate_forecast(y_test, y_pred)

    plot_df = test.loc[y_test.index, ["cdr_date"]].copy()
    plot_df["actual"] = y_test.values
    plot_df["predicted"] = y_pred

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["cdr_date"], plot_df["actual"], label="実測値", marker="o")
    plt.plot(plot_df["cdr_date"], plot_df["predicted"], label="予測値（XGBoost）", marker="o")
    plt.title("XGBoost による入電数予測")
    plt.xlabel("日付")
    plt.ylabel("入電数")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="dot")

    return model, X_train, y_train, X_test, y_test, y_pred


def get_train_subset(train_df: pd.DataFrame, target_row: pd.Series) -> pd.DataFrame:
    """target_rowと休日属性（holiday_flag等）が一致するtrainデータのサブセットを返す。"""
    condition = (
        (train_df["holiday_flag"] == target_row["holiday_flag"])
        & (train_df["day_before_holiday_flag"] == target_row["day_before_holiday_flag"])
        & (train_df["day_after_holiday_flag"] == target_row["day_after_holiday_flag"])
        & (train_df["dow"] == target_row["dow"])
    )
    return train_df[condition]


def predict_with_local_models(
    global_model: XGBRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    train_attr: pd.DataFrame,
    test_attr: pd.DataFrame,
    group_cols: Sequence[str] = GROUP_COLS,
    min_group_size: int = 10,
    log_target: bool = False,
) -> List[float]:
    """休日属性が一致するデータ件数が十分なテスト行はグループ専用のローカルモデルで、
    そうでない行はglobal_modelで予測する（条件付きモデル分割）。

    log_target=True の場合、y_train・global_modelはlog1pスケールで学習されている前提で、
    予測値をexpm1で元スケールに戻す。
    """
    y_preds = []
    for idx, test_row in X_test.iterrows():
        condition = pd.Series(True, index=train_attr.index)
        for col in group_cols:
            condition &= train_attr[col] == test_attr.loc[idx, col]
        subset_X = X_train[condition]
        subset_y = y_train[condition]

        if len(subset_X) >= min_group_size:
            local_model = XGBRegressor(
                n_estimators=100,
                max_depth=2,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
            )
            local_model.fit(subset_X, subset_y)
            pred = local_model.predict(test_row.values.reshape(1, -1))[0]
        else:
            pred = global_model.predict(test_row.values.reshape(1, -1))[0]

        y_preds.append(np.expm1(pred) if log_target else pred)

    return y_preds


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_trials: int = 100,
    log_target: bool = False,
) -> Dict:
    """OptunaでXGBoostのハイパーパラメータをチューニングし、best_paramsを返す。

    log_target=True の場合、y_trainはlog1pスケール、y_testは元スケールを渡し、
    RMSEは元スケールに戻して評価する。
    """

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 1, 100),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "objective": "reg:squarederror",
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        if log_target:
            preds = np.expm1(preds)
        return np.sqrt(mean_squared_error(y_test, preds))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best params:", study.best_params)
    return study.best_params


def save_model(model, filename: str, models_dir: Path = MODELS_DIR) -> Path:
    """学習済みモデルを outputs/models/ にjoblibで保存する。"""
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / filename
    joblib.dump(model, out_path)
    return out_path


def load_model(filename: str, models_dir: Path = MODELS_DIR):
    """joblibで保存したモデルを読み込む。"""
    return joblib.load(models_dir / filename)
