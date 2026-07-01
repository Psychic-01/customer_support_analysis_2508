# customer_support_analysis_2508

ある企業のヘルプデスク入電数（コールセンター問合せ件数）を分析し、日次の入電数を予測するプロジェクト。
EDAで変動要因を洗い出したうえで、統計モデル（SARIMA）と機械学習モデル（Prophet, XGBoost）を比較し、
特徴量エンジニアリングとハイパーパラメータチューニングでXGBoostモデルを改善、
最終的に必要な応答キャパシティの試算まで行う。

## ディレクトリ構成

```
.
├── data/
│   ├── raw/            # 元データCSV（各自配置、Git管理外）
│   └── processed/      # 前処理済みデータ（Notebook実行で生成、Git管理外）
├── notebooks/
│   ├── 01_eda.ipynb                        # データ準備・EDA
│   ├── 02_baseline_models.ipynb            # 課題仮説・SARIMA/Prophet/XGBoost比較
│   ├── 03_feature_engineering_tuning.ipynb # 特徴量エンジニアリング・チューニング
│   └── 04_effect_measurement.ipynb         # 効果測定
├── outputs/
│   ├── figures/         # 保存した図
│   └── models/          # 学習済みモデル（Git管理外）
├── src/
│   ├── data/loader.py           # データ読込・カレンダーマージ・空白期間補完
│   ├── visualization/plots.py   # EDA可視化関数
│   ├── features/engineering.py  # 特徴量エンジニアリング関数
│   └── models/forecasting.py    # モデル学習・チューニング・セグメント予測
└── requirements.txt
```

## セットアップ手順

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`data/raw/` に以下の5ファイルを配置する（元データはGoogle Drive上にあり本リポジトリには同梱していない）。

- `calender_data .csv`
- `cm_data .csv`
- `gt_service_name .csv`
- `regi_acc_get_data_transform .csv`
- `regi_call_data_transform .csv`

## Notebookの実行順

`notebooks/` 配下を番号順に実行する。各Notebookは前段の出力（`data/processed/` 配下のCSV、`outputs/models/` 配下のモデル）を読み込む前提。

1. `01_eda.ipynb` — 生データ読込・前処理・EDA。`data/processed/` に前処理済みテーブルを保存する
2. `02_baseline_models.ipynb` — 課題仮説の整理とSARIMA/Prophet/XGBoostのベースライン比較
3. `03_feature_engineering_tuning.ipynb` — 特徴量エンジニアリングとOptunaチューニングで精度改善し、最終モデルを `outputs/models/` に保存する
4. `04_effect_measurement.ipynb` — 最終モデルの予測結果から応答率・必要キャパシティを試算する
