"""共用設定：路徑、特徵欄位、模型超參數。

這裡的設定值都是延續 ml_lab 專案 01~09 notebook 已經驗證過的設定，
目的是讓「正式追蹤系統」跟「實驗 notebook」的邏輯是同一套，
之後如果要回頭對照結果會比較好對得起來。
"""
from pathlib import Path

# ---- 路徑 ----
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

STOCK_DB_PATH = DATA_DIR / "stock_data.db"       # 從 stock-system 抓下來的最新資料庫（含 feature_panel）
TRACKING_DB_PATH = DATA_DIR / "tracking.db"      # 這個 repo 自己維護的追蹤資料庫（predictions 表）
STATE_PATH = DATA_DIR / "train_state.json"       # 記錄「上次重訓是哪一天」的狀態檔

# ---- stock-system repo（用來抓最新 Release 附件）----
STOCK_SYSTEM_REPO = "yantingque/stock-system"
STOCK_SYSTEM_ASSET_NAME = "stock_data.db"        # Release 附件檔名，若正式系統那邊改了檔名要跟著改

# ---- 特徵欄位（跟 01_feature_engineering.ipynb 一致，16 個特徵）----
FEATURE_COLS = [
    "ma240_gap_pct", "ma20_ma60_gap_pct", "macd_hist", "rsi_14", "volume_chg_pct",
    "volume_ratio_20d", "dividend_yield", "revenue_yoy_pct", "eps_qoq_growth_pct",
    "roe_pct", "inst_net_3d", "per", "pbr", "momentum_20d_pct", "volatility_20d",
    "inst_buy_streak",
]
LABEL_COL = "future_return_pct"

# ---- walk-forward / 重訓設定（跟 04~09 notebook 一致）----
TOP_N = 10
REBALANCE_STEP = 20      # 每幾個交易日重新訓練一次
LOOKBACK = 60            # LSTM 用的序列長度（過去 N 個交易日）

# ---- LightGBM / XGBoost（跟 08 一致：用 TimeSeriesSplit 做因果調參）----
TREE_PARAM_GRID = {"max_depth": [3, 4], "learning_rate": [0.05, 0.1]}
TREE_N_ESTIMATORS = 150

# ---- LSTM（跟 09 一致：雲端/CPU 也能在合理時間內跑完的保守設定）----
LSTM_HIDDEN = 24
LSTM_EPOCHS = 6
LSTM_BATCH_SIZE = 256

MODEL_NAMES = ["lightgbm", "xgboost", "lstm"]
