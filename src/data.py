"""讀取 feature_panel、整理成模型可以用的格式。

樹模型（LightGBM/XGBoost）用橫斷面資料（每列 = 一支股票在某一天的16個特徵）；
LSTM 需要序列資料（每支股票過去 LOOKBACK 天的特徵），這裡也一併處理。
"""
import sqlite3
import numpy as np
import pandas as pd

from . import config


def load_panel_clean(db_path=None) -> pd.DataFrame:
    """讀 feature_panel，丟掉特徵或 label 有缺值的列，回傳依 stock_id, date 排序好的表。"""
    db_path = db_path or config.STOCK_DB_PATH
    conn = sqlite3.connect(db_path)
    cols_sql = ", ".join(["stock_id", "date"] + config.FEATURE_COLS + [config.LABEL_COL])
    panel = pd.read_sql(f"SELECT {cols_sql} FROM feature_panel", conn)
    conn.close()

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel_clean = panel.dropna(subset=config.FEATURE_COLS + [config.LABEL_COL]).reset_index(drop=True)
    panel_clean = panel_clean.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return panel_clean


def load_panel_for_predict(db_path=None) -> pd.DataFrame:
    """讀 feature_panel 用來『預測』：只要求特徵欄位齊全，不要求 label（因為未來報酬率還不知道，本來就是 NaN）。"""
    db_path = db_path or config.STOCK_DB_PATH
    conn = sqlite3.connect(db_path)
    cols_sql = ", ".join(["stock_id", "date"] + config.FEATURE_COLS + [config.LABEL_COL])
    panel = pd.read_sql(f"SELECT {cols_sql} FROM feature_panel", conn)
    conn.close()

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.replace([np.inf, -np.inf], np.nan)
    panel_clean = panel.dropna(subset=config.FEATURE_COLS).reset_index(drop=True)
    panel_clean = panel_clean.sort_values(["stock_id", "date"]).reset_index(drop=True)
    return panel_clean


def trading_dates(panel: pd.DataFrame) -> np.ndarray:
    """全部出現過的交易日，由舊到新排序。"""
    return np.sort(panel["date"].unique())


def trading_days_since(panel: pd.DataFrame, since_date) -> int:
    """算「最新一個交易日」距離 since_date 之後，總共經過了幾個交易日（不含 since_date 當天）。"""
    dates = trading_dates(panel)
    since_date = pd.Timestamp(since_date)
    return int((dates > since_date).sum())


def build_sequences(panel_clean: pd.DataFrame, lookback: int = None):
    """把橫斷面 panel 轉成 LSTM 用的序列（每支股票用過去 lookback 天的特徵預測當天的 label）。

    回傳：X_all (n, lookback, n_feat), y_all (n,), dates_all (n,), stock_all (n,), last_day_feat_all (n, n_feat)
    """
    lookback = lookback or config.LOOKBACK
    feature_cols = config.FEATURE_COLS
    n_feat = len(feature_cols)

    X_list, y_list, date_list, stock_list = [], [], [], []
    for sid, g in panel_clean.groupby("stock_id"):
        g = g.sort_values("date")
        feat_mat = g[feature_cols].values
        labels = g[config.LABEL_COL].values
        dates = g["date"].values
        n = len(g)
        if n < lookback:
            continue
        for i in range(lookback - 1, n):
            X_list.append(feat_mat[i - lookback + 1: i + 1])
            y_list.append(labels[i])
            date_list.append(dates[i])
            stock_list.append(sid)

    X_all = np.array(X_list, dtype=np.float32)
    y_all = np.array(y_list, dtype=np.float32)
    dates_all = np.array(date_list)
    stock_all = np.array(stock_list)
    last_day_feat_all = X_all[:, -1, :] if len(X_all) else np.zeros((0, n_feat), dtype=np.float32)
    return X_all, y_all, dates_all, stock_all, last_day_feat_all


def build_latest_sequences_for_predict(panel_for_predict: pd.DataFrame, lookback: int = None):
    """只取『最新一天』每支股票的序列（用來做今天的預測，不要求 label 有值）。

    回傳：X_today (n, lookback, n_feat), stock_today (n,), today_date, feat_today (n, n_feat)
    """
    lookback = lookback or config.LOOKBACK
    feature_cols = config.FEATURE_COLS
    n_feat = len(feature_cols)
    latest_date = panel_for_predict["date"].max()

    X_list, stock_list, feat_list = [], [], []
    for sid, g in panel_for_predict.groupby("stock_id"):
        g = g.sort_values("date")
        if g["date"].iloc[-1] != latest_date:
            continue  # 這支股票最新一筆不是最新交易日，代表今天沒資料，跳過
        if len(g) < lookback:
            continue
        feat_mat = g[feature_cols].values
        X_list.append(feat_mat[-lookback:])
        stock_list.append(sid)
        feat_list.append(feat_mat[-1])

    if not X_list:
        return (np.zeros((0, lookback, n_feat), dtype=np.float32), np.array([]),
                latest_date, np.zeros((0, n_feat), dtype=np.float32))

    X_today = np.array(X_list, dtype=np.float32)
    stock_today = np.array(stock_list)
    feat_today = np.array(feat_list, dtype=np.float32)
    return X_today, stock_today, latest_date, feat_today
