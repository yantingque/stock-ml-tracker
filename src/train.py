"""判斷是否到了重訓時機（每 REBALANCE_STEP=20 個交易日一次），是的話重新訓練三個模型並存檔。

跟 ml_lab 的 04~09 notebook 不同：notebook 是為了「回測驗證」，所以要模擬很多輪
walk-forward；這裡是正式環境，只需要「用目前為止的全部歷史資料訓練一次、供接下來
一段時間預測使用」，重訓的節奏才會需要另外一個狀態檔來記錄「上次是哪一天訓練的」。
"""
import json
import sys
import time

from . import config
from . import data as data_mod
from . import models


def _load_state():
    if config.STATE_PATH.exists():
        with open(config.STATE_PATH) as f:
            return json.load(f)
    return {}


def _save_state(state: dict):
    config.STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.STATE_PATH, "w") as f:
        json.dump(state, f, default=str)


def should_retrain(panel_clean, state: dict) -> bool:
    last_train_date = state.get("last_train_date")
    if not last_train_date:
        return True  # 從來沒訓練過（第一次跑），一定要訓練
    days_since = data_mod.trading_days_since(panel_clean, last_train_date)
    return days_since >= config.REBALANCE_STEP


def run(force: bool = False):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    panel_clean = data_mod.load_panel_clean()
    if panel_clean.empty:
        print("feature_panel 清理後沒有可用資料，中止訓練。", file=sys.stderr)
        sys.exit(1)

    state = _load_state()
    latest_date = panel_clean["date"].max()

    if not force and not should_retrain(panel_clean, state):
        last = state.get("last_train_date")
        days_since = data_mod.trading_days_since(panel_clean, last)
        print(f"距離上次重訓（{last}）只過了 {days_since} 個交易日，"
              f"還沒到 {config.REBALANCE_STEP} 天門檻，跳過本次重訓。")
        return False

    print(f"開始重訓（資料最新到 {latest_date}）...")
    t0 = time.time()

    # ---- 樹模型：用全部乾淨資料訓練 ----
    X_tree = panel_clean[config.FEATURE_COLS].values
    y_tree = panel_clean[config.LABEL_COL].values
    lgb_model, xgb_model = models.train_tree_models(X_tree, y_tree)
    models.save_tree_model(lgb_model, "lightgbm")
    models.save_tree_model(xgb_model, "xgboost")
    print(f"  LightGBM/XGBoost 訓練完成，耗時 {time.time()-t0:.1f}s")

    # ---- LSTM：用序列資料訓練 ----
    t1 = time.time()
    X_seq, y_seq, dates_seq, stock_seq, _ = data_mod.build_sequences(panel_clean)
    if len(X_seq) == 0:
        print("  警告：序列資料筆數為 0（歷史天數可能還不滿 LOOKBACK），跳過 LSTM 訓練。")
    else:
        n_feat = X_seq.shape[2]
        X_seq_scaled, _, scaler = models.scale_sequences(X_seq, X_seq[:0], n_feat)
        lstm_model = models.train_lstm(X_seq_scaled, y_seq, n_feat)
        models.save_lstm_model(lstm_model, scaler)
        print(f"  LSTM 訓練完成（序列數 {len(X_seq)}），耗時 {time.time()-t1:.1f}s")

    state["last_train_date"] = str(latest_date)
    state["last_train_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save_state(state)
    print(f"重訓全部完成，總耗時 {time.time()-t0:.1f}s，狀態已更新（last_train_date={latest_date}）。")
    return True


if __name__ == "__main__":
    force_flag = "--force" in sys.argv
    run(force=force_flag)
