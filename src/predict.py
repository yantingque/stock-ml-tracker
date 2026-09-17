"""讀取目前存檔的三個模型，對『最新一個交易日』的候選股票產生預測與 Top10 名單。

樹模型（LightGBM/XGBoost）只要當天特徵齊全就能預測；LSTM 額外要求該股票要有
過去 LOOKBACK(60) 個交易日的連續歷史，所以 LSTM 的候選股票數通常會比樹模型少一些
——這點跟 09 notebook 裡「LSTM 樣本數比 04~08 少」是同樣的原因，不是 bug。
"""
import sys

import pandas as pd

from . import config
from . import data as data_mod
from . import models


def _rank_df(stock_ids, preds, top_n=None):
    top_n = top_n or config.TOP_N
    df = pd.DataFrame({"stock_id": stock_ids, "predicted_return": preds})
    df = df.sort_values("predicted_return", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["top_n"] = df["rank"] <= top_n
    return df


def run():
    panel = data_mod.load_panel_for_predict()
    if panel.empty:
        print("feature_panel 沒有可用資料，無法預測。", file=sys.stderr)
        sys.exit(1)

    latest_date = panel["date"].max()
    today_rows = panel[panel["date"] == latest_date].copy()

    results = {}

    # ---- LightGBM / XGBoost：橫斷面預測 ----
    X_today = today_rows[config.FEATURE_COLS].values
    for name in ("lightgbm", "xgboost"):
        model = models.load_tree_model(name)
        if model is None:
            print(f"警告：找不到已存檔的 {name} 模型，先執行過 train.py 才有模型可用。", file=sys.stderr)
            continue
        preds = model.predict(X_today)
        results[name] = _rank_df(today_rows["stock_id"].values, preds)

    # ---- LSTM：序列預測 ----
    lstm_model, lstm_scaler = models.load_lstm_model()
    if lstm_model is None:
        print("警告：找不到已存檔的 LSTM 模型，先執行過 train.py 才有模型可用。", file=sys.stderr)
    else:
        X_seq_today, stock_today, seq_date, _ = data_mod.build_latest_sequences_for_predict(panel)
        if seq_date != latest_date:
            print(f"提醒：LSTM 序列資料的最新日期（{seq_date}）跟橫斷面最新日期（{latest_date}）不同，"
                  f"以橫斷面日期為準記錄。")
        if len(X_seq_today) == 0:
            print("警告：目前沒有股票的歷史天數足夠組成 LSTM 序列，跳過 LSTM 預測。")
        else:
            preds = models.predict_lstm(lstm_model, lstm_scaler, X_seq_today)
            results["lstm"] = _rank_df(stock_today, preds)

    return latest_date, results


if __name__ == "__main__":
    latest_date, results = run()
    print(f"預測日期：{latest_date}")
    for name, df in results.items():
        print(f"\n=== {name} Top{config.TOP_N} ===")
        print(df[df["top_n"]][["stock_id", "predicted_return", "rank"]].to_string(index=False))
