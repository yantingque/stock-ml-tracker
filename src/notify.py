"""把三個模型今天的 Top10 整理成一則文字訊息，用 LINE Messaging API 的
broadcast 端點發出去。

刻意選擇「新建一個獨立的 LINE 官方帳號 + broadcast」而不是 push 給特定
user/group id，是因為 broadcast 會發給所有加這個官方帳號好友的人——只要
這個新帳號只有你自己加好友，效果就等同於「只發給你」，但完全不需要另外
處理「怎麼取得自己在這個新 channel 底下的 userId」這個比較麻煩的步驟
（LINE 的 userId 是每個 channel 各自獨立的，不能沿用舊帳號的）。

需要環境變數 LINE_CHANNEL_ACCESS_TOKEN：在 LINE Developers Console 建立
新的 Messaging API channel 後，在該 channel 的 Messaging API 分頁
「Channel access token (long-lived)」按 Issue 產生。
"""
import os
import sqlite3
import sys

import pandas as pd
import requests

from . import config

LINE_BROADCAST_URL = "https://api.line.me/v2/bot/message/broadcast"

MODEL_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "lstm": "LSTM"}


def load_stock_names(db_path=None) -> dict:
    db_path = db_path or config.STOCK_DB_PATH
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT stock_id, stock_name FROM stock_info", conn)
    except Exception:  # noqa: BLE001  表存在與否不影響主流程，抓不到名稱就顯示代號就好
        return {}
    finally:
        conn.close()
    return dict(zip(df["stock_id"].astype(str), df["stock_name"]))


def format_message(latest_date, results: dict, stock_names: dict = None) -> str:
    stock_names = stock_names or {}
    date_str = pd.Timestamp(latest_date).date()
    lines = [f"📊 股票 ML 模型每日追蹤 - {date_str}"]

    for model_name in ("lightgbm", "xgboost", "lstm"):
        df = results.get(model_name)
        if df is None:
            continue
        top = df[df["top_n"]].sort_values("rank")
        lines.append(f"\n【{MODEL_LABELS[model_name]} Top{config.TOP_N}】")
        for _, r in top.iterrows():
            sid = str(r["stock_id"])
            name = stock_names.get(sid, "")
            lines.append(f"{int(r['rank'])}. {sid} {name}  預測 {r['predicted_return']:+.1f}%")

    lines.append("\n（實驗性模型預測，僅供追蹤參考，非投資建議）")
    return "\n".join(lines)


def send_line_broadcast(message: str, token: str = None) -> bool:
    token = token or os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
    if not token:
        print("找不到 LINE_CHANNEL_ACCESS_TOKEN 環境變數，跳過 LINE 通知。", file=sys.stderr)
        return False

    # LINE 文字訊息單則長度上限是 5000 字，這裡的內容（3模型 x 10支)通常遠低於此，
    # 保險起見還是裁切一下避免萬一超過時整支請求被拒絕。
    text = message[:5000]

    resp = requests.post(
        LINE_BROADCAST_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={"messages": [{"type": "text", "text": text}]},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"LINE 廣播失敗：HTTP {resp.status_code} {resp.text}", file=sys.stderr)
        return False
    print("LINE 廣播已送出。")
    return True


def notify(latest_date, results: dict):
    stock_names = load_stock_names()
    message = format_message(latest_date, results, stock_names)
    return send_line_broadcast(message)


if __name__ == "__main__":
    # 本機測試用：只印出訊息內容，預設不會真的發送，
    # 想實際測試發送效果的話加 --send（要先 export LINE_CHANNEL_ACCESS_TOKEN）。
    from . import predict as predict_mod

    latest_date, results = predict_mod.run()
    names = load_stock_names()
    msg = format_message(latest_date, results, names)
    print(msg)
    print(f"\n(字數: {len(msg)})")

    if "--send" in sys.argv:
        send_line_broadcast(msg)
    else:
        print("\n（預覽模式，未實際發送；要實際測試發送請加 --send 參數）")
