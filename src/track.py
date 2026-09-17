"""把當天三個模型的預測寫進 tracking.db 的 predictions 表；
並回填「之前預測、現在已經知道實際報酬率」的資料列。

回填的原理：`feature_panel` 的 `future_return_pct` 是「往未來看 N 天」算出來的，
剛預測完的當下這個值一定是 NaN（未來還沒發生），但等過了足夠的交易日之後，
`feature_panel` 裡同一個 (stock_id, date) 的這個欄位就會自然被算出實際值——
所以回填只需要重新查一次 feature_panel，把還缺 actual_return 的列補上即可，
不需要另外自己去算報酬率。
"""
import sqlite3
import sys
import time

import pandas as pd

from . import config
from . import data as data_mod
from . import predict as predict_mod

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    date TEXT NOT NULL,
    model_name TEXT NOT NULL,
    stock_id TEXT NOT NULL,
    predicted_return REAL,
    rank INTEGER,
    top_n INTEGER,
    actual_return REAL,
    created_at TEXT,
    backfilled_at TEXT,
    PRIMARY KEY (date, model_name, stock_id)
);
"""


def _get_conn():
    config.TRACKING_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.TRACKING_DB_PATH)
    conn.execute(SCHEMA)
    return conn


def write_today_predictions(latest_date, results: dict):
    conn = _get_conn()
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    date_str = str(pd.Timestamp(latest_date).date())
    rows = []
    for model_name, df in results.items():
        for _, r in df.iterrows():
            rows.append((
                date_str, model_name, str(r["stock_id"]), float(r["predicted_return"]),
                int(r["rank"]), int(bool(r["top_n"])), None, now, None,
            ))
    conn.executemany(
        """INSERT OR REPLACE INTO predictions
           (date, model_name, stock_id, predicted_return, rank, top_n, actual_return, created_at, backfilled_at)
           VALUES (?, ?, ?, ?, ?, ?,
                   COALESCE((SELECT actual_return FROM predictions
                             WHERE date=? AND model_name=? AND stock_id=?), ?),
                   ?,
                   (SELECT backfilled_at FROM predictions
                    WHERE date=? AND model_name=? AND stock_id=?))""",
        [(d, m, s, p, rk, tn, d, m, s, a, c, d, m, s)
         for (d, m, s, p, rk, tn, a, c, _) in rows],
    )
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM predictions WHERE date=?", (date_str,)).fetchone()[0]
    conn.close()
    print(f"已寫入 {date_str} 的預測，共 {len(rows)} 筆（該日期累計 {n} 筆）。")


def backfill_actual_returns():
    """把 actual_return 還是 NULL、但 feature_panel 現在已經有值的列補上。"""
    conn = _get_conn()
    pending = pd.read_sql(
        "SELECT DISTINCT date, stock_id FROM predictions WHERE actual_return IS NULL", conn
    )
    if pending.empty:
        print("沒有待回填的預測列。")
        conn.close()
        return 0

    panel = data_mod.load_panel_for_predict()  # 有特徵即可，不要求 label 非 NaN
    panel_lookup = panel.set_index(["stock_id", panel["date"].dt.strftime("%Y-%m-%d")])[config.LABEL_COL]

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    updated = 0
    for _, row in pending.iterrows():
        key = (row["stock_id"], row["date"])
        if key not in panel_lookup.index:
            continue
        val = panel_lookup.loc[key]
        if pd.isna(val):
            continue
        val = float(val.iloc[0]) if hasattr(val, "iloc") else float(val)
        conn.execute(
            "UPDATE predictions SET actual_return=?, backfilled_at=? WHERE date=? AND stock_id=?",
            (val, now, row["date"], row["stock_id"]),
        )
        updated += 1
    conn.commit()
    conn.close()
    print(f"回填完成，共補上 {updated} 筆實際報酬率。")
    return updated


def run():
    latest_date, results = predict_mod.run()
    if not results:
        print("沒有任何模型產生預測，跳過寫入。", file=sys.stderr)
        return
    write_today_predictions(latest_date, results)
    backfill_actual_returns()


if __name__ == "__main__":
    run()
