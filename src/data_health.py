"""資料品質自動健檢，搬自 ml_lab 的 01_feature_engineering.ipynb（步驟 0.5 / 2 / 3）。

背景：這個 repo 每次執行都用自己從 stock-system 抓下來的 stock_data.db 副本
重新計算 feature_panel（見 feature_engineering.py 開頭的說明），不會回頭去改
stock-system 的正式資料庫，兩邊完全獨立——所以這裡的自動修復只會讓「這個 repo
自己算出來的 feature_panel」更乾淨，不影響 stock-system 那邊的資料。

跟 01 刻意保持邏輯一致（README 已經說明：01 的邏輯之後如果調整，這裡要跟著
同步），這個檔案就是那個同步責任的第一次實踐：

  1) backfill_missing_dates()：缺交易日自動回補（01 的「步驟 0.5」）
  2) detect_halt_windows()    ：單日異常跳動（停牌/斷點）自動偵測，寫回
                                 data/known_halt_windows.json（01 的「步驟 2」）——
                                 台股漲跌停 ±10%，單日變動超過 JUMP_THRESHOLD_PCT
                                 幾乎只可能是停牌恢復交易的價格重置，不是正常波動
  3) fix_zero_close()         ：close=0/NULL 的資料庫層級修復（01 的「步驟 3」）——
                                 只在 FinMind 重新查到的值真的 >0 時才覆蓋，不會
                                 用同樣壞掉的資料蓋掉原本壞掉的資料

detect_halt_windows() 偵測到的排除區間，實際「排除 future_return_pct」的動作
在 feature_engineering.py 裡套用這份 json（只影響 label，不動特徵欄位本身，
跟 01 的設計一致）。
"""
import json
import sqlite3

import pandas as pd

from . import config

DATA_HEALTH_LOOKBACK_DAYS = 120     # 平常只檢查最近 4 個月；全歷史體檢時自行調大
JUMP_THRESHOLD_PCT = 12.0            # 台股漲跌停 ±10%，留一點緩衝抓門檻
FORWARD_BUFFER_TRADING_DAYS = 5      # 事件發生後再往後排除幾個交易日
N_FORWARD = 20                       # 跟 feature_engineering.py 的 N_FORWARD 保持一致

HALT_WINDOWS_PATH = config.DATA_DIR / "known_halt_windows.json"


def load_halt_windows() -> dict:
    if HALT_WINDOWS_PATH.exists():
        with open(HALT_WINDOWS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_halt_windows(windows: dict):
    HALT_WINDOWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(HALT_WINDOWS_PATH, "w", encoding="utf-8") as f:
        json.dump(windows, f, ensure_ascii=False, indent=2)


def backfill_missing_dates(db_path=None, token=None) -> int:
    """比對每檔股票在最近 DATA_HEALTH_LOOKBACK_DAYS 天內的交易日，缺的用 FinMind 補回。"""
    db_path = db_path or config.STOCK_DB_PATH
    token = token or config.FINMIND_TOKEN
    if not token:
        print("⚠️ 沒有 FINMIND_TOKEN，略過自動缺口修復，直接用現有資料繼續跑。")
        return 0

    from FinMind.data import DataLoader

    conn = sqlite3.connect(db_path)
    calendar_df = pd.read_sql("SELECT DISTINCT date FROM daily_price", conn)
    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    full_calendar = sorted(calendar_df["date"])
    if not full_calendar:
        conn.close()
        return 0

    cutoff = full_calendar[-1] - pd.Timedelta(days=DATA_HEALTH_LOOKBACK_DAYS)
    window_calendar = [d for d in full_calendar if d >= cutoff]

    per_stock = pd.read_sql(
        "SELECT stock_id, date FROM daily_price WHERE date >= ?",
        conn, params=(cutoff.strftime("%Y-%m-%d"),),
    )
    per_stock["date"] = pd.to_datetime(per_stock["date"])
    all_stock_ids = pd.read_sql("SELECT DISTINCT stock_id FROM daily_price", conn)["stock_id"].tolist()

    gaps_found = {}
    for sid in all_stock_ids:
        have = set(per_stock.loc[per_stock["stock_id"] == sid, "date"])
        missing = [d for d in window_calendar if d not in have]
        if missing:
            gaps_found[sid] = missing

    if not gaps_found:
        print(f"✅ 資料健康檢查（最近 {DATA_HEALTH_LOOKBACK_DAYS} 天）：沒有發現缺交易日。")
        conn.close()
        return 0

    print(f"⚠️ 發現 {len(gaps_found)} 檔股票有缺交易日，開始自動用 FinMind 補回：")
    api = DataLoader()
    api.login_by_token(api_token=token)
    total_written = 0
    for sid, missing_dates in gaps_found.items():
        fetch_start = (min(missing_dates) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        fetch_end = (max(missing_dates) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        try:
            df = api.taiwan_stock_daily(stock_id=sid, start_date=fetch_start, end_date=fetch_end)
        except Exception as e:
            print(f"  {sid}: FinMind 抓取失敗（{e}），略過，下次執行會再試一次")
            continue
        if df.empty:
            print(f"  {sid}: FinMind 在這段期間也查不到資料（可能真的停牌/下市），略過")
            continue
        df = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
        rows = df[["stock_id", "date", "open", "high", "low", "close", "volume"]].values.tolist()
        conn.executemany(
            "INSERT OR REPLACE INTO daily_price (stock_id, date, open, high, low, close, volume) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        print(f"  {sid}: 補回 {len(rows)} 筆（原本缺 {len(missing_dates)} 天）")
        total_written += len(rows)
    conn.close()
    print(f"自動修復完成，總共寫入 {total_written} 筆。")
    return total_written


def fix_zero_close(db_path=None, token=None) -> int:
    """重新用 FinMind 抓正確值，修復 close=0/NULL 的異常列（只在抓到的值真的 >0 時才覆蓋）。"""
    db_path = db_path or config.STOCK_DB_PATH
    token = token or config.FINMIND_TOKEN
    if not token:
        print("⚠️ 沒有 FINMIND_TOKEN，略過 close=0/NULL 的資料庫層級修復（feature_engineering "
              "計算特徵時的 ffill 還是會自動補上，不影響結果，只是治標不治本）。")
        return 0

    from FinMind.data import DataLoader

    conn = sqlite3.connect(db_path)
    bad_rows = pd.read_sql("SELECT stock_id, date FROM daily_price WHERE close IS NULL OR close <= 0", conn)
    if bad_rows.empty:
        print("✅ 沒有發現 close=0 或 NULL 的異常列。")
        conn.close()
        return 0

    print(f"⚠️ 發現 {len(bad_rows)} 筆 close=0/NULL 的異常列，開始自動用 FinMind 重新抓正確值：")
    api = DataLoader()
    api.login_by_token(api_token=token)
    total_fixed = 0
    for sid, group in bad_rows.groupby("stock_id"):
        bad_dates = sorted(group["date"].tolist())
        fetch_start = (pd.Timestamp(bad_dates[0]) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        fetch_end = (pd.Timestamp(bad_dates[-1]) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        try:
            df = api.taiwan_stock_daily(stock_id=sid, start_date=fetch_start, end_date=fetch_end)
        except Exception as e:
            print(f"  {sid}: FinMind 抓取失敗（{e}），略過")
            continue
        if df.empty:
            print(f"  {sid}: FinMind 這幾天也查不到資料，略過")
            continue
        df = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
        df = df[df["date"].isin(bad_dates) & (df["close"] > 0)]
        if df.empty:
            print(f"  {sid}: FinMind 重新查也是 0（資料源本身記錄壞掉，不一定是停牌——停牌通常連"
                  f"成交量也會是 0），維持原樣，交給 feature_engineering 的 ffill 處理")
            continue
        rows = df[["stock_id", "date", "open", "high", "low", "close", "volume"]].values.tolist()
        conn.executemany(
            "INSERT OR REPLACE INTO daily_price (stock_id, date, open, high, low, close, volume) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        print(f"  {sid}: 修復 {len(rows)} 筆（原本有 {len(bad_dates)} 筆異常）")
        total_fixed += len(rows)
    conn.close()
    print(f"自動修復完成，總共修正 {total_fixed} 筆。")
    return total_fixed


def detect_halt_windows(db_path=None) -> dict:
    """偵測單日異常跳動（|漲跌幅| > JUMP_THRESHOLD_PCT%），寫回 known_halt_windows.json。

    只有在兩個相鄰列真的是「日曆相鄰的交易日」時才判定為停牌事件——如果中間剛好
    缺了一個交易日（backfill_missing_dates 沒補到、或本來就查不到），兩個真實交易日
    的價格差被誤判成單日跳動的假訊號要排除，交給缺口回補處理，不能算成停牌。
    """
    db_path = db_path or config.STOCK_DB_PATH
    conn = sqlite3.connect(db_path)
    price_all = pd.read_sql("SELECT stock_id, date, close FROM daily_price", conn)
    conn.close()
    price_all["date"] = pd.to_datetime(price_all["date"])
    price_all = price_all.sort_values(["stock_id", "date"])

    known_halt_windows = load_halt_windows()

    def _covered(sid, event_date):
        for start, end in known_halt_windows.get(sid, []):
            if pd.Timestamp(start) <= event_date <= pd.Timestamp(end):
                return True
        return False

    calendar_list = sorted(price_all["date"].unique())
    calendar_index = {d: i for i, d in enumerate(calendar_list)}

    new_events = []
    skipped_due_to_gap = 0
    for sid, g in price_all.groupby("stock_id"):
        g = g.reset_index(drop=True)
        pct = g["close"].pct_change() * 100
        jump_positions = pct[pct.abs() > JUMP_THRESHOLD_PCT].index
        for pos in jump_positions:
            if pos == 0:
                continue
            prev_date = g.loc[pos - 1, "date"]
            curr_date = g.loc[pos, "date"]
            if calendar_index[curr_date] - calendar_index[prev_date] != 1:
                skipped_due_to_gap += 1
                continue
            event_date = curr_date
            if _covered(sid, event_date):
                continue
            start_pos = max(0, pos - N_FORWARD)
            end_pos = min(len(g) - 1, pos + FORWARD_BUFFER_TRADING_DAYS)
            window_start = g.loc[start_pos, "date"].strftime("%Y-%m-%d")
            window_end = g.loc[end_pos, "date"].strftime("%Y-%m-%d")
            known_halt_windows.setdefault(sid, []).append([window_start, window_end])
            new_events.append((sid, event_date.strftime("%Y-%m-%d"), pct.loc[pos], window_start, window_end))

    if skipped_due_to_gap:
        print(f"（另外有 {skipped_due_to_gap} 個疑似跳動其實是缺交易日造成的假訊號，已略過——"
              f"如果 backfill_missing_dates 有偵測到對應缺口，下次執行會被補上）")

    if new_events:
        save_halt_windows(known_halt_windows)
        print(f"⚠️ 自動偵測到 {len(new_events)} 個新的單日異常跳動事件，已加入排除清單並寫回 "
              f"{HALT_WINDOWS_PATH.name}：")
        for sid, d, p, ws, we in new_events:
            print(f"  {sid}  {d}  單日變動 {p:+.1f}%  → 排除區間 {ws} ~ {we}")
    else:
        print(f"✅ 沒有發現新的單日異常跳動事件（門檻 {JUMP_THRESHOLD_PCT}%）。")

    return known_halt_windows


def run_all(db_path=None):
    """依序執行三個健檢步驟。在 run_daily.py 裡排在 fetch_data 之後、feature_engineering 之前呼叫。"""
    db_path = db_path or config.STOCK_DB_PATH
    backfill_missing_dates(db_path)
    fix_zero_close(db_path)
    detect_halt_windows(db_path)


if __name__ == "__main__":
    run_all()
