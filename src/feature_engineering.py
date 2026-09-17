"""重新計算 feature_panel。

背景：stock-system 自動發佈到 Release 的 stock_data.db，只有 daily_run.py
每天自動更新的原始表（daily_price、valuation、monthly_revenue、
financial_statement、balance_sheet、institutional...），並不包含
feature_panel——那張表在 ml_lab 專案裡是靠手動執行 01_feature_engineering.ipynb
才產生的。這支模組把該 notebook 的特徵計算邏輯原封不動搬過來，讓這個 repo
每天抓到最新原始資料後，能自己重新算出 feature_panel，不需要依賴人工介入。

特徵定義、計算方式刻意跟 01_feature_engineering.ipynb 保持逐行一致，
之後如果 01 那邊的邏輯有調整，這裡也要跟著同步修改。
"""
import sqlite3
import time

import numpy as np
import pandas as pd

from . import config

OUTPUT_TABLE = "feature_panel"
N_FORWARD = 20  # 預測目標：未來幾個交易日後的報酬率，跟 01 一致


def _compute_streak(s: pd.Series) -> pd.Series:
    pos = s > 0
    streak = pos.groupby((~pos).cumsum()).cumcount() + 1
    return streak.where(pos, 0)


def build_feature_panel(db_path=None) -> int:
    """重新計算 feature_panel 並覆蓋寫回資料庫。回傳寫入的列數。"""
    db_path = db_path or config.STOCK_DB_PATH
    t0 = time.time()

    conn = sqlite3.connect(db_path)
    price_all = pd.read_sql("SELECT stock_id, date, close, volume FROM daily_price", conn)
    valuation_all = pd.read_sql(
        "SELECT stock_id, date, per, pbr, dividend_yield FROM valuation", conn)
    revenue_all = pd.read_sql(
        "SELECT stock_id, revenue_year, revenue_month, revenue, effective_date "
        "FROM monthly_revenue", conn)
    eps_all = pd.read_sql(
        "SELECT stock_id, date, effective_date, value AS eps "
        "FROM financial_statement WHERE type='EPS'", conn)
    income_all = pd.read_sql(
        "SELECT stock_id, date, effective_date, value AS income_after_tax "
        "FROM financial_statement WHERE type='IncomeAfterTaxes'", conn)
    equity_all = pd.read_sql(
        "SELECT stock_id, date, effective_date, value AS equity "
        "FROM balance_sheet WHERE type='EquityAttributableToOwnersOfParent'", conn)
    inst_all = pd.read_sql(
        "SELECT stock_id, date, name, net FROM institutional "
        "WHERE name IN ('Foreign_Investor','Investment_Trust')", conn)
    conn.close()

    # ---- 基本面：先在原始頻率上算增率，再掛 effective_date ----
    revenue_all = revenue_all.sort_values(["stock_id", "revenue_year", "revenue_month"])
    revenue_all["revenue_prev_year"] = revenue_all.groupby("stock_id")["revenue"].shift(12)
    revenue_all["revenue_yoy_pct"] = (
        revenue_all["revenue"] / revenue_all["revenue_prev_year"] - 1
    ) * 100
    revenue_feat = revenue_all[["stock_id", "effective_date", "revenue_yoy_pct"]].dropna()
    revenue_feat = revenue_feat.rename(columns={"effective_date": "date"})

    eps_all = eps_all.sort_values(["stock_id", "date"])
    eps_all["eps_qoq_growth_pct"] = eps_all.groupby("stock_id")["eps"].pct_change() * 100
    eps_feat = eps_all[["stock_id", "effective_date", "eps", "eps_qoq_growth_pct"]].rename(
        columns={"effective_date": "date", "eps": "eps_latest"})

    roe_all = income_all.merge(
        equity_all, on=["stock_id", "date", "effective_date"], how="inner")
    roe_all["roe_pct"] = roe_all["income_after_tax"] * 4 / roe_all["equity"] * 100
    roe_feat = roe_all[["stock_id", "effective_date", "roe_pct"]].rename(
        columns={"effective_date": "date"})

    inst_pivot = inst_all.pivot_table(
        index=["stock_id", "date"], columns="name", values="net", aggfunc="sum"
    ).reset_index()
    inst_pivot["inst_net"] = (
        inst_pivot.get("Foreign_Investor", 0).fillna(0)
        + inst_pivot.get("Investment_Trust", 0).fillna(0)
    )
    inst_pivot = inst_pivot[["stock_id", "date", "inst_net"]]

    # ---- 技術面：逐股計算滾動指標 + label ----
    price_all["date"] = pd.to_datetime(price_all["date"])
    price_all = price_all.sort_values(["stock_id", "date"])

    groups = []
    for sid, px in price_all.groupby("stock_id"):
        if len(px) < 60:
            continue
        px = px.copy()
        close = px["close"]

        ma240 = close.rolling(240).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        px["ma240_gap_pct"] = (close - ma240) / ma240 * 100
        px["ma20_ma60_gap_pct"] = (ma20 - ma60) / ma60 * 100

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        px["macd_hist"] = macd_line - signal_line

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        px["rsi_14"] = 100 - 100 / (1 + rs)

        px["volume_chg_pct"] = px["volume"].pct_change() * 100
        px["volume_ratio_20d"] = px["volume"] / px["volume"].rolling(20).mean()

        px["momentum_20d_pct"] = (close / close.shift(20) - 1) * 100
        px["volatility_20d"] = close.pct_change().rolling(20).std() * 100

        px["future_close"] = close.shift(-N_FORWARD)
        px["future_return_pct"] = (px["future_close"] / close - 1) * 100

        groups.append(px)

    panel = pd.concat(groups, ignore_index=True)

    # ---- 基本面 as-of 合併 ----
    panel = panel.sort_values("date")
    for feat_df, cols in [
        (valuation_all, ["per", "pbr", "dividend_yield"]),
        (revenue_feat, ["revenue_yoy_pct"]),
        (eps_feat, ["eps_latest", "eps_qoq_growth_pct"]),
        (roe_feat, ["roe_pct"]),
        (inst_pivot, ["inst_net"]),
    ]:
        feat_df = feat_df.copy()
        feat_df["date"] = pd.to_datetime(feat_df["date"])
        feat_df = feat_df.sort_values("date")
        panel = pd.merge_asof(
            panel, feat_df[["stock_id", "date"] + cols],
            on="date", by="stock_id", direction="backward",
        )

    panel = panel.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # ---- 法人延伸特徵 ----
    panel["inst_net_3d"] = panel.groupby("stock_id")["inst_net"].transform(
        lambda s: s.rolling(3).sum()
    )
    panel["inst_buy_streak"] = panel.groupby("stock_id")["inst_net"].transform(_compute_streak)

    # ---- 寫回資料庫（整表覆蓋，理由跟 01 一致：滾動窗口牽動很多列，覆蓋最不容易出錯）----
    conn = sqlite3.connect(db_path)
    panel.to_sql(OUTPUT_TABLE, conn, if_exists="replace", index=False)
    conn.close()

    print(f"feature_panel 重新計算完成：{len(panel)} 列，耗時 {time.time()-t0:.1f}s")
    return len(panel)


if __name__ == "__main__":
    build_feature_panel()
