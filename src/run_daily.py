"""每日排程的進入點：抓最新資料庫 → 資料品質健檢 → 重新計算 feature_panel →
判斷是否重訓 → 預測今天 → 寫入/回填追蹤表。

stock-system 自動發佈的 Release 附件只有原始表，不包含 feature_panel
（那張表在 ml_lab 是靠手動跑 01_feature_engineering.ipynb 才產生的），
所以這裡每次都會自己重新計算一次，不依賴人工介入。

data_health.run_all()（缺交易日回補、停牌事件偵測、close=0/NULL 修復）一定要排在
feature_engineering 之前：feature_panel 是照當下的 daily_price 算出來的，越乾淨的
輸入才有越乾淨的特徵/label。

在 GitHub Actions 裡透過 `python -m src.run_daily` 執行；
本機測試時如果已經手動把 stock_data.db 放在 data/ 底下，
可以加上 --skip-fetch 跳過下載那一步，用本地檔案直接測後面的邏輯。
"""
import sys

from . import fetch_data
from . import data_health
from . import feature_engineering
from . import train
from . import track


def main():
    skip_fetch = "--skip-fetch" in sys.argv
    force_train = "--force-train" in sys.argv

    if not skip_fetch:
        fetch_data.fetch_latest_db()
    else:
        print("--skip-fetch：跳過下載，使用 data/ 底下現有的 stock_data.db")

    data_health.run_all()
    feature_engineering.build_feature_panel()
    train.run(force=force_train)
    track.run()


if __name__ == "__main__":
    main()
