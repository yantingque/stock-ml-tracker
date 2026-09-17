"""每日排程的進入點：抓最新資料庫 → 判斷是否重訓 → 預測今天 → 寫入/回填追蹤表。

在 GitHub Actions 裡透過 `python -m src.run_daily` 執行；
本機測試時如果已經手動把 stock_data.db 放在 data/ 底下，
可以加上 --skip-fetch 跳過下載那一步，用本地檔案直接測後面的邏輯。
"""
import sys

from . import fetch_data
from . import train
from . import track


def main():
    skip_fetch = "--skip-fetch" in sys.argv
    force_train = "--force-train" in sys.argv

    if not skip_fetch:
        fetch_data.fetch_latest_db()
    else:
        print("--skip-fetch：跳過下載，使用 data/ 底下現有的 stock_data.db")

    train.run(force=force_train)
    track.run()


if __name__ == "__main__":
    main()
