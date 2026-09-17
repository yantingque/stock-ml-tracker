"""從 stock-system（private repo）抓最新 Release 的 stock_data.db 附件。

需要環境變數 STOCK_SYSTEM_TOKEN：一個對 stock-system 有 Contents:Read-only
權限的 fine-grained GitHub Personal Access Token，存在這個 repo 的
GitHub Actions Secret 裡（Settings → Secrets and variables → Actions）。
"""
import os
import sys
import requests

from . import config

GITHUB_API = "https://api.github.com"


def fetch_latest_db(dest_path=None, token=None):
    dest_path = dest_path or config.STOCK_DB_PATH
    token = token or os.environ.get("STOCK_SYSTEM_TOKEN")
    if not token:
        raise RuntimeError(
            "找不到 STOCK_SYSTEM_TOKEN 環境變數。本機測試時可以 export "
            "STOCK_SYSTEM_TOKEN=你的token 再執行；在 GitHub Actions 裡則是從 "
            "repo 的 Secrets 設定讀取。"
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # 1) 找最新的 release
    resp = requests.get(
        f"{GITHUB_API}/repos/{config.STOCK_SYSTEM_REPO}/releases/latest",
        headers=headers, timeout=30,
    )
    resp.raise_for_status()
    release = resp.json()

    # 2) 在附件清單裡找到 stock_data.db
    asset = next(
        (a for a in release.get("assets", []) if a["name"] == config.STOCK_SYSTEM_ASSET_NAME),
        None,
    )
    if asset is None:
        names = [a["name"] for a in release.get("assets", [])]
        raise RuntimeError(
            f"最新 release（{release.get('tag_name')}）裡找不到附件 "
            f"'{config.STOCK_SYSTEM_ASSET_NAME}'，實際附件有：{names}"
        )

    # 3) 下載附件（private repo 的附件要用 asset API + 專屬的 Accept header，
    #    不能直接用 browser_download_url，那個網址對 private repo 會需要另外的認證流程）
    asset_headers = dict(headers, Accept="application/octet-stream")
    with requests.get(asset["url"], headers=asset_headers, stream=True, timeout=300) as r:
        r.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

    print(f"已下載 {config.STOCK_SYSTEM_ASSET_NAME}（release {release.get('tag_name')}）到 {dest_path}")
    return dest_path


if __name__ == "__main__":
    try:
        fetch_latest_db()
    except Exception as e:  # noqa: BLE001
        print(f"下載失敗：{e}", file=sys.stderr)
        sys.exit(1)
