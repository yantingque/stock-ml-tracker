# stock-ml-tracker

這個 repo 是「多策略自動選股系統」底下 `ml_lab` 實驗成果的**正式化追蹤系統**。

跟正式運作中的規則系統（`stock-system` repo 的 `daily_run.py` / LINE Bot）是**完全獨立、並行**的兩套系統：
規則系統照舊每天自動選股、推播 LINE，不受這裡影響；這個 repo 只是額外每天用三個機器學習模型
（LightGBM、XGBoost、LSTM）各自產生一次 Top10 預測，記錄下來，讓我們可以持續累積資料，
之後拿來跟規則系統、以及三個模型彼此之間做統計比較，觀察誰的表現比較穩定可信。

這三個模型是 `ml_lab` 專案的 `08_lightgbm_xgboost.ipynb`、`09_lstm_walkforward.ipynb`
驗證過的設定，細節、統計檢定結果可以參考那個 repo 的 README 實驗日誌。

## 架構

```
.github/workflows/daily_track.yml   # 每天排程觸發的入口
src/
  config.py       # 共用設定：路徑、16個特徵欄位、模型超參數、FINMIND_TOKEN
  fetch_data.py   # 從 stock-system 抓最新 Release 的 stock_data.db
  data_health.py  # 資料品質自動健檢：缺交易日回補、停牌事件偵測、close=0/NULL 修復
                  # （搬自 ml_lab 的 01_feature_engineering.ipynb，寫回 data/known_halt_windows.json）
  feature_engineering.py  # 從原始表重新計算 feature_panel（搬自 ml_lab 的 01_feature_engineering.ipynb），
                  # 並套用 known_halt_windows.json 排除受停牌污染的 label
  data.py         # 讀 feature_panel、組成橫斷面/序列資料
  models.py       # LightGBM/XGBoost（TimeSeriesSplit調參）、LSTM（PyTorch）訓練與存讀檔
  train.py        # 判斷是否到了重訓時機（每20個交易日），是的話重訓三個模型
  predict.py      # 用目前存檔的模型，對最新一天的候選股票產生 Top10 預測
  track.py        # 把預測寫進 data/tracking.db，並回填之前預測、現在已知的實際報酬率
  notify.py       # 把三個模型的 Top10 整理成文字訊息，用 LINE Messaging API 廣播出去
  run_daily.py    # 每日流程的進入點：fetch → 資料健檢 → 重算 feature_panel → train（視情況）→ predict → track → LINE 通知
data/
  tracking.db     # 這個 repo 自己維護的追蹤資料庫（predictions 表），由 Actions 每天自動更新、commit
  train_state.json# 記錄「上次重訓是哪一天」
  known_halt_windows.json  # 累積偵測到的停牌/單日異常跳動排除區間，由 Actions 每天自動更新、commit
models/           # 最新一次訓練好的模型檔（.pkl / .pt），由 Actions 每天自動更新、commit
```

`data/stock_data.db`（從 `stock-system` 抓下來的原始資料庫）**不會**被 commit 進這個 repo
（`.gitignore` 已排除），因為那份資料的來源另有 `stock-system` repo 自己的版本控管，
這裡只是每次執行時抓一份暫用。

## 每天自動跑的流程

`daily_track.yml` 排程（預設每天 UTC 10:00，約台灣時間 18:00，可在檔案裡調整）會依序執行：

1. **fetch_data.py**：用 `STOCK_SYSTEM_TOKEN` 這個 Secret，呼叫 GitHub API 抓 `stock-system` 最新 Release 的 `stock_data.db`——**這份 db 只有原始表**（`daily_price`、`institutional`...），因為 `daily_run.py` 的每日自動化不會算 `feature_panel`
2. **data_health.py**：資料品質自動健檢——缺交易日回補、單日異常跳動（停牌）自動偵測並寫回 `known_halt_windows.json`、close=0/NULL 資料庫層級修復（後兩步靠自己本機邏輯或 FinMind，沒設定 `FINMIND_TOKEN` 也不會讓流程失敗，只是缺口回補/close=0 修復這兩步會被跳過）
3. **feature_engineering.py**：從這些原始表**自己重新計算一次 `feature_panel`**（邏輯逐行搬自 `ml_lab` 的 `01_feature_engineering.ipynb`），並套用上一步的 `known_halt_windows.json`，把落在排除區間內的列的 `future_return_pct` 蓋成 NaN，不依賴人工手動跑過 01
4. **train.py**：檢查距離上次重訓過了幾個交易日，滿 20 天才會重新訓練三個模型（訓練成本低，正式環境不需要每天都重訓，這跟 `ml_lab` notebook 裡驗證過的 walk-forward 節奏一致）
5. **predict.py**：用目前存檔的模型，對「今天」的候選股票各自產生 Top10 預測
6. **track.py**：把今天的預測寫進 `tracking.db`；同時回填 20 個交易日前那批預測、現在已經知道的實際報酬率（`feature_panel` 的 `future_return_pct` 到那時候就會自然算出來，不需要另外自己算）
7. **notify.py**：把三個模型今天的 Top10 整理成一則文字訊息，用 LINE Messaging API 的 broadcast 發出去（見下方設定說明）
8. 把更新後的 `tracking.db`、`train_state.json`、`known_halt_windows.json`、`models/` 自動 commit、push 回這個 repo

> 這個 repo 算出來的 `feature_panel` 只存在**這裡自己抓下來的那份 db 副本**裡，不會回頭去改
> `stock-system` 的正式資料庫，兩邊完全獨立，不用擔心互相干擾。

也可以在 GitHub 網頁上手動點 **Actions → Daily model tracking → Run workflow** 立即觸發一次，不用等排程時間。

## 設定 Secret（一次性設定）

因為 `stock-system` 是 private repo，這裡需要一個有讀取權限的 GitHub token 才能自動抓到最新的 db：

1. GitHub 右上角頭像 → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token
2. Repository access 選 **Only select repositories** → 選 `stock-system`
3. Permissions → Repository permissions → **Contents: Read-only**
4. 產生後複製 token，貼到這個 repo 的 **Settings → Secrets and variables → Actions → New repository secret**，Name 填 `STOCK_SYSTEM_TOKEN`

Token 有到期日，快到期時 GitHub 會提醒，記得回來更新。

## 設定 FINMIND_TOKEN（選用，沒設定也不會壞）

`data_health.py` 的缺交易日回補、close=0/NULL 修復這兩步需要呼叫 FinMind API 補資料，
需要一個 FinMind token（跟 `03_專題_股票分析/code/config_template.py` 裡用的是同一種）：

1. 把 token 貼到這個 repo 的 **Settings → Secrets and variables → Actions → New repository secret**，Name 填 `FINMIND_TOKEN`
2. 沒有設定也沒關係，`data_health.py` 會印一行提醒然後跳過這兩步，不會讓整條每日流程失敗——停牌事件偵測不需要 FinMind，不受影響，還是會照常執行

## 設定 LINE 通知（另建一個獨立的官方帳號，跟現有規則系統的 LINE Bot 分開）

刻意用一個全新、獨立的 LINE 官方帳號來發這個系統的通知，不會跟現有規則系統的正式通知混在一起。
發送方式用 LINE Messaging API 的 **broadcast**（廣播給所有加這個官方帳號好友的人），
好處是完全不需要另外處理「取得自己的 userId」這個步驟——因為這是全新帳號，只要你自己加好友，
broadcast 效果就等同於「只發給你」。

1. 到 [LINE Developers Console](https://developers.line.biz/console/)，用你的 LINE 帳號登入
2. 建立一個新的 Provider（或沿用現有的都可以），底下建立一個新的 **Messaging API channel**（這就是一個全新的官方帳號，跟正式系統用的那個是分開的）
3. 填基本資料（Channel name 例如「股票ML追蹤」，icon 隨意）
4. 建立後進到該 channel 的 **Messaging API** 分頁，把 **Auto-reply messages**、**Greeting messages** 都關掉（避免這個帳號自動回覆奇怪的訊息，我們只需要它單向發通知）
5. 同一頁面找到 **Channel access token (long-lived)**，按 **Issue** 產生 token，複製起來
6. 頁面上方會有加好友用的 QR code 或連結，用你自己的 LINE 掃碼/點擊加這個新官方帳號好友（只有加好友的人才收得到 broadcast，所以只加你自己就好）
7. 把 token 貼到 `stock-ml-tracker` 的 **Settings → Secrets and variables → Actions → New repository secret**，Name 填 `LINE_CHANNEL_ACCESS_TOKEN`

沒有設定這個 Secret 也沒關係，`notify.py` 會印一行提醒然後跳過，不會讓整條每日流程失敗。

## 本機測試

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 如果本機已經手動有一份 data/stock_data.db，可以跳過下載直接測後面邏輯：
python -m src.run_daily --skip-fetch

# 想強制重訓（不管距離上次重訓幾天）：
python -m src.run_daily --skip-fetch --force-train

# 只想看某天的預測結果，不寫入追蹤表：
python -m src.predict

# 預覽 LINE 通知訊息長怎樣（預設不會真的發送）：
python -m src.notify
# 想實際測試發送效果（需先 export LINE_CHANNEL_ACCESS_TOKEN=你的token）：
python -m src.notify --send
```

## 之後怎麼做比較

`data/tracking.db` 的 `predictions` 表結構：

| 欄位 | 說明 |
|---|---|
| `date` | 預測的日期（該股票當天的候選日） |
| `model_name` | `lightgbm` / `xgboost` / `lstm` |
| `stock_id` | 股票代號 |
| `predicted_return` | 模型預測的未來報酬率 |
| `rank` | 當天該模型預測值的排名（1 = 最看好） |
| `top_n` | 是否進入該模型當天的 Top10（1/0） |
| `actual_return` | 實際報酬率，預測當下是 NULL，約 20 個交易日後才會被回填 |

等累積到有意義的樣本數之後，可以用類似 `ml_lab` 的 `03`/`04`/`08`/`09` notebook 那套配對統計檢定
（`scipy.stats.ttest_rel`、`wilcoxon`）比較各模型 `top_n=1` 那些列的 `actual_return` 平均值，
以及跟規則系統（`stock-system` 的 `selection_log`）的歷史選股做比較。

## 已知限制 / 保留意見

延續 `ml_lab` notebook 的保留意見，這裡列出的是「值得持續追蹤的訊號」，還不是「已驗證可信賴」：

- LSTM 目前只用單一隨機種子、保守的小架構（`HIDDEN=24`、`EPOCHS=6`），還沒有像 LightGBM/XGBoost 一樣認真調參，`09` notebook裡回測數字最亮眼但也最不確定，建議之後在有 GPU 的機器上多驗證幾個隨機種子的穩定性後，再決定要不要拿掉「保守設定」。
- 已經連續測試了很多種模型組合（多重比較問題），單一模型顯著贏過基準不代表一定是真訊號，需要靠這裡累積的正式環境資料持續驗證。
- 目前重訓、預測都只用 CPU（GitHub Actions 免費方案沒有 GPU）；LightGBM/XGBoost 完全沒問題，LSTM 用這組保守設定測試過（在雲端 2 核心 CPU 環境訓練一次約 1 分鐘內），實際生產環境應該也跑得動，但還沒在 GitHub Actions 上實測過完整一次排程執行。

## 版本紀錄

見 [CHANGELOG.md](./CHANGELOG.md)。
