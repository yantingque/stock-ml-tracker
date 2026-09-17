# Changelog

## v0.3.0 - 2026-09-17

修正第一次正式排程執行時發現的問題：`stock-system` 自動發佈到 Release 的
`stock_data.db` 只有原始表，不包含 `feature_panel`（那張表在 `ml_lab` 專案裡
是靠手動執行 `01_feature_engineering.ipynb` 才產生的），導致 `train.py` 讀
`feature_panel` 時噴 `no such table: feature_panel`。

- 新增 `src/feature_engineering.py`，把 `01_feature_engineering.ipynb` 的特徵
  計算邏輯整支搬過來，`run_daily.py` 在 fetch 之後、train 之前會自動重新計算
  一次 `feature_panel`，不再依賴人工介入
- 已在雲端用「移除 feature_panel 表、模擬 stock-system 自動發佈的原始 db」的
  方式驗證過：重新計算耗時約 3 秒、產出列數與欄位跟原本手動跑 01 的結果一致，
  接續的 train/predict/track 全部正常

## v0.2.0 - 2026-09-17

新增 LINE 通知：`src/notify.py` 把三個模型當天的 Top10 整理成文字訊息，用 LINE
Messaging API 的 broadcast 端點發送。刻意用**另一個獨立、新建的 LINE 官方帳號**接收，
不會跟現有規則系統的正式通知混在一起。`track.py` 完成寫入/回填後會自動呼叫；沒有設定
`LINE_CHANNEL_ACCESS_TOKEN` 這個 Secret 也不會讓整條流程失敗，只會印一行提醒後跳過。

- 新增 `src/notify.py`（含 `python -m src.notify` 本機預覽/測試發送用的 entry point）
- `daily_track.yml` 新增 `LINE_CHANNEL_ACCESS_TOKEN` 環境變數
- README 新增 LINE 官方帳號 + Messaging API channel 的設定步驟
- 已在雲端用測試資料驗證訊息格式產出正確（未設定 token 時會正確跳過、不影響其餘流程）

## v0.1.0 - 2026-09-17

第一版：三模型（LightGBM、XGBoost、LSTM）正式上線每日追蹤。

- 新增 `src/` 底下完整流程：`fetch_data.py`（抓 stock-system 最新 Release 附件）、
  `data.py`（讀 feature_panel、組橫斷面/序列資料）、`models.py`（訓練/存讀三個模型）、
  `train.py`（每 20 個交易日判斷是否重訓）、`predict.py`（產生今日 Top10 預測）、
  `track.py`（寫入追蹤表 + 回填實際報酬率）
- 新增 `.github/workflows/daily_track.yml`，每天排程自動執行完整流程並 commit 結果
- 邏輯已在雲端環境用測試資料庫驗證過一次完整流程（fetch 略過、train、predict、track、
  重訓門檻判斷、回填邏輯皆正確），尚未在 GitHub Actions 正式環境跑過第一次
- 三個模型的超參數、訓練設定沿用 `ml_lab` repo 的 `08_lightgbm_xgboost.ipynb`、
  `09_lstm_walkforward.ipynb` 驗證結果，尚未做進一步調整

### 待辦 / 下一步

- 在 GitHub 上設定 `STOCK_SYSTEM_TOKEN` Secret，跑第一次正式排程確認全部串接正常
- 累積足夠天數的追蹤資料後，寫一支比較/報表腳本（或 notebook）定期拉 `tracking.db`
  出來做統計檢定，並跟規則系統的 `selection_log` 做對照
- LSTM 之後考慮在有 GPU 的環境多驗證幾個隨機種子的穩定性
