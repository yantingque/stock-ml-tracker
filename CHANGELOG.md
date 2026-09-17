# Changelog

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
