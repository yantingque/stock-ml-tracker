"""三個模型（LightGBM / XGBoost / LSTM）的訓練、存檔、讀檔邏輯。

樹模型的調參方式沿用 08_lightgbm_xgboost.ipynb：用 TimeSeriesSplit 做因果交叉驗證
（只用「當下之前」的資料，不偷看未來）。LSTM 沿用 09_lstm_walkforward.ipynb 的
保守設定（HIDDEN=24、EPOCHS=6），確保沒有 GPU 也能在合理時間內訓練完成。
"""
import json
import joblib
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from . import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tscv_n_splits(n_rows: int) -> int:
    return min(5, max(2, n_rows // 500))


# ---------------------------------------------------------------------------
# LightGBM / XGBoost
# ---------------------------------------------------------------------------

def train_tree_models(X_train, y_train):
    """回傳 (lgb_model, xgb_model)，都用 TimeSeriesSplit 因果調參選出的最佳參數訓練。

    注意：GridSearchCV 用 n_jobs=-1 平行，但底層估計器本身要設 n_jobs=1，
    不然兩層平行會互相搶執行緒，在 CPU 核心數少的機器上會嚴重拖慢速度
    （這是 08 notebook 開發過程中踩過的坑）。
    """
    n_splits = _tscv_n_splits(len(X_train))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    lgb_search = GridSearchCV(
        lgb.LGBMRegressor(n_estimators=config.TREE_N_ESTIMATORS, verbosity=-1,
                           random_state=42, n_jobs=1),
        config.TREE_PARAM_GRID, cv=tscv, scoring="r2", n_jobs=-1,
    ).fit(X_train, y_train)

    xgb_search = GridSearchCV(
        xgb.XGBRegressor(n_estimators=config.TREE_N_ESTIMATORS, random_state=42,
                          verbosity=0, n_jobs=1),
        config.TREE_PARAM_GRID, cv=tscv, scoring="r2", n_jobs=-1,
    ).fit(X_train, y_train)

    return lgb_search.best_estimator_, xgb_search.best_estimator_


def save_tree_model(model, name: str):
    joblib.dump(model, config.MODELS_DIR / f"{name}.pkl")


def load_tree_model(name: str):
    path = config.MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


# ---------------------------------------------------------------------------
# LSTM（PyTorch）
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = torch.relu(self.fc1(h_n[-1]))
        return self.fc2(out).squeeze(-1)


def train_lstm(X_train, y_train, n_features, epochs=None):
    """X_train 需已標準化（見 scale_sequences）。回傳訓練好的 LSTMModel（尚未存檔）。"""
    epochs = epochs or config.LSTM_EPOCHS
    torch.manual_seed(42)
    model = LSTMModel(n_features, config.LSTM_HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dl = DataLoader(ds, batch_size=config.LSTM_BATCH_SIZE, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return model


def scale_sequences(X_train, X_other, n_feat):
    """在訓練集上 fit StandardScaler，套用到訓練集與另一批資料（例如測試集或今天的預測資料）。

    回傳 (X_train_scaled, X_other_scaled, scaler)。
    """
    lookback = X_train.shape[1]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_feat))
    X_train_scaled = scaler.transform(X_train.reshape(-1, n_feat)).reshape(-1, lookback, n_feat).astype(np.float32)
    if len(X_other):
        X_other_scaled = scaler.transform(X_other.reshape(-1, n_feat)).reshape(-1, lookback, n_feat).astype(np.float32)
    else:
        X_other_scaled = X_other
    return X_train_scaled, X_other_scaled, scaler


def save_lstm_model(model, scaler):
    torch.save(model.state_dict(), config.MODELS_DIR / "lstm.pt")
    joblib.dump(scaler, config.MODELS_DIR / "lstm_scaler.pkl")
    meta = {"n_features": len(config.FEATURE_COLS), "hidden": config.LSTM_HIDDEN}
    with open(config.MODELS_DIR / "lstm_meta.json", "w") as f:
        json.dump(meta, f)


def load_lstm_model():
    path = config.MODELS_DIR / "lstm.pt"
    scaler_path = config.MODELS_DIR / "lstm_scaler.pkl"
    meta_path = config.MODELS_DIR / "lstm_meta.json"
    if not (path.exists() and scaler_path.exists() and meta_path.exists()):
        return None, None
    with open(meta_path) as f:
        meta = json.load(f)
    model = LSTMModel(meta["n_features"], meta["hidden"]).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_lstm(model, scaler, X):
    """X: (n, lookback, n_feat) 未標準化。回傳長度 n 的預測陣列。"""
    if len(X) == 0:
        return np.array([])
    lookback = X.shape[1]
    n_feat = X.shape[2]
    X_scaled = scaler.transform(X.reshape(-1, n_feat)).reshape(-1, lookback, n_feat).astype(np.float32)
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_scaled).to(DEVICE)).cpu().numpy()
    return pred
