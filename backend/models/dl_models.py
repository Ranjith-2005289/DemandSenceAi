"""dl_models.py — Deep Learning time series forecasting models
RNN | LSTM | GRU | Bi-LSTM | CNN-1D | TCN | Transformer

All models:
  - Use sliding-window sequences as input
  - 80/20 time-ordered train/test split
  - Early stopping to prevent overfitting
  - Recursive multi-step forecast
  - Returns {"model_name", "rmse", "forecast"}
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
from services.metrics import calculate_metrics

warnings.filterwarnings("ignore")

# ── Try importing torch; fall back to TF if not available ────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    # CRITICAL: Prevent CPU thread contention deadlock when running 7 models concurrently
    torch.set_num_threads(1)
    
    BACKEND = "torch"
except ImportError:
    try:
        from tensorflow import keras
        BACKEND = "tf"
    except ImportError:
        BACKEND = None


# ── Shared config ─────────────────────────────────────────────────────────────
SEQ_LEN    = 30      # look-back window
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE   = 5      # early stopping patience
LR         = 1e-3
HIDDEN     = 64
N_LAYERS   = 2
DROPOUT    = 0.2


# ══════════════════════════════════════════════════════════════════════════════
#  PYTORCH BACKEND
# ══════════════════════════════════════════════════════════════════════════════

if BACKEND == "torch":

    # ── Sequence builder ──────────────────────────────────────────────────────
    def _make_sequences(data: np.ndarray, seq_len: int):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i: i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


    def _get_loaders(X, y, split=0.8):
        n       = int(len(X) * split)
        X_tr, X_te = X[:n], X[n:]
        y_tr, y_te = y[:n], y[n:]
        tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        te_ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
        return (DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=False),
                DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False),
                X_te, y_te)


    def _train_model(model, tr_loader, te_loader, device):
        opt       = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
        best_loss, patience_cnt, best_state = np.inf, 0, None

        model.to(device)
        for epoch in range(MAX_EPOCHS):
            model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                pred = model(xb).squeeze(-1)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in te_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss += criterion(model(xb).squeeze(-1), yb).item()
            val_loss /= max(1, len(te_loader))
            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss   = val_loss
                patience_cnt = 0
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    break

        if best_state:
            model.load_state_dict(best_state)
        return model


    def _recursive_forecast_dl(model, scaler, last_seq, horizon, device):
        model.eval()
        seq   = last_seq.copy()     # shape (seq_len, 1)
        preds = []
        with torch.no_grad():
            for _ in range(horizon):
                inp  = torch.from_numpy(seq.reshape(1, len(seq), 1)).float().to(device)
                yhat = model(inp).squeeze().item()
                preds.append(yhat)
                seq  = np.append(seq[1:], [[yhat]], axis=0)
        return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()


    def _common_pipeline(series: pd.Series, model_cls, model_kwargs: dict, model_name: str, forecast_horizon: int) -> dict:
        """Shared train-eval-forecast pipeline for all PyTorch models."""
        series = series.dropna().astype(float)
        seq_len = min(SEQ_LEN, len(series) // 3)
        if len(series) < seq_len + 5:
            raise ValueError(f"{model_name} needs ≥ {seq_len+5} observations, got {len(series)}.")

        # Fit the scaler on the chronological train portion only — fitting on
        # the full series would leak test-range statistics into training.
        split_idx = int(len(series) * 0.8)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(series.values[:split_idx].reshape(-1, 1))
        scaled = scaler.transform(series.values.reshape(-1, 1))

        X, y = _make_sequences(scaled, seq_len)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tr_loader, te_loader, X_te, y_te = _get_loaders(X, y)

        model = model_cls(input_size=1, hidden=HIDDEN, n_layers=N_LAYERS,
                          dropout=DROPOUT, **model_kwargs)
        t0 = time.perf_counter()
        model = _train_model(model, tr_loader, te_loader, device)
        train_time = time.perf_counter() - t0

        # Eval on test
        model.eval()
        t1 = time.perf_counter()
        with torch.no_grad():
            inp   = torch.from_numpy(X_te).float().to(device)
            pred  = model(inp).squeeze(-1).cpu().numpy()
        pred_time = time.perf_counter() - t1
        
        pred_inv = scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
        y_inv    = scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()
        floor    = 0.0 if series.min() >= 0 else None
        if floor is not None:
            pred_inv = np.clip(pred_inv, floor, None)
            
        metrics = calculate_metrics(y_inv, pred_inv)

        # Forecast
        last_seq = scaled[-seq_len:]
        fc_scaled = _recursive_forecast_dl(model, scaler, last_seq, forecast_horizon, device)
        if floor is not None:
            fc_scaled = np.clip(fc_scaled, floor, None)

        return {
            "model_name": model_name,
            **metrics,
            "train_time_sec": round(train_time, 4),
            "pred_time_sec": round(pred_time, 4),
            "forecast":   [round(float(v), 4) for v in fc_scaled],
            "seq_len":    seq_len,
            "epochs_run": MAX_EPOCHS,
            "backend":    "PyTorch",
        }


    # ── Model definitions ─────────────────────────────────────────────────────

    class _RNNModel(nn.Module):
        def __init__(self, input_size, hidden, n_layers, dropout, **_):
            super().__init__()
            self.rnn = nn.RNN(input_size, hidden, n_layers, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0)
            self.fc  = nn.Linear(hidden, 1)
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :])


    class _LSTMModel(nn.Module):
        def __init__(self, input_size, hidden, n_layers, dropout, **_):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, n_layers, batch_first=True,
                                dropout=dropout if n_layers > 1 else 0)
            self.fc   = nn.Linear(hidden, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])


    class _GRUModel(nn.Module):
        def __init__(self, input_size, hidden, n_layers, dropout, **_):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden, n_layers, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0)
            self.fc  = nn.Linear(hidden, 1)
        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])


    class _BiLSTMModel(nn.Module):
        def __init__(self, input_size, hidden, n_layers, dropout, **_):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, n_layers, batch_first=True,
                                dropout=dropout if n_layers > 1 else 0,
                                bidirectional=True)
            self.fc   = nn.Linear(hidden * 2, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])


    class _CNNModel(nn.Module):
        def __init__(self, input_size, hidden, n_layers, dropout, seq_len=SEQ_LEN, **_):
            super().__init__()
            self.conv1 = nn.Conv1d(input_size, hidden, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
            self.relu  = nn.ReLU()
            self.drop  = nn.Dropout(dropout)
            self.pool  = nn.AdaptiveAvgPool1d(1)
            self.fc    = nn.Linear(hidden, 1)
        def forward(self, x):
            x = x.permute(0, 2, 1)          # (B, C, T)
            x = self.relu(self.conv1(x))
            x = self.drop(x)
            x = self.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)


    class _TCNBlock(nn.Module):
        """Single dilated causal residual block for TCN."""
        def __init__(self, in_ch, out_ch, kernel, dilation, dropout):
            super().__init__()
            pad = (kernel - 1) * dilation
            self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel, padding=pad, dilation=dilation)
            self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, dilation=dilation)
            self.relu  = nn.ReLU()
            self.drop  = nn.Dropout(dropout)
            self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            self.chomp = lambda x, p: x[:, :, :-p] if p > 0 else x

        def forward(self, x):
            pad = self.conv1.padding[0]
            out = self.chomp(self.conv1(x), pad)
            out = self.relu(self.drop(out))
            out = self.chomp(self.conv2(out), pad)
            out = self.relu(self.drop(out))
            return self.relu(out + self.skip(x))


    class _TCNModel(nn.Module):
        def __init__(self, input_size, hidden, n_layers, dropout, **_):
            super().__init__()
            layers = []
            for i in range(n_layers):
                in_ch = input_size if i == 0 else hidden
                layers.append(_TCNBlock(in_ch, hidden, kernel=3, dilation=2**i, dropout=dropout))
            self.net = nn.Sequential(*layers)
            self.fc  = nn.Linear(hidden, 1)
        def forward(self, x):
            x   = x.permute(0, 2, 1)
            out = self.net(x)
            return self.fc(out[:, :, -1])


    class _TransformerModel(nn.Module):
        def __init__(self, input_size, hidden, n_layers, dropout, **_):
            super().__init__()
            self.input_proj = nn.Linear(input_size, hidden)
            encoder_layer   = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=max(1, hidden // 16),
                dim_feedforward=hidden * 4, dropout=dropout, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.fc          = nn.Linear(hidden, 1)
        def forward(self, x):
            x = self.input_proj(x)
            x = self.transformer(x)
            return self.fc(x[:, -1, :])


    # ── Public train_and_forecast functions ───────────────────────────────────

    def train_and_forecast_rnn(series: pd.Series, forecast_horizon: int) -> dict:
        return _common_pipeline(series, _RNNModel, {}, "RNN", forecast_horizon)

    def train_and_forecast_lstm(series: pd.Series, forecast_horizon: int) -> dict:
        return _common_pipeline(series, _LSTMModel, {}, "LSTM", forecast_horizon)

    def train_and_forecast_gru(series: pd.Series, forecast_horizon: int) -> dict:
        return _common_pipeline(series, _GRUModel, {}, "GRU", forecast_horizon)

    def train_and_forecast_bilstm(series: pd.Series, forecast_horizon: int) -> dict:
        return _common_pipeline(series, _BiLSTMModel, {}, "BiLSTM", forecast_horizon)

    def train_and_forecast_cnn(series: pd.Series, forecast_horizon: int) -> dict:
        return _common_pipeline(series, _CNNModel, {}, "CNN1D", forecast_horizon)

    def train_and_forecast_tcn(series: pd.Series, forecast_horizon: int) -> dict:
        return _common_pipeline(series, _TCNModel, {}, "TCN", forecast_horizon)

    def train_and_forecast_transformer(series: pd.Series, forecast_horizon: int) -> dict:
        return _common_pipeline(series, _TransformerModel, {}, "Transformer", forecast_horizon)


# ══════════════════════════════════════════════════════════════════════════════
#  TENSORFLOW / KERAS BACKEND  (fallback if torch not installed)
# ══════════════════════════════════════════════════════════════════════════════

elif BACKEND == "tf":

    def _make_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i: i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    def _fit_keras(model, X_tr, y_tr, X_te, y_te):
        cb = [
            keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        ]
        model.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
        model.fit(X_tr, y_tr, validation_data=(X_te, y_te),
                  epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
                  callbacks=cb, verbose=0)
        return model

    def _pipeline_tf(series, build_fn, model_name, forecast_horizon):
        series  = series.dropna().astype(float)
        seq_len = min(SEQ_LEN, len(series) // 3)
        if len(series) < seq_len + 5:
            raise ValueError(f"{model_name} needs ≥ {seq_len+5} obs, got {len(series)}.")

        # Fit the scaler on the chronological train portion only — fitting on
        # the full series would leak test-range statistics into training.
        split_idx = int(len(series) * 0.8)
        scaler = MinMaxScaler()
        scaler.fit(series.values[:split_idx].reshape(-1, 1))
        scaled = scaler.transform(series.values.reshape(-1, 1))
        X, y   = _make_sequences(scaled, seq_len)

        split     = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        model = build_fn(seq_len)
        t0 = time.perf_counter()
        model = _fit_keras(model, X_tr, y_tr, X_te, y_te)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        pred_inv = scaler.inverse_transform(model.predict(X_te, verbose=0))
        pred_time = time.perf_counter() - t1
        
        y_inv    = scaler.inverse_transform(y_te)
        floor    = 0.0 if series.min() >= 0 else None
        if floor is not None:
            pred_inv = np.clip(pred_inv, floor, None)
            
        metrics = calculate_metrics(y_inv, pred_inv)

        # Recursive forecast
        seq  = scaled[-seq_len:].copy()
        fc   = []
        for _ in range(forecast_horizon):
            yhat = float(model.predict(seq.reshape(1, seq_len, 1), verbose=0)[0, 0])
            fc.append(yhat)
            seq  = np.append(seq[1:], [[yhat]], axis=0)
        fc_inv = scaler.inverse_transform(np.array(fc).reshape(-1, 1)).ravel()
        if floor is not None:
            fc_inv = np.clip(fc_inv, floor, None)

        return {
            "model_name": model_name,
            **metrics,
            "train_time_sec": round(train_time, 4),
            "pred_time_sec": round(pred_time, 4),
            "forecast":   [round(float(v), 4) for v in fc_inv],
            "seq_len":    seq_len,
            "backend":    "TensorFlow",
        }

    def train_and_forecast_rnn(series, forecast_horizon):
        def build(sl):
            return keras.Sequential([
                keras.layers.SimpleRNN(HIDDEN, return_sequences=True, input_shape=(sl, 1)),
                keras.layers.Dropout(DROPOUT),
                keras.layers.SimpleRNN(HIDDEN),
                keras.layers.Dense(1),
            ])
        return _pipeline_tf(series, build, "RNN", forecast_horizon)

    def train_and_forecast_lstm(series, forecast_horizon):
        def build(sl):
            return keras.Sequential([
                keras.layers.LSTM(HIDDEN, return_sequences=True, input_shape=(sl, 1)),
                keras.layers.Dropout(DROPOUT),
                keras.layers.LSTM(HIDDEN),
                keras.layers.Dense(1),
            ])
        return _pipeline_tf(series, build, "LSTM", forecast_horizon)

    def train_and_forecast_gru(series, forecast_horizon):
        def build(sl):
            return keras.Sequential([
                keras.layers.GRU(HIDDEN, return_sequences=True, input_shape=(sl, 1)),
                keras.layers.Dropout(DROPOUT),
                keras.layers.GRU(HIDDEN),
                keras.layers.Dense(1),
            ])
        return _pipeline_tf(series, build, "GRU", forecast_horizon)

    def train_and_forecast_bilstm(series, forecast_horizon):
        def build(sl):
            return keras.Sequential([
                keras.layers.Bidirectional(
                    keras.layers.LSTM(HIDDEN, return_sequences=True), input_shape=(sl, 1)
                ),
                keras.layers.Dropout(DROPOUT),
                keras.layers.Bidirectional(keras.layers.LSTM(HIDDEN // 2)),
                keras.layers.Dense(1),
            ])
        return _pipeline_tf(series, build, "BiLSTM", forecast_horizon)

    def train_and_forecast_cnn(series, forecast_horizon):
        def build(sl):
            return keras.Sequential([
                keras.layers.Conv1D(HIDDEN, 3, activation="relu", padding="same", input_shape=(sl, 1)),
                keras.layers.Conv1D(HIDDEN, 3, activation="relu", padding="same"),
                keras.layers.GlobalAveragePooling1D(),
                keras.layers.Dropout(DROPOUT),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(1),
            ])
        return _pipeline_tf(series, build, "CNN1D", forecast_horizon)

    def train_and_forecast_tcn(series, forecast_horizon):
        """TCN via dilated causal convolutions in Keras."""
        def build(sl):
            inp = keras.Input(shape=(sl, 1))
            x   = inp
            for i in range(N_LAYERS):
                d = 2 ** i
                x = keras.layers.Conv1D(HIDDEN, 3, dilation_rate=d,
                                        padding="causal", activation="relu")(x)
                x = keras.layers.Dropout(DROPOUT)(x)
            x = keras.layers.GlobalAveragePooling1D()(x)
            out = keras.layers.Dense(1)(x)
            return keras.Model(inp, out)
        return _pipeline_tf(series, build, "TCN", forecast_horizon)

    def train_and_forecast_transformer(series, forecast_horizon):
        def build(sl):
            inp = keras.Input(shape=(sl, 1))
            x   = keras.layers.Dense(HIDDEN)(inp)
            x   = keras.layers.MultiHeadAttention(num_heads=4, key_dim=HIDDEN // 4)(x, x)
            x   = keras.layers.LayerNormalization()(x)
            x   = keras.layers.GlobalAveragePooling1D()(x)
            x   = keras.layers.Dropout(DROPOUT)(x)
            out = keras.layers.Dense(1)(x)
            return keras.Model(inp, out)
        return _pipeline_tf(series, build, "Transformer", forecast_horizon)


# ══════════════════════════════════════════════════════════════════════════════
#  NO DL BACKEND — graceful stubs
# ══════════════════════════════════════════════════════════════════════════════

else:
    def _no_backend(name):
        def _stub(series, forecast_horizon):
            raise RuntimeError(
                f"{name} requires PyTorch or TensorFlow. "
                "Install with: pip install torch  OR  pip install tensorflow"
            )
        return _stub

    train_and_forecast_rnn         = _no_backend("RNN")
    train_and_forecast_lstm        = _no_backend("LSTM")
    train_and_forecast_gru         = _no_backend("GRU")
    train_and_forecast_bilstm      = _no_backend("BiLSTM")
    train_and_forecast_cnn         = _no_backend("CNN1D")
    train_and_forecast_tcn         = _no_backend("TCN")
    train_and_forecast_transformer = _no_backend("Transformer")
