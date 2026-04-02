"""
LSTM: predict load AND solar generation
════════════════════════════════════════════════════════════════
Input features (5):
    load_lag1, load_lag24, load_lag168, hour_of_day, day_of_week

Output (2):
    predicted_load(t+1)
    predicted_solar(t+1)   ← synthetic PV, generated from hour

Change only DATA_PATH below, then run this file.
"""

import numpy as np
import pandas as pd
import pickle, os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = r"D:\OneDrive - Yokogawa Electric Corporation\Desktop\Smart grid 2\household_power_consumption.csv"

# Config

FEATURE_COLS = [
    "load_lag1",
    "load_lag24",
    "load_lag168",
    "hour_of_day",
    "day_of_week",
]
TARGET_COLS = ["load", "solar"]   # dual output
SEQ_LEN     = 24
SOLAR_CAP   = 3.0   # kW peak solar capacity (matches household scale)



# Solar generation model (synthetic PV)
# Standard formula used in microgrid research

def generate_solar(hour: int, noise: bool = True) -> float:
    """
    Synthetic solar PV output (kW).
    Peaks at noon, zero at night.
    With optional Gaussian noise to simulate cloud variability.
    """
    if hour < 6 or hour > 20:
        return 0.0
    peak   = np.sin((hour - 6) / 14 * np.pi)   # 0→1→0 between 6am and 8pm
    output = max(0.0, peak) * SOLAR_CAP
    if noise:
        output = max(0.0, output + np.random.normal(0, 0.1 * SOLAR_CAP))
    return float(output)


# Electricity price model
# Clear peak/off-peak structure for RL to exploit

def price_model(hour: int) -> float:
    """
    3-tier price ($/kWh) — wide spread so PPO has clear incentive:
      Off-peak  00-05  → $0.10
      Shoulder  06-17  → $0.20
      Peak      18-23  → $0.40   ← 4x off-peak, PPO will learn to discharge here
    """
    if   hour < 6:  return 0.10
    elif hour < 18: return 0.20
    else:           return 0.40


# 1. Load & clean UCI data

def load_uci_data(filepath):
    print("Loading UCI dataset...")
    df = pd.read_csv(filepath, low_memory=False, na_values=["?"])
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], dayfirst=True, format="mixed")
    df = df[["datetime", "Global_active_power"]].copy()
    df.columns = ["datetime", "load"]
    df["load"] = pd.to_numeric(df["load"], errors="coerce")
    df = df.set_index("datetime").resample("1h").mean().dropna()
    print(f"  Total hours: {len(df)}")
    print(f"  Date range:  {df.index.min()} -> {df.index.max()}")
    return df


# 2. Feature engineering

def add_features(df):
    """Add lag features, time features, solar column, price column."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df["hour_of_day"] = df.index.hour
    df["day_of_week"]  = df.index.dayofweek
    df["load_lag1"]    = df["load"].shift(1)
    df["load_lag24"]   = df["load"].shift(24)
    df["load_lag168"]  = df["load"].shift(168)

    # Synthetic solar (deterministic, no noise) for training target
    df["solar"] = df.index.hour.map(lambda h: generate_solar(h, noise=False))

    # Price column (for environment use, not ML target)
    df["price"] = df.index.hour.map(price_model)

    return df.dropna()


# 3. Dataset

class LoadDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def build_sequences(df, X_scaler=None, y_scaler=None, fit=True):
    if fit:
        X_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        features = X_scaler.fit_transform(df[FEATURE_COLS].values)
        targets  = y_scaler.fit_transform(df[TARGET_COLS].values)
    else:
        features = X_scaler.transform(df[FEATURE_COLS].values)
        targets  = y_scaler.transform(df[TARGET_COLS].values)

    X_seq, y_seq = [], []
    for i in range(SEQ_LEN, len(features)):
        X_seq.append(features[i - SEQ_LEN : i])
        y_seq.append(targets[i])   # shape (2,): [load, solar]

    return np.array(X_seq), np.array(y_seq), X_scaler, y_scaler


# 4. LSTM model (dual output)

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=5, hidden_size=64,
                 num_layers=2, dropout=0.2, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])   # (batch, 2)



# 5. Train
def train(filepath=DATA_PATH, epochs=30, batch_size=64, lr=1e-3):

    df    = load_uci_data(filepath)
    df    = add_features(df)
    split = int(len(df) * 0.8)

    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    X_train, y_train, X_scaler, y_scaler = build_sequences(train_df, fit=True)
    X_test,  y_test,  _,        _        = build_sequences(
        test_df, X_scaler, y_scaler, fit=False)

    train_loader = DataLoader(
        LoadDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(
        LoadDataset(X_test,  y_test),  batch_size=batch_size)

    print(f"\nTrain: {len(X_train)} sequences")
    print(f"Test:  {len(X_test)}  sequences")
    print(f"Input: {len(FEATURE_COLS)} features x {SEQ_LEN} steps")
    print(f"Output: {TARGET_COLS}\n")

    model     = LSTMForecaster(input_size=len(FEATURE_COLS),
                               output_size=len(TARGET_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, load_maes, solar_maes = [], [], []
    best_mae = float("inf")

    os.makedirs("saved_models", exist_ok=True)   # ← 移到这里
    os.makedirs("results",      exist_ok=True) 
    print("Training LSTM...")
    for epoch in range(1, epochs + 1):

        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                ps.append(model(xb).numpy())
                ts.append(yb.numpy())

        preds_real = y_scaler.inverse_transform(np.concatenate(ps))
        trues_real = y_scaler.inverse_transform(np.concatenate(ts))

        load_mae  = mean_absolute_error(trues_real[:, 0], preds_real[:, 0])
        solar_mae = mean_absolute_error(trues_real[:, 1], preds_real[:, 1])
        load_maes.append(load_mae)
        solar_maes.append(solar_mae)

        if load_mae + solar_mae < best_mae:
            best_mae = load_mae + solar_mae
            torch.save(model.state_dict(), "saved_models/lstm_best.pth")

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {epoch_loss:.5f} | "
                  f"Load MAE: {load_mae:.4f} kW | Solar MAE: {solar_mae:.4f} kW")

    # save
    torch.save(model.state_dict(), "saved_models/lstm_model.pth")
    with open("saved_models/scalers.pkl", "wb") as f:
        pickle.dump({"X_scaler": X_scaler, "y_scaler": y_scaler}, f)
    test_df.to_csv("saved_models/test_data.csv")
    print("\nSaved: lstm_best.pth / scalers.pkl / test_data.csv")

    # plots
    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(train_losses, color="#2ca02c")
    axes[0].set_title("Training Loss (MSE)")
    axes[0].set_xlabel("Epoch"); axes[0].grid(alpha=0.3)

    axes[1].plot(load_maes, color="#1f77b4")
    axes[1].set_title("Load MAE (kW)")
    axes[1].set_xlabel("Epoch"); axes[1].grid(alpha=0.3)

    axes[2].plot(solar_maes, color="#e07b39")
    axes[2].set_title("Solar MAE (kW)")
    axes[2].set_xlabel("Epoch"); axes[2].grid(alpha=0.3)

    fig.suptitle("LSTM Training — Load & Solar Forecast", fontsize=12)
    fig.tight_layout()
    fig.savefig("results/lstm_training_curve.png", dpi=150, bbox_inches="tight")

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax1.plot(trues_real[:168, 0], label="Actual",    color="#1f77b4", lw=1.2)
    ax1.plot(preds_real[:168, 0], label="Predicted", color="#e07b39", lw=1.2, ls="--")
    ax1.set_ylabel("Load (kW)"); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_title("LSTM Prediction vs Actual — first 168h of test set")

    ax2.plot(trues_real[:168, 1], label="Actual",    color="#1f77b4", lw=1.2)
    ax2.plot(preds_real[:168, 1], label="Predicted", color="#e07b39", lw=1.2, ls="--")
    ax2.set_ylabel("Solar (kW)"); ax2.set_xlabel("Hour")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig2.tight_layout()
    fig2.savefig("results/lstm_prediction.png", dpi=150, bbox_inches="tight")
    print("Saved: results/lstm_training_curve.png / lstm_prediction.png")
    plt.show()
    print("\nML done! Next: python rl_environment.py")
    return model, X_scaler, y_scaler


if __name__ == "__main__":
    train()