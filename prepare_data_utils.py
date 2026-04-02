"""
prepare_data_utils.py
共享模块：被 1_prepare_data.py 和 2_rl_environment.py 同时 import
"""
import numpy as np
import pandas as pd
import torch.nn as nn

FEATURE_COLS = [
    "load_lag1",
    "load_lag24",
    "load_lag168",
    "hour_of_day",
    "day_of_week",
]
TARGET_COL = "load"
SEQ_LEN    = 24


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    df["load_lag1"]   = df["load"].shift(1)
    df["load_lag24"]  = df["load"].shift(24)
    df["load_lag168"] = df["load"].shift(168)
    return df.dropna()


class LSTMForecaster(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)
