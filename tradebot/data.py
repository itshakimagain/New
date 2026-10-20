"""Price data: CSV load/save, conversion of MT5 rates, and a synthetic generator for offline testing."""
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close", "tick_volume"]


def rates_to_frame(rates):
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df.set_index("time").sort_index()


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().lstrip("<").rstrip(">") for c in df.columns]
    if "time" not in df.columns and "date" in df.columns:
        df["time"] = df["date"]
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    missing = {"open", "high", "low", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "tick_volume" not in df.columns:
        df["tick_volume"] = df["volume"] if "volume" in df.columns else 0.0
    return df[PRICE_COLUMNS].astype(float)


def save_csv(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index_label="time")


def synthetic(n_bars=20000, seed=1, start="2024-01-01", freq="15min", price=1.10):
    """Regime-switching prices (trending / mean-reverting / noise) with volatility clustering.

    Only for exercising the pipeline offline - real markets are far less predictable.
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=int(n_bars * 1.5), freq=freq)
    index = index[index.dayofweek < 5][:n_bars]
    regimes = {0: 0.15, 1: -0.15, 2: 0.0}
    regime, ret, vol = 0, 0.0, 0.0006
    closes = np.empty(n_bars)
    for i in range(n_bars):
        if rng.random() < 1 / 800:
            regime = int(rng.integers(0, 3))
        vol = 0.97 * vol + 0.03 * 0.0006 * np.exp(rng.normal(0, 0.5))
        ret = regimes[regime] * ret + rng.normal(0, vol)
        price *= np.exp(ret)
        closes[i] = price
    opens = np.concatenate([[closes[0]], closes[:-1]])
    wick = np.abs(rng.normal(0, 0.0003, n_bars)) * closes
    highs = np.maximum(opens, closes) + wick
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 0.0003, n_bars)) * closes
    volume = rng.poisson(200, n_bars) * (1 + 1000 * np.abs(closes / opens - 1))
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "tick_volume": volume}, index=index
    ).rename_axis("time")
