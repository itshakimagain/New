"""Indicators, labels and training-set construction.

Every feature at bar t uses only bars <= t, so a model never sees the future.
Columns are named "<group>__<name>" so a genome can switch whole groups on or off.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

WARMUP_BARS = 210  # longest lookback (SMA 200 + slope) plus a margin


def load_hook(path, name):
    """Load function `name` from a user Python file, or None if the file does not exist."""
    if not path or not Path(path).exists():
        return None
    spec = importlib.util.spec_from_file_location(f"user_hook_{Path(path).stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name, None)


def atr(df, n=14):
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def build_features(df, custom_fn=None):
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    a = atr(df)
    logret = np.log(c).diff()
    f = {}

    for n in (1, 3, 6, 12, 24, 48):
        f[f"momentum__ret_{n}"] = (c - c.shift(n)) / a
    f["momentum__ret_1_lag1"] = f["momentum__ret_1"].shift(1)
    f["momentum__ret_1_lag2"] = f["momentum__ret_1"].shift(2)

    f["volatility__atr_pct"] = a / c
    f["volatility__std_12"] = logret.rolling(12).std()
    f["volatility__std_48"] = logret.rolling(48).std()
    f["volatility__std_ratio"] = f["volatility__std_12"] / f["volatility__std_48"]
    f["volatility__atr_ratio"] = a / a.rolling(200).mean()

    for n in (20, 50, 200):
        sma = c.rolling(n).mean()
        f[f"trend__dist_sma_{n}"] = (c - sma) / a
        f[f"trend__slope_sma_{n}"] = (sma - sma.shift(5)) / a
    ema_fast, ema_slow = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    f["trend__macd"] = macd / a
    f["trend__macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / a

    f["oscillators__rsi_14"] = rsi(c, 14) / 100 - 0.5
    f["oscillators__rsi_4"] = rsi(c, 4) / 100 - 0.5
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    f["oscillators__stoch_k"] = (c - lo14) / (hi14 - lo14).replace(0, np.nan) - 0.5
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["oscillators__bb_pctb"] = (c - mid) / (2 * sd).replace(0, np.nan)
    f["oscillators__donchian_pos"] = (c - l.rolling(55).min()) / (h.rolling(55).max() - l.rolling(55).min()).replace(0, np.nan) - 0.5

    hour = df.index.hour + df.index.minute / 60
    f["time__hour_sin"] = pd.Series(np.sin(2 * np.pi * hour / 24), index=df.index)
    f["time__hour_cos"] = pd.Series(np.cos(2 * np.pi * hour / 24), index=df.index)
    f["time__weekday"] = pd.Series(df.index.dayofweek / 4 - 0.5, index=df.index)

    vol = df["tick_volume"].astype(float)
    f["volume__z_50"] = (vol - vol.rolling(50).mean()) / vol.rolling(50).std().replace(0, np.nan)
    f["volume__ratio_5_50"] = vol.rolling(5).mean() / vol.rolling(50).mean().replace(0, np.nan)

    rng = (h - l).replace(0, np.nan)
    f["candles__body"] = (c - o) / rng
    f["candles__upper_wick"] = (h - np.maximum(o, c)) / rng
    f["candles__lower_wick"] = (np.minimum(o, c) - l) / rng
    f["candles__range_atr"] = (h - l) / a

    out = pd.DataFrame(f, index=df.index)
    if custom_fn is not None:
        extra = custom_fn(df)
        if extra is not None and len(extra.columns):
            extra = extra.reindex(df.index).astype(float)
            extra.columns = [f"custom__{col}" for col in extra.columns]
            out = out.join(extra)
    out = out.replace([np.inf, -np.inf], np.nan)
    # After the warm-up, a missing value only means a degenerate input (e.g. zero range or flat volume).
    out.iloc[WARMUP_BARS:] = out.iloc[WARMUP_BARS:].fillna(0.0)
    out["_atr"] = a
    return out


def triple_barrier_labels(df, atr_series, horizon, mult):
    """1 if price rises `mult`*ATR before falling `mult`*ATR within `horizon` bars, 0 if the reverse.

    NaN when neither barrier is hit, when both are hit inside the same bar (ambiguous),
    and for the last `horizon` bars whose outcome is not known yet.
    """
    c, h, l = (df[k].to_numpy(float) for k in ("close", "high", "low"))
    a = atr_series.to_numpy(float)
    n = len(c)
    upper, lower = c + mult * a, c - mult * a
    labels = np.full(n, np.nan)
    done = np.zeros(n, bool)
    for k in range(1, horizon + 1):
        hk, lk = np.full(n, np.nan), np.full(n, np.nan)
        hk[: n - k], lk[: n - k] = h[k:], l[k:]
        hit_up, hit_down = (hk >= upper) & ~done, (lk <= lower) & ~done
        labels[hit_up & ~hit_down] = 1.0
        labels[hit_down & ~hit_up] = 0.0
        done |= hit_up | hit_down
    labels[max(n - horizon, 0):] = np.nan
    return pd.Series(labels, index=df.index)


def select_columns(feats, groups):
    groups = set(groups)
    return [col for col in feats.columns if col.split("__")[0] in groups]


def load_hints(path, symbol):
    """User-supplied examples: rows of time,symbol,direction(up/down),weight,note."""
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["time", "direction", "weight"])
    hints = pd.read_csv(path, comment="#")
    if hints.empty:
        return pd.DataFrame(columns=["time", "direction", "weight"])
    hints = hints[hints["symbol"].str.upper() == symbol.upper()].copy()
    hints["time"] = pd.to_datetime(hints["time"])
    return hints


def make_dataset(df, feats, genome, hints=None, hint_weight=5.0):
    """Feature matrix X, labels y (NaN = unknown) and sample weights w, all aligned to df.index."""
    X = feats[select_columns(feats, genome["feature_groups"])]
    horizon = int(genome["horizon"])
    y = triple_barrier_labels(df, feats["_atr"], horizon, float(genome["label_atr_mult"]))
    w = pd.Series(1.0, index=df.index)
    if hints is not None and len(hints):
        mature_until = len(df) - horizon
        for _, hint in hints.iterrows():
            pos = df.index.searchsorted(hint["time"], side="right") - 1
            if 0 <= pos < mature_until:
                y.iloc[pos] = 1.0 if str(hint["direction"]).lower() in ("up", "buy", "long", "1") else 0.0
                weight = hint.get("weight")
                w.iloc[pos] = float(weight) if pd.notna(weight) else hint_weight
    return X, y, w
