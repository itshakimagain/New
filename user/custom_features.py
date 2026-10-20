"""Your own indicators. Return a DataFrame of extra columns with the same index as `df`.

They are added as the "custom" feature group and the model learns how much to trust them.
Only use the current and past bars (e.g. rolling windows, shift(+n)); never shift(-n), which peeks
into the future and makes backtests lie.
"""
import pandas as pd


def add_features(df):
    out = pd.DataFrame(index=df.index)
    bar_range = df["high"] - df["low"]
    out["range_vs_avg_50"] = bar_range / bar_range.rolling(50).mean()
    out["close_vs_prev_day_high"] = df["close"] / df["high"].rolling(96).max().shift(1) - 1
    return out
