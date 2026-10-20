"""Rules every trade must pass, shared by the backtester and the live bot."""
import math

import pandas as pd

from .features import load_hook


class TradeGate:
    def __init__(self, cfg):
        risk = cfg["risk"]
        self.allowed_hours = set(risk["allowed_hours"]) if risk.get("allowed_hours") else None
        self.allowed_weekdays = set(risk.get("allowed_weekdays") or range(7))
        self.blackout = [
            (pd.Timestamp(b["start"]), pd.Timestamp(b["end"]), b.get("note", "")) for b in risk.get("blackout") or []
        ]
        self.bias = {k.upper(): v for k, v in (cfg["user"].get("bias") or {}).items()}
        self.user_filter = load_hook(cfg["paths"].get("custom_filters"), "allow_trade")

    def check(self, symbol, when, direction, prob, features=None):
        """Return (allowed, reason). `when` is the entry time in the data's (broker's) timezone."""
        when = pd.Timestamp(when)
        if when.dayofweek not in self.allowed_weekdays:
            return False, "weekday not allowed"
        if self.allowed_hours is not None and when.hour not in self.allowed_hours:
            return False, "hour not allowed"
        for start, end, note in self.blackout:
            if start <= when <= end:
                return False, f"blackout {note}".strip()
        bias = self.bias.get(symbol.upper(), "both")
        if bias == "off":
            return False, "symbol switched off"
        if bias == "long_only" and direction < 0:
            return False, "bias long_only"
        if bias == "short_only" and direction > 0:
            return False, "bias short_only"
        if self.user_filter is not None and not self.user_filter(symbol, when, direction, prob, features):
            return False, "user filter"
        return True, "ok"


def signal_direction(prob, threshold):
    if prob >= threshold:
        return 1
    if prob <= 1 - threshold:
        return -1
    return 0


def compute_lots(risk_money, sl_distance, tick_size, tick_value, vol_min, vol_max, vol_step):
    """Lots such that hitting the stop loses about `risk_money`. Returns 0 if even the minimum lot is too risky."""
    if sl_distance <= 0 or tick_size <= 0 or tick_value <= 0:
        return 0.0
    loss_per_lot = sl_distance / tick_size * tick_value
    lots = math.floor(risk_money / loss_per_lot / vol_step) * vol_step
    if lots < vol_min:
        return 0.0
    return round(min(lots, vol_max), 8)
