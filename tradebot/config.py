"""Configuration loading. Everything the bot does can be overridden in config.yaml."""
import copy
import json
import os
from pathlib import Path

import yaml

ALL_FEATURE_GROUPS = ["momentum", "volatility", "trend", "oscillators", "time", "volume", "candles", "custom"]

DEFAULTS = {
    # dry_run: read live prices, learn and log signals, but never send orders.
    # trade:   send orders to the logged-in MT5 account.
    "mode": "dry_run",
    # Even in "trade" mode the bot refuses to trade a REAL (non-demo) account unless this is true.
    "allow_real_money": False,
    "mt5": {"login": None, "password": None, "server": None, "path": None, "magic": 20261020, "deviation": 20},
    "symbols": ["EURUSD"],
    "timeframe": "M15",
    "history_bars": 20000,
    "poll_seconds": 10,
    "paths": {
        "state_dir": "state",
        "data_dir": "data",
        "hints_file": "user/hints.csv",
        "custom_features": "user/custom_features.py",
        "custom_filters": "user/custom_filters.py",
    },
    # The starting "genome". `evolve` searches around it and saves improved versions per symbol.
    "strategy": {
        "entry_threshold": 0.58,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.0,
        "label_atr_mult": 1.5,
        "horizon": 16,
        "max_hold_bars": 32,
        "feature_groups": list(ALL_FEATURE_GROUPS),
        "gbm_max_depth": 4,
        "gbm_learning_rate": 0.05,
        "recency_half_life": 3000,
    },
    "learning": {
        "initial_train_bars": 4000,
        "train_window_bars": 6000,
        "retrain_every_bars": 500,
        "step_bars": 24,
        "validation_bars": 800,
        "promote_min_improvement": 0.0,
        "drift_window": 300,
        "drift_threshold": 0.48,
        "gbm_max_iter": 150,
        "online_alpha": 0.0001,
        "seed": 7,
    },
    "risk": {
        "risk_per_trade_pct": 0.5,
        "max_daily_loss_pct": 3.0,
        "max_open_positions": 1,
        "max_spread_points": 30,
        "allowed_hours": None,
        "allowed_weekdays": [0, 1, 2, 3, 4],
        "blackout": [],
    },
    "user": {"bias": {}, "hint_weight": 5.0},
    "backtest": {"initial_equity": 10000.0, "spread": 0.00012, "slippage": 0.00002},
    "evolution": {
        "population": 12,
        "generations": 6,
        "elite": 3,
        "mutation_rate": 0.3,
        "holdout_frac": 0.25,
        "min_trades": 30,
        "min_holdout_improvement": 0.0,
        "n_jobs": -1,
        "seed": 11,
        "bounds": {
            "entry_threshold": [0.52, 0.70],
            "sl_atr_mult": [0.8, 3.0],
            "tp_atr_mult": [0.8, 5.0],
            "label_atr_mult": [0.5, 3.0],
            "horizon": [4, 48],
            "max_hold_bars": [4, 96],
            "gbm_max_depth": [2, 8],
            "gbm_learning_rate": [0.02, 0.3],
            "recency_half_life": [500, 10000],
        },
    },
}


def deep_merge(base, override):
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path="config.yaml"):
    cfg = copy.deepcopy(DEFAULTS)
    if path and Path(path).exists():
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    cfg["mt5"]["password"] = os.environ.get("MT5_PASSWORD", cfg["mt5"].get("password"))
    return cfg


def state_dir(cfg):
    path = Path(cfg["paths"]["state_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def genome_path(cfg, symbol):
    return state_dir(cfg) / f"genome_{symbol}.json"


def load_genome(cfg, symbol):
    """The evolved genome for a symbol if one was saved, else the strategy section of the config."""
    path = genome_path(cfg, symbol)
    if path.exists():
        genome = json.loads(path.read_text())
    else:
        genome = copy.deepcopy(cfg["strategy"])
        genome["version"] = 0
    return genome


def save_genome(cfg, symbol, genome):
    genome_path(cfg, symbol).write_text(json.dumps(genome, indent=2))
