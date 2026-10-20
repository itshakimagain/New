"""Walk-forward backtesting: the model is trained only on the past and keeps learning as the test
moves forward through time, exactly as it would live."""
import numpy as np
import pandas as pd

from .features import make_dataset
from .model import Learner
from .risk import TradeGate, signal_direction


def walk_forward_probs(X, y, w, cfg, genome):
    learning = cfg["learning"]
    horizon = int(genome["horizon"])
    start = int(learning["initial_train_bars"])
    n = len(X)
    if n < start + 2 * horizon:
        raise ValueError(f"Need more than {start + 2 * horizon} bars, got {n}")
    learner = Learner(learning, genome, X.columns)
    learner.fit(X.iloc[: start - horizon], y.iloc[: start - horizon], w.iloc[: start - horizon])
    probs = np.full(n, np.nan)
    known_until = start - horizon
    step = int(learning["step_bars"])
    for t in range(start, n, step):
        end = min(t + step, n)
        probs[t:end] = learner.predict_proba(X.iloc[t:end])
        # By the time bar `end` opens, outcomes of bars before end - horizon are known.
        new_until = end - horizon
        if new_until > known_until:
            sl = slice(known_until, new_until)
            learner.observe(X.iloc[sl], y.iloc[sl], w.iloc[sl])
            known_until = new_until
    return probs, learner


def simulate(df, X, probs, atr_series, genome, cfg, symbol, gate=None):
    bt, risk = cfg["backtest"], cfg["risk"]
    gate = gate or TradeGate(cfg)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr_v = atr_series.to_numpy(float)
    times = df.index
    n = len(df)
    cost = bt["spread"] + bt["slippage"]
    threshold = float(genome["entry_threshold"])
    max_hold = int(genome["max_hold_bars"])
    risk_frac = risk["risk_per_trade_pct"] / 100
    max_daily_loss = risk["max_daily_loss_pct"] / 100

    equity = float(bt["initial_equity"])
    curve = np.empty(n)
    trades, pos = [], None
    day, day_start = None, equity
    for i in range(n):
        if pos is not None:
            d = pos["dir"]
            exit_price = None
            if d * (o[i] - pos["sl"]) <= 0 and i > pos["entry_i"]:
                exit_price = o[i]  # gapped through the stop
            elif (d > 0 and l[i] <= pos["sl"]) or (d < 0 and h[i] >= pos["sl"]):
                exit_price = pos["sl"]  # stop first if both touched in one bar (pessimistic)
            elif (d > 0 and h[i] >= pos["tp"]) or (d < 0 and l[i] <= pos["tp"]):
                exit_price = pos["tp"]
            elif i - pos["entry_i"] + 1 >= max_hold:
                exit_price = c[i]
            if exit_price is not None:
                pnl = ((exit_price - pos["entry"]) * d - cost) * pos["size"]
                equity += pnl
                trades.append({
                    "entry_time": times[pos["entry_i"]], "exit_time": times[i], "direction": d,
                    "entry": pos["entry"], "exit": exit_price, "prob": pos["prob"], "pnl": pnl,
                    "r": pnl / pos["risk_money"],
                })
                pos = None
        curve[i] = equity
        if times[i].date() != day:
            day, day_start = times[i].date(), equity

        if pos is None and i + 1 < n and np.isfinite(probs[i]) and np.isfinite(atr_v[i]) and atr_v[i] > 0:
            d = signal_direction(probs[i], threshold)
            if d == 0 or equity <= day_start * (1 - max_daily_loss):
                continue
            allowed, _ = gate.check(symbol, times[i + 1], d, probs[i], X.iloc[i])
            if not allowed:
                continue
            sl_dist = float(genome["sl_atr_mult"]) * atr_v[i]
            risk_money = equity * risk_frac
            entry = o[i + 1]
            pos = {
                "dir": d, "entry": entry, "entry_i": i + 1, "prob": probs[i],
                "sl": entry - d * sl_dist, "tp": entry + d * float(genome["tp_atr_mult"]) * atr_v[i],
                "size": risk_money / sl_dist, "risk_money": risk_money,
            }
    return pd.DataFrame(trades), pd.Series(curve, index=times)


def metrics(trades, curve, score_from=None):
    if score_from is not None:
        trades = trades[trades["entry_time"] >= score_from] if len(trades) else trades
        curve = curve[curve.index >= score_from]
    daily = curve.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 2 and rets.std() > 0 else 0.0
    drawdown = float((1 - curve / curve.cummax()).max() * 100) if len(curve) else 0.0
    out = {
        "trades": int(len(trades)),
        "return_pct": float((curve.iloc[-1] / curve.iloc[0] - 1) * 100) if len(curve) else 0.0,
        "sharpe": sharpe,
        "max_drawdown_pct": drawdown,
        "win_rate_pct": 0.0, "profit_factor": 0.0, "avg_r": 0.0,
    }
    if len(trades):
        wins, losses = trades.loc[trades.pnl > 0, "pnl"].sum(), -trades.loc[trades.pnl < 0, "pnl"].sum()
        out["win_rate_pct"] = float((trades.pnl > 0).mean() * 100)
        out["profit_factor"] = float(wins / losses) if losses > 0 else float("inf")
        out["avg_r"] = float(trades.r.mean())
    return out


def run_backtest(df, feats, cfg, genome, symbol, hints=None, score_from=None):
    X, y, w = make_dataset(df, feats, genome, hints, cfg["user"]["hint_weight"])
    probs, learner = walk_forward_probs(X, y, w, cfg, genome)
    trades, curve = simulate(df, X, probs, feats["_atr"], genome, cfg, symbol)
    first_scored = df.index[int(cfg["learning"]["initial_train_bars"])]
    score_from = max(score_from, first_scored) if score_from is not None else first_scored
    return {"metrics": metrics(trades, curve, score_from), "trades": trades, "equity": curve, "learner": learner}
