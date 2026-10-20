import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tradebot.backtest import simulate, walk_forward_probs
from tradebot.bot import LiveBot
from tradebot.config import DEFAULTS, deep_merge
from tradebot.data import synthetic
from tradebot.evolve import evolve, mutate
from tradebot.features import build_features, make_dataset, triple_barrier_labels
from tradebot.model import Learner
from tradebot.risk import TradeGate, compute_lots


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(DEFAULTS)
    c["paths"] = {k: str(tmp_path / k) for k in DEFAULTS["paths"]}
    c["learning"] = deep_merge(c["learning"], {"initial_train_bars": 1500, "train_window_bars": 2000,
                                               "retrain_every_bars": 300, "validation_bars": 300, "gbm_max_iter": 30})
    return c


@pytest.fixture(scope="module")
def prices():
    return synthetic(3000, seed=2)


def test_features_do_not_look_ahead(prices):
    full = build_features(prices)
    cut = build_features(prices.iloc[:2000])
    pd.testing.assert_frame_equal(full.iloc[1500:2000], cut.iloc[1500:2000])


def test_triple_barrier_labels():
    idx = pd.date_range("2024-01-01", periods=6, freq="15min")
    df = pd.DataFrame({"close": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "high": [1.0, 1.2, 1.0, 1.0, 1.0, 1.0],
                       "low": [1.0, 1.0, 0.8, 1.0, 1.0, 1.0]}, index=idx)
    labels = triple_barrier_labels(df, pd.Series(0.1, index=idx), horizon=2, mult=1.0)
    assert labels.iloc[0] == 1.0  # high hit first
    assert labels.iloc[1] == 0.0  # low hit first
    assert labels.iloc[-2:].isna().all()  # not yet known


def test_learner_learns_and_round_trips(tmp_path, cfg):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((2000, 4)), columns=list("abcd"))
    y = pd.Series((X["a"] + 0.3 * rng.standard_normal(2000) > 0).astype(float))
    learner = Learner(cfg["learning"], cfg["strategy"], X.columns).fit(X.iloc[:1500], y.iloc[:1500])
    learner.observe(X.iloc[1500:1800], y.iloc[1500:1800])
    acc = ((learner.predict_proba(X.iloc[1800:]) > 0.5) == y.iloc[1800:]).mean()
    assert acc > 0.8
    learner.save(tmp_path / "m.joblib")
    assert np.allclose(Learner.load(tmp_path / "m.joblib").predict_proba(X.iloc[:5]), learner.predict_proba(X.iloc[:5]))


def test_compute_lots():
    # 20 pips stop on EURUSD, $10 per pip per lot -> $200 per lot; $100 risk -> 0.5 lots
    assert compute_lots(100, 0.0020, 0.00001, 1.0, 0.01, 100, 0.01) == pytest.approx(0.5)
    assert compute_lots(1, 0.0020, 0.00001, 1.0, 0.01, 100, 0.01) == 0.0


def test_gate_rules(cfg):
    cfg["user"]["bias"] = {"EURUSD": "long_only"}
    cfg["risk"]["blackout"] = [{"start": "2026-11-06 13:00", "end": "2026-11-06 14:30", "note": "NFP"}]
    gate = TradeGate(cfg)
    assert gate.check("EURUSD", "2026-11-05 10:00", 1, 0.7)[0]
    assert gate.check("EURUSD", "2026-11-05 10:00", -1, 0.3) == (False, "bias long_only")
    assert gate.check("EURUSD", "2026-11-06 13:30", 1, 0.7) == (False, "blackout NFP")
    assert not gate.check("EURUSD", "2026-11-07 10:00", 1, 0.7)[0]  # Saturday


def test_simulate_stop_loss_and_take_profit(cfg):
    idx = pd.date_range("2024-01-01", periods=4, freq="15min")
    df = pd.DataFrame({"open": [1.0, 1.0, 1.0, 1.0], "high": [1.0, 1.0, 1.03, 1.0],
                       "low": [1.0, 1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    cfg["backtest"].update(spread=0.0, slippage=0.0)
    genome = dict(cfg["strategy"], sl_atr_mult=1.0, tp_atr_mult=2.0, entry_threshold=0.6)
    probs = np.array([0.9, np.nan, np.nan, np.nan])
    trades, curve = simulate(df, pd.DataFrame(index=idx), probs, pd.Series(0.01, index=idx), genome, cfg, "X")
    assert len(trades) == 1 and trades.iloc[0]["r"] == pytest.approx(2.0)
    assert curve.iloc[-1] == pytest.approx(10000 * (1 + 2 * 0.005))


def test_walk_forward_runs(cfg, prices):
    feats = build_features(prices)
    X, y, w = make_dataset(prices, feats, cfg["strategy"])
    probs, learner = walk_forward_probs(X, y, w, cfg, cfg["strategy"])
    assert np.isnan(probs[:1500]).all() and np.isfinite(probs[1500:]).all()
    assert learner.version >= 1


def test_hints_override_labels(cfg, prices):
    feats = build_features(prices)
    hints = pd.DataFrame({"time": [prices.index[1000]], "direction": ["up"], "weight": [9.0]})
    _, y, w = make_dataset(prices, feats, cfg["strategy"], hints)
    assert y.iloc[1000] == 1.0 and w.iloc[1000] == 9.0


def test_mutate_respects_bounds(cfg):
    rng = np.random.default_rng(0)
    bounds = {k: tuple(v) for k, v in cfg["evolution"]["bounds"].items()}
    for _ in range(50):
        g = mutate(cfg["strategy"], bounds, 1.0, rng)
        for key, (lo, hi) in bounds.items():
            assert lo <= g[key] <= hi
        assert isinstance(g["horizon"], int)


def test_evolve_small(cfg, prices):
    cfg["evolution"].update(population=3, generations=1, elite=1, n_jobs=1, holdout_frac=0.3)
    feats = build_features(prices)
    result = evolve(prices, feats, cfg, dict(cfg["strategy"], version=0), "X", log=lambda *_: None)
    assert set(result) >= {"best", "adopted", "holdout_best"}


class FakeBroker:
    """Plays bars of a DataFrame one at a time like MT5 would."""

    def __init__(self, df, real=False):
        self.df, self.real, self.cursor, self.orders = df, real, 2000, []

    def connect(self):
        return SimpleNamespace(login=1, server="fake", equity=10000)

    def shutdown(self):
        pass

    def is_real_account(self):
        return self.real

    def closed_bars(self, symbol, timeframe, count):
        return self.df.iloc[max(0, self.cursor - count): self.cursor]

    def server_time(self, symbol):
        return int(self.df.index[self.cursor - 1].timestamp())

    def positions(self, symbol=None):
        return []

    def spread_points(self, symbol):
        return 10

    def equity(self):
        return 10000.0

    def lots_for_risk(self, symbol, risk_money, sl_distance):
        return 0.1

    def market_order(self, *args, **kwargs):
        self.orders.append(args)


def test_live_bot_learns_bar_by_bar(cfg, prices):
    cfg["mode"] = "trade"
    cfg["strategy"]["entry_threshold"] = 0.5
    broker = FakeBroker(prices)
    bot = LiveBot(cfg, broker, log=lambda *_: None)
    for _ in range(30):
        bot.step("EURUSD")
        broker.cursor += 1
    learner = bot.symbols["EURUSD"]["learner"]
    assert learner.ready and len(broker.orders) > 0
    assert bot.symbols["EURUSD"]["last_observed"] == prices.index[2028 - cfg["strategy"]["horizon"]]


def test_live_bot_refuses_real_account(cfg, prices):
    from tradebot.broker import BrokerError
    cfg["mode"] = "trade"
    with pytest.raises(BrokerError):
        LiveBot(cfg, FakeBroker(prices, real=True), log=lambda *_: None).run(once=True)
