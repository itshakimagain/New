"""The live loop: on each newly closed bar it learns from outcomes that just became known,
predicts the next move, and (in trade mode) places orders with stop-loss and take-profit."""
import csv
import datetime as dt
import time
from pathlib import Path

import joblib
import numpy as np

from .backtest import check_exit
from .broker import BAR_SECONDS, BrokerError, MT5Broker
from .config import load_genome, state_dir
from .features import build_features, load_hints, load_hook, make_dataset
from .model import Learner
from .risk import TradeGate, signal_direction

JOURNAL_FIELDS = ["logged_at", "bar_time", "symbol", "action", "direction", "prob", "volume",
                  "sl_distance", "tp_distance", "model_version", "genome_version", "note"]
PAPER_FIELDS = ["logged_at", "symbol", "signal_bar", "direction", "prob", "volume", "entry_time", "entry",
                "exit_time", "exit", "reason", "r", "pnl"]


class Journal:
    def __init__(self, path, fields=JOURNAL_FIELDS):
        self.path, self.fields = Path(path), fields

    def write(self, **row):
        new = not self.path.exists()
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            if new:
                writer.writeheader()
            writer.writerow({"logged_at": dt.datetime.now().isoformat(timespec="seconds"), **row})


class LiveBot:
    def __init__(self, cfg, broker=None, log=print):
        self.cfg, self.log = cfg, log
        self.broker = broker or MT5Broker(cfg)
        self.gate = TradeGate(cfg)
        self.custom_features = load_hook(cfg["paths"]["custom_features"], "add_features")
        self.journal = Journal(state_dir(cfg) / "journal.csv")
        # In dry_run, every signal becomes a pretend trade that is followed bar by bar with the
        # backtest's exit rules, so results can be judged without placing orders.
        self.paper_log = Journal(state_dir(cfg) / "paper_trades.csv", PAPER_FIELDS)
        self.models_dir = state_dir(cfg) / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.symbols = {}
        self.day, self.day_start_equity = None, None

    def _dataset(self, symbol, df, genome):
        feats = build_features(df, self.custom_features)
        hints = load_hints(self.cfg["paths"]["hints_file"], symbol)
        X, y, w = make_dataset(df, feats, genome, hints, self.cfg["user"]["hint_weight"])
        return feats, X, y, w

    def _bootstrap(self, symbol, df):
        genome = load_genome(self.cfg, symbol)
        path = self.models_dir / f"{symbol}.joblib"
        if path.exists():
            saved = joblib.load(path)
            if saved.get("timeframe") != self.cfg["timeframe"]:
                self.log(f"{symbol}: saved model is for timeframe {saved.get('timeframe', 'unknown')}, "
                         f"retraining for {self.cfg['timeframe']}")
            elif saved["genome_version"] != genome.get("version", 0):
                self.log(f"{symbol}: genome changed to v{genome.get('version', 0)}, retraining")
            else:
                self.log(f"{symbol}: resumed model v{saved['learner'].version}")
                return {"paper_open": []} | saved | {"genome": genome}
        _, X, y, w = self._dataset(symbol, df, genome)
        learner = Learner(self.cfg["learning"], genome, X.columns).fit(X, y, w)
        known = y.dropna()
        self.log(f"{symbol}: trained model v{learner.version} on {len(known)} labelled bars")
        return {"learner": learner, "genome": genome, "genome_version": genome.get("version", 0),
                "last_observed": known.index[-1] if len(known) else None, "last_bar": None, "paper_open": []}

    def _save(self, symbol):
        state = self.symbols[symbol]
        saved = {k: state[k] for k in ("learner", "genome_version", "last_observed", "last_bar", "paper_open")}
        joblib.dump(saved | {"timeframe": self.cfg["timeframe"]}, self.models_dir / f"{symbol}.joblib")

    def _check_account(self):
        if self.cfg["mode"] == "trade" and self.broker.is_real_account() and not self.cfg["allow_real_money"]:
            raise BrokerError("Logged in to a REAL account but allow_real_money is false. Use a demo account, "
                              "or set allow_real_money: true in config.yaml once you accept the risk.")

    def run(self, once=False):
        info = self.broker.connect()
        self.log(f"connected: account {info.login} on {info.server}, equity {info.equity}, mode {self.cfg['mode']}")
        try:
            self._check_account()
            while True:
                failed = False
                for symbol in self.cfg["symbols"]:
                    try:
                        self.step(symbol)
                    except BrokerError as exc:
                        self.log(f"{symbol}: {exc}")
                    except Exception as exc:  # e.g. MT5 returns None after losing its connection
                        failed = True
                        self.log(f"{symbol}: error {exc!r} - will retry")
                if failed:
                    self._reconnect()
                if once:
                    return
                time.sleep(self.cfg["poll_seconds"])
        finally:
            self.broker.shutdown()

    def _reconnect(self):
        try:
            self.broker.shutdown()
            info = self.broker.connect()
        except BrokerError as exc:
            self.log(f"reconnect failed ({exc}) - retrying in {self.cfg['poll_seconds']}s")
            return
        self.log(f"reconnected: account {info.login}, equity {info.equity}")
        self._check_account()  # the terminal may now be logged in to a different account

    def step(self, symbol):
        cfg, tf = self.cfg, self.cfg["timeframe"]
        df = self.broker.closed_bars(symbol, tf, cfg["history_bars"])
        if symbol not in self.symbols:
            self.symbols[symbol] = self._bootstrap(symbol, df)
        state = self.symbols[symbol]
        if state["last_bar"] == df.index[-1]:
            return
        state["last_bar"] = df.index[-1]
        learner, genome = state["learner"], state["genome"]

        feats, X, y, w = self._dataset(symbol, df, genome)
        known = y.dropna()
        if state["last_observed"] is not None:
            known = known[known.index > state["last_observed"]]
        if len(known):
            n_events = len(learner.events)
            learner.observe(X.loc[known.index], known, w.loc[known.index])
            state["last_observed"] = known.index[-1]
            for event in learner.events[n_events:]:
                self.log(f"{symbol}: retrain ({event['reason']}) promoted={event['promoted']} -> model v{event['version']}")

        if cfg["mode"] == "trade":
            self._manage_open_positions(symbol, genome)
        else:
            self._update_paper_trades(symbol, df)
        prob = float(learner.predict_proba(X.iloc[[-1]])[0])
        direction = signal_direction(prob, float(genome["entry_threshold"]))
        acc = learner.recent_accuracy
        self.log(f"{symbol} {df.index[-1]} p(up)={prob:.3f} signal={direction:+d} "
                 f"recent_acc={'n/a' if acc is None else f'{acc:.3f}'} model=v{learner.version}")
        if direction:
            self._try_enter(symbol, df, feats, X, genome, learner, direction, prob)
        self._save(symbol)

    def _manage_open_positions(self, symbol, genome):
        max_age = int(genome["max_hold_bars"]) * BAR_SECONDS[self.cfg["timeframe"]]
        now = self.broker.server_time(symbol)
        for pos in self.broker.positions(symbol):
            if now - pos.time >= max_age:
                if self.cfg["mode"] == "trade":
                    self.broker.close_position(pos)
                self.journal.write(bar_time="", symbol=symbol, action="close_max_hold", direction="",
                                   prob="", volume=pos.volume, sl_distance="", tp_distance="",
                                   model_version="", genome_version="", note=f"ticket {pos.ticket}")

    def _update_paper_trades(self, symbol, df):
        state = self.symbols[symbol]
        bt = self.cfg["backtest"]
        cost = bt["spread"] + bt["slippage"]
        o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        still_open = []
        for trade in state["paper_open"]:
            if trade["signal_bar"] < df.index[0]:
                self.log(f"{symbol}: dropped paper trade from {trade['signal_bar']} (older than loaded history)")
                continue
            start = df.index.searchsorted(trade["signal_bar"], side="right")  # entry at the next bar's open
            d, hit = trade["direction"], None
            if start < len(df):
                entry = o[start]
                sl, tp = entry - d * trade["sl_dist"], entry + d * trade["tp_dist"]
                for i in range(start, len(df)):
                    hit = check_exit(d, sl, tp, o[i], h[i], l[i], c[i], i - start + 1, trade["max_hold"])
                    if hit is not None:
                        break
            if hit is None:
                still_open.append(trade)
                continue
            exit_price, reason = hit
            move = (exit_price - entry) * d - cost
            r, pnl = move / trade["sl_dist"], move * trade["value_per_unit"]
            self.paper_log.write(symbol=symbol, signal_bar=trade["signal_bar"], direction=d,
                                 prob=trade["prob"], volume=trade["volume"], entry_time=df.index[start],
                                 entry=entry, exit_time=df.index[i], exit=exit_price, reason=reason,
                                 r=round(r, 3), pnl=round(pnl, 2))
            self.journal.write(bar_time=df.index[i], symbol=symbol, action="paper_close", direction=d,
                               prob=trade["prob"], volume=trade["volume"], sl_distance="", tp_distance="",
                               model_version="", genome_version="", note=f"{reason} R={r:+.2f} ${pnl:+.2f}")
            self.log(f"{symbol}: paper trade from {trade['signal_bar']} closed ({reason}) R={r:+.2f} ${pnl:+.2f}")
        state["paper_open"] = still_open

    def _try_enter(self, symbol, df, feats, X, genome, learner, direction, prob):
        cfg, risk = self.cfg, self.cfg["risk"]
        entry_time = df.index[-1] + dt.timedelta(seconds=BAR_SECONDS[cfg["timeframe"]])
        base = dict(bar_time=df.index[-1], symbol=symbol, direction=direction, prob=round(prob, 4),
                    model_version=learner.version, genome_version=genome.get("version", 0))

        def skip(reason):
            self.journal.write(action="skip", volume="", sl_distance="", tp_distance="", note=reason, **base)
            self.log(f"{symbol}: skip ({reason})")

        allowed, reason = self.gate.check(symbol, entry_time, direction, prob, X.iloc[-1])
        if not allowed:
            return skip(reason)
        open_now = len(self.broker.positions(symbol)) if cfg["mode"] == "trade" else len(self.symbols[symbol]["paper_open"])
        if open_now >= risk["max_open_positions"]:
            return skip("max open positions")
        if self.broker.spread_points(symbol) > risk["max_spread_points"]:
            return skip("spread too wide")
        equity, today = self.broker.equity(), dt.date.today()
        if self.day != today:
            self.day, self.day_start_equity = today, equity
        if equity <= self.day_start_equity * (1 - risk["max_daily_loss_pct"] / 100):
            return skip("daily loss limit reached")

        atr_now = float(feats["_atr"].iloc[-1])
        if not np.isfinite(atr_now) or atr_now <= 0:
            return skip("no ATR")
        sl_dist, tp_dist = float(genome["sl_atr_mult"]) * atr_now, float(genome["tp_atr_mult"]) * atr_now
        volume = self.broker.lots_for_risk(symbol, equity * risk["risk_per_trade_pct"] / 100, sl_dist)
        if volume <= 0:
            return skip("minimum lot exceeds risk budget")

        if cfg["mode"] == "trade":
            self.broker.market_order(symbol, direction, volume, sl_dist, tp_dist,
                                     comment=f"tb m{learner.version} g{genome.get('version', 0)}")
            action = "open"
        else:
            action = "signal_dry_run"
            self.symbols[symbol]["paper_open"].append({
                "signal_bar": df.index[-1], "direction": direction, "prob": round(prob, 4), "volume": volume,
                "sl_dist": sl_dist, "tp_dist": tp_dist, "max_hold": int(genome["max_hold_bars"]),
                "value_per_unit": volume * self.broker.value_per_price_unit(symbol),
            })
        self.journal.write(action=action, volume=volume, sl_distance=sl_dist, tp_distance=tp_dist, note="", **base)
        self.log(f"{symbol}: {action} {'BUY' if direction > 0 else 'SELL'} {volume} lots")
