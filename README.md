# tradebot: a self-learning MetaTrader 5 robot

A Python trading robot for MetaTrader 5 that:

* **studies the market on its own.** It computes about 35 indicators (momentum, volatility, trend,
  oscillators, time of day, volume, candle shape) and learns which combinations come before a move up
  or down.
* **adapts as the market changes.** Every new closed bar teaches it something: an online model updates
  right away, a gradient-boosted model is retrained on a rolling window, and the two are blended
  according to which has been more accurate lately. If accuracy drops (the market regime changed),
  it retrains early. A new model only replaces the old one if it does better on the most recent data.
* **develops its own program.** `evolve` runs a genetic search over the bot's own strategy: entry
  threshold, stop and target sizes, what it tries to predict, which indicator groups it uses, and model
  settings. New settings are adopted only if they also win on a recent period the search never saw.
* **takes your input.** You can change any setting, add your own indicators, veto trades with your
  own rules, and teach it with hints (see [Your input](#your-input)).

> **Read this first.** No software can promise profits, and most automated strategies lose money once
> spreads and commissions are paid. This bot is a framework for learning and testing, not a money
> machine. Run it on a **demo account** for weeks before you even consider real money. By default it
> runs in `dry_run` mode and refuses to trade a real account. Never risk money you can't afford to lose.

## Setup (Windows, where MetaTrader 5 runs)

1. Install MetaTrader 5 from your broker, log in (to a demo account to start), and enable
   *Tools → Options → Expert Advisors → Allow algorithmic trading*.
2. Install Python 3.10+ from python.org, then in this folder run:
   ```
   pip install -r requirements.txt
   ```
3. Edit `config.yaml` (symbols, timeframe, risk). Keep the terminal open while the bot runs.

## Daily workflow

```
python -m tradebot fetch                  # download history from MT5 into data/
python -m tradebot backtest               # honest walk-forward test: trained only on the past, learns as it goes
python -m tradebot evolve                 # let it search for better settings (takes a while)
python -m tradebot run                    # live, dry_run: learns and logs signals, places no orders
python -m tradebot run --mode trade       # places orders on the logged-in (demo) account
python -m tradebot status                 # current settings, model version, accuracy, recent journal
```

Add `--symbol GBPUSD` to any command for another symbol. Without MT5 (on Mac or Linux) you can still
try everything except `fetch` and `run`: `python -m tradebot demo-data` writes synthetic prices.

Every decision is written to `state/journal.csv`: the probability, the model and strategy versions,
and why a trade was skipped. Backtest trades and equity curves are saved in `state/`.

## Your input

| What you want | How |
|---|---|
| Change risk, timeframe, stops, thresholds | `config.yaml` |
| Only long / only short / pause a symbol | `user: bias: {EURUSD: long_only}` |
| Avoid news or certain hours | `risk: blackout` and `risk: allowed_hours` |
| Add your own indicator | write it in `user/custom_features.py`. The model decides how much to trust it. |
| Your own trade rules | `user/custom_filters.py`: return `False` to block a trade (used in backtests too) |
| Teach it from your own reading of the chart | `python -m tradebot hint --time "2026-10-19 14:15" --direction up --note "breakout"` |

Hints are extra training examples that count `hint_weight` times more than a normal bar. Hints are
hindsight, so backtests over hinted periods will look better than real trading would.

## How it works

```
MT5 bars ─► features.py (indicators + your custom ones)
          ─► labels: "did price move N×ATR up before N×ATR down within H bars?"
          ─► model.py  Learner = online linear model + gradient-boosted trees,
                       blended by recent accuracy, retrained on schedule or on drift,
                       champion/challenger so a worse model never replaces a better one
          ─► risk.py   your rules, bias, hours, blackouts, daily loss limit, position sizing
          ─► broker.py market orders with stop-loss and take-profit on MT5
evolve.py  genetic search over the strategy, adopted only if it wins on hidden recent data
```

Backtests are pessimistic on purpose: entries happen at the next bar's open, spread and slippage are
charged on every trade, and if the stop and the target are both touched in one bar the stop counts.
Position size is set so a stop-out loses `risk_per_trade_pct` of equity. The backtest assumes the
account is in the quote currency; live trading uses MT5's tick value, so lot sizing is exact.

## Tests

```
pip install pytest
python -m pytest tests
```
