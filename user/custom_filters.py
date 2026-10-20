"""Your veto over every trade, in backtests and live. Return False to block a trade.

symbol:    e.g. "EURUSD"
time:      pandas Timestamp of the entry, in broker server time
direction: +1 buy, -1 sell
prob:      the model's probability that price goes up first
features:  pandas Series with the indicator values the model saw (names like "trend__dist_sma_200")
"""


def allow_trade(symbol, time, direction, prob, features):
    # Example rule: don't buy far above the 200-period average, don't sell far below it.
    dist = features.get("trend__dist_sma_200") if features is not None else None
    if dist is not None and direction * dist > 8:
        return False
    return True
