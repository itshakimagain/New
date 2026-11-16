"""Thin wrapper around the official MetaTrader5 Python package (Windows, with the MT5 terminal installed)."""
import time

from .data import rates_to_frame
from .risk import compute_lots

BAR_SECONDS = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D1": 86400}


class BrokerError(RuntimeError):
    pass


class MT5Broker:
    def __init__(self, cfg):
        try:
            import MetaTrader5 as mt5
        except ImportError as exc:
            raise BrokerError(
                "The MetaTrader5 package is not installed. It only runs on Windows next to an MT5 terminal: "
                "pip install MetaTrader5"
            ) from exc
        self.mt5 = mt5
        self.cfg = cfg["mt5"]
        self.magic = int(self.cfg["magic"])

    def connect(self):
        kwargs = {k: self.cfg[k] for k in ("path", "login", "password", "server") if self.cfg.get(k)}
        if "login" in kwargs:
            kwargs["login"] = int(kwargs["login"])
        path = kwargs.pop("path", None)
        ok = self.mt5.initialize(path, **kwargs) if path else self.mt5.initialize(**kwargs)
        if not ok:
            raise BrokerError(f"MT5 initialize failed: {self.mt5.last_error()}")
        info = self.mt5.account_info()
        if info is None:
            raise BrokerError(f"Not logged in to an MT5 account: {self.mt5.last_error()}")
        return info

    def shutdown(self):
        self.mt5.shutdown()

    def is_real_account(self):
        return self.mt5.account_info().trade_mode == self.mt5.ACCOUNT_TRADE_MODE_REAL

    def equity(self):
        return float(self.mt5.account_info().equity)

    def timeframe(self, name):
        return getattr(self.mt5, f"TIMEFRAME_{name}")

    def closed_bars(self, symbol, timeframe, count):
        """The last `count` fully closed bars (position 0 is the still-forming bar, so start at 1)."""
        if not self.mt5.symbol_select(symbol, True):
            base = symbol.split(".")[0][:6]
            similar = [s.name for s in self.mt5.symbols_get(f"*{base}*") or []][:10]
            hint = f" Similar names at your broker: {', '.join(similar)}" if similar else ""
            raise BrokerError(f"Symbol '{symbol}' not found (names are case-sensitive).{hint}")
        # The terminal may still be downloading history, or cap how many bars it keeps
        # (Tools > Options > Charts > Max bars in chart), so retry with fewer bars.
        while True:
            rates = None
            for _ in range(3):
                rates = self.mt5.copy_rates_from_pos(symbol, self.timeframe(timeframe), 1, count)
                if rates is not None and len(rates):
                    return rates_to_frame(rates)
                time.sleep(2)
            if count <= 1000:
                raise BrokerError(f"No bars for {symbol}: {self.mt5.last_error()}. Open a {symbol} chart "
                                  f"in MT5 so it downloads history, then try again.")
            count //= 2

    def spread_points(self, symbol):
        tick, info = self.mt5.symbol_info_tick(symbol), self.mt5.symbol_info(symbol)
        return (tick.ask - tick.bid) / info.point

    def positions(self, symbol=None):
        found = self.mt5.positions_get(symbol=symbol) if symbol else self.mt5.positions_get()
        return [p for p in (found or []) if p.magic == self.magic]

    def lots_for_risk(self, symbol, risk_money, sl_distance):
        info = self.mt5.symbol_info(symbol)
        return compute_lots(risk_money, sl_distance, info.trade_tick_size, info.trade_tick_value,
                            info.volume_min, info.volume_max, info.volume_step)

    def _filling(self, info):
        if info.filling_mode & 1:
            return self.mt5.ORDER_FILLING_FOK
        if info.filling_mode & 2:
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    def _send(self, request):
        result = self.mt5.order_send(request)
        if result is None or result.retcode != self.mt5.TRADE_RETCODE_DONE:
            raise BrokerError(f"order_send failed: {getattr(result, 'retcode', None)} {getattr(result, 'comment', self.mt5.last_error())}")
        return result

    def market_order(self, symbol, direction, volume, sl_distance, tp_distance, comment=""):
        info, tick = self.mt5.symbol_info(symbol), self.mt5.symbol_info_tick(symbol)
        price = tick.ask if direction > 0 else tick.bid
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": self.mt5.ORDER_TYPE_BUY if direction > 0 else self.mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": round(price - direction * sl_distance, info.digits),
            "tp": round(price + direction * tp_distance, info.digits),
            "deviation": int(self.cfg["deviation"]),
            "magic": self.magic,
            "comment": comment[:31],
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(info),
        }
        return self._send(request)

    def close_position(self, position, comment="max hold"):
        info, tick = self.mt5.symbol_info(position.symbol), self.mt5.symbol_info_tick(position.symbol)
        is_buy = position.type == self.mt5.POSITION_TYPE_BUY
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": self.mt5.ORDER_TYPE_SELL if is_buy else self.mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "price": tick.bid if is_buy else tick.ask,
            "deviation": int(self.cfg["deviation"]),
            "magic": self.magic,
            "comment": comment[:31],
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(info),
        }
        return self._send(request)

    def server_time(self, symbol):
        """Current broker server time of the latest tick (MT5 bar times are in server time)."""
        return self.mt5.symbol_info_tick(symbol).time
