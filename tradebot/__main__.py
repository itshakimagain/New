"""Command line: python -m tradebot <command> --help"""
import argparse
import csv
import json
import sys
from pathlib import Path

from .config import load_config, load_genome, save_genome, state_dir
from .data import load_csv, save_csv, synthetic
from .features import build_features, load_hints, load_hook


def _data_path(cfg, symbol):
    return Path(cfg["paths"]["data_dir"]) / f"{symbol}_{cfg['timeframe']}.csv"


def _load(cfg, args):
    path = Path(args.csv) if args.csv else _data_path(cfg, args.symbol)
    if not path.exists():
        sys.exit(f"{path} not found. Run `fetch` (needs MT5) or `demo-data` first, or pass --csv.")
    df = load_csv(path)
    feats = build_features(df, load_hook(cfg["paths"]["custom_features"], "add_features"))
    return df, feats, load_hints(cfg["paths"]["hints_file"], args.symbol)


def _print_metrics(title, m):
    print(f"\n{title}")
    for key, value in m.items():
        print(f"  {key:>18}: {value:.3f}" if isinstance(value, float) else f"  {key:>18}: {value}")


def cmd_demo_data(cfg, args):
    path = Path(args.csv) if args.csv else _data_path(cfg, args.symbol)
    save_csv(synthetic(args.bars, seed=args.seed), path)
    print(f"wrote {args.bars} synthetic bars to {path}")


def cmd_fetch(cfg, args):
    from .broker import MT5Broker
    broker = MT5Broker(cfg)
    broker.connect()
    try:
        df = broker.closed_bars(args.symbol, cfg["timeframe"], args.bars or cfg["history_bars"])
    finally:
        broker.shutdown()
    path = _data_path(cfg, args.symbol)
    save_csv(df, path)
    print(f"wrote {len(df)} bars to {path}")


def cmd_backtest(cfg, args):
    from .backtest import run_backtest
    df, feats, hints = _load(cfg, args)
    genome = load_genome(cfg, args.symbol)
    result = run_backtest(df, feats, cfg, genome, args.symbol, hints)
    _print_metrics(f"walk-forward backtest {args.symbol} (genome v{genome.get('version', 0)})", result["metrics"])
    out = state_dir(cfg)
    result["trades"].to_csv(out / f"backtest_trades_{args.symbol}.csv", index=False)
    result["equity"].rename("equity").to_csv(out / f"backtest_equity_{args.symbol}.csv")
    promoted = sum(e["promoted"] for e in result["learner"].events)
    print(f"\nmodel retrained {len(result['learner'].events)} times during the test, {promoted} promoted")
    print(f"trades and equity curve saved in {out}/")


def cmd_evolve(cfg, args):
    from .evolve import evolve
    df, feats, hints = _load(cfg, args)
    if args.population:
        cfg["evolution"]["population"] = args.population
    if args.generations:
        cfg["evolution"]["generations"] = args.generations
    base = load_genome(cfg, args.symbol)
    result = evolve(df, feats, cfg, base, args.symbol, hints)
    _print_metrics(f"holdout since {result['holdout_start']}: current genome", result["holdout_current"])
    _print_metrics("holdout: evolved genome", result["holdout_best"])
    print(f"\nholdout fitness current {result['holdout_fitness_current']:.3f} vs evolved {result['holdout_fitness_best']:.3f}")
    print(json.dumps(result["best"], indent=2))
    if not result["adopted"]:
        print("evolved genome did not beat the current one on unseen data - keeping current genome")
    elif args.dry_run:
        print("evolved genome is better (not saved because of --dry-run)")
    else:
        save_genome(cfg, args.symbol, result["best"])
        print(f"saved genome v{result['best']['version']} - the live bot will retrain with it on next start")


def cmd_run(cfg, args):
    from .bot import LiveBot
    if args.mode:
        cfg["mode"] = args.mode
    LiveBot(cfg).run(once=args.once)


def cmd_hint(cfg, args):
    path = Path(cfg["paths"]["hints_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(["time", "symbol", "direction", "weight", "note"])
        writer.writerow([args.time, args.symbol, args.direction, args.weight or "", args.note or ""])
    print(f"added hint to {path}")


def cmd_status(cfg, args):
    print(f"mode: {cfg['mode']}  timeframe: {cfg['timeframe']}  symbols: {', '.join(cfg['symbols'])}")
    for symbol in cfg["symbols"]:
        print(f"\n{symbol} genome:")
        print(json.dumps(load_genome(cfg, symbol), indent=2))
        model = state_dir(cfg) / "models" / f"{symbol}.joblib"
        if model.exists():
            import joblib
            learner = joblib.load(model)["learner"]
            acc = learner.recent_accuracy
            print(f"model v{learner.version}, recent accuracy {'n/a' if acc is None else f'{acc:.3f}'}, "
                  f"last retrains: {learner.events[-3:]}")
    journal = state_dir(cfg) / "journal.csv"
    if journal.exists():
        print(f"\nlast journal entries ({journal}):")
        print("".join(journal.read_text().splitlines(keepends=True)[-10:]))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tradebot", description="Self-learning MetaTrader 5 trading bot")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, help_text):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--symbol", default=None)
        p.set_defaults(fn=fn)
        return p

    p = add("demo-data", cmd_demo_data, "generate synthetic prices to try the bot without MT5")
    p.add_argument("--csv")
    p.add_argument("--bars", type=int, default=20000)
    p.add_argument("--seed", type=int, default=1)
    p = add("fetch", cmd_fetch, "download history from MT5 into data/")
    p.add_argument("--bars", type=int)
    p = add("backtest", cmd_backtest, "walk-forward backtest with continuous learning")
    p.add_argument("--csv")
    p = add("evolve", cmd_evolve, "let the bot search for better settings and adopt them if they hold up")
    p.add_argument("--csv")
    p.add_argument("--population", type=int)
    p.add_argument("--generations", type=int)
    p.add_argument("--dry-run", action="store_true")
    p = add("run", cmd_run, "run live on MT5")
    p.add_argument("--mode", choices=["dry_run", "trade"])
    p.add_argument("--once", action="store_true", help="process one bar per symbol and exit")
    p = add("hint", cmd_hint, "teach the bot: say which way the market should have been read at a time")
    p.add_argument("--time", required=True, help='bar time, e.g. "2026-10-19 14:15"')
    p.add_argument("--direction", required=True, choices=["up", "down"])
    p.add_argument("--weight", type=float)
    p.add_argument("--note")
    add("status", cmd_status, "show genomes, model versions and recent journal")

    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    args.symbol = (args.symbol or cfg["symbols"][0]).upper()
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
