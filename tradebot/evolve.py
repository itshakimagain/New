"""Strategy evolution: a genetic search over the bot's own settings (thresholds, stops, label
definition, which indicator groups to use, model hyper-parameters).

Candidates are scored by walk-forward backtests on the older part of the data. The winner is then
tested on a recent holdout period it never saw, and is only adopted if it beats the current genome
there as well - which guards against the search simply memorising the past.
"""
import copy

import numpy as np
from joblib import Parallel, delayed

from .backtest import run_backtest
from .config import ALL_FEATURE_GROUPS


def fitness(m, min_trades):
    if m["trades"] == 0:
        return -10.0
    return m["sharpe"] * min(1.0, m["trades"] / min_trades) - 0.02 * m["max_drawdown_pct"]


def _is_int(bounds):
    return all(isinstance(b, int) for b in bounds)


def mutate(genome, bounds, rate, rng):
    g = copy.deepcopy(genome)
    for key, (lo, hi) in bounds.items():
        if rng.random() < rate:
            value = g[key] + rng.normal(0, (hi - lo) * 0.2)
            g[key] = int(np.clip(round(value), lo, hi)) if _is_int((lo, hi)) else float(np.clip(value, lo, hi))
    groups = set(g["feature_groups"])
    for group in ALL_FEATURE_GROUPS:
        if rng.random() < rate / 3:
            groups ^= {group}
    if len(groups - {"custom"}) >= 2:
        g["feature_groups"] = [x for x in ALL_FEATURE_GROUPS if x in groups]
    return g


def crossover(a, b, bounds, rng):
    child = copy.deepcopy(a)
    for key in list(bounds) + ["feature_groups"]:
        if rng.random() < 0.5:
            child[key] = copy.deepcopy(b[key])
    return child


def _evaluate(df, feats, cfg, genome, symbol, hints, score_from=None):
    try:
        return run_backtest(df, feats, cfg, genome, symbol, hints, score_from)["metrics"]
    except ValueError:
        return {"trades": 0, "sharpe": 0.0, "max_drawdown_pct": 100.0, "return_pct": 0.0}


def evolve(df, feats, cfg, base_genome, symbol, hints=None, log=print):
    ev = cfg["evolution"]
    rng = np.random.default_rng(ev["seed"])
    bounds = {k: tuple(v) for k, v in ev["bounds"].items()}
    split = int(len(df) * (1 - ev["holdout_frac"]))
    df_in, feats_in = df.iloc[:split], feats.iloc[:split]
    holdout_start = df.index[split]
    parallel = Parallel(n_jobs=ev["n_jobs"])

    population = [copy.deepcopy(base_genome)] + [
        mutate(base_genome, bounds, max(ev["mutation_rate"], 0.6), rng) for _ in range(ev["population"] - 1)
    ]
    scored = []
    for gen in range(ev["generations"]):
        results = parallel(delayed(_evaluate)(df_in, feats_in, cfg, g, symbol, hints) for g in population)
        scored = sorted(
            ((fitness(m, ev["min_trades"]), g, m) for g, m in zip(population, results)),
            key=lambda item: item[0], reverse=True,
        )
        best_score, _, best_m = scored[0]
        log(f"generation {gen + 1}/{ev['generations']}: best fitness {best_score:.3f} "
            f"(sharpe {best_m['sharpe']:.2f}, trades {best_m['trades']}, dd {best_m['max_drawdown_pct']:.1f}%)")
        elites = [g for _, g, _ in scored[: ev["elite"]]]
        parents = [g for _, g, _ in scored[: max(2, len(scored) // 2)]]
        children = []
        while len(elites) + len(children) < ev["population"]:
            a, b = rng.choice(len(parents), 2, replace=False)
            children.append(mutate(crossover(parents[a], parents[b], bounds, rng), bounds, ev["mutation_rate"], rng))
        population = elites + children

    best = scored[0][1]
    base_hold, best_hold = parallel(
        delayed(_evaluate)(df, feats, cfg, g, symbol, hints, holdout_start) for g in (base_genome, best)
    )
    base_fit, best_fit = fitness(base_hold, ev["min_trades"]), fitness(best_hold, ev["min_trades"])
    changed = {k: v for k, v in best.items() if k != "version"} != {k: v for k, v in base_genome.items() if k != "version"}
    adopted = changed and best_fit > base_fit + ev["min_holdout_improvement"]
    if adopted:
        best = copy.deepcopy(best)
        best["version"] = int(base_genome.get("version", 0)) + 1
    return {
        "best": best, "adopted": adopted, "holdout_start": str(holdout_start),
        "holdout_current": base_hold, "holdout_best": best_hold,
        "holdout_fitness_current": base_fit, "holdout_fitness_best": best_fit,
    }
