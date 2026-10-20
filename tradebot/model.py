"""The self-learning part.

A `Learner` holds a champion ensemble of two models:
  * an online linear model that is updated on every newly-labelled bar (fast adaptation), and
  * a gradient-boosted tree model retrained periodically on a rolling window (captures non-linear patterns).
The two are blended with weights that follow their recent accuracy, so whichever is working better
right now gets more say.

Retraining happens on a schedule and whenever recent accuracy drops below a threshold (concept
drift). A retrained "challenger" only replaces the champion if it scores better on the most recent
data, so the bot cannot replace a working model with a worse one.
"""
import collections

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

LN2 = float(np.log(2))
WEIGHT_SHARPNESS = 20.0
LOSS_EMA = 0.05


def _log_loss(y, p, w=None):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(np.average(losses, weights=w))


class _Ensemble:
    def __init__(self, learning, genome, seed):
        self.learning, self.genome, self.seed = learning, genome, seed

    def fit(self, X, y, w):
        self.scaler = StandardScaler().fit(X)
        Xs = self._transform(X)
        self.online = SGDClassifier(loss="log_loss", alpha=self.learning["online_alpha"], random_state=self.seed)
        for _ in range(5):
            self.online.partial_fit(Xs, y, classes=[0, 1], sample_weight=w)
        self.batch = HistGradientBoostingClassifier(
            max_depth=int(self.genome["gbm_max_depth"]),
            learning_rate=float(self.genome["gbm_learning_rate"]),
            max_iter=int(self.learning["gbm_max_iter"]),
            l2_regularization=1.0,
            max_bins=63,
            early_stopping=False,
            random_state=self.seed,
        )
        # Small data: one thread is faster, and keeps parallel evolution workers from oversubscribing CPUs.
        with threadpool_limits(1):
            self.batch.fit(Xs, y, sample_weight=w)
        self.loss = np.array([LN2, LN2])
        return self

    def _transform(self, X):
        return np.nan_to_num(self.scaler.transform(X), nan=0.0, posinf=0.0, neginf=0.0)

    def components(self, X):
        Xs = self._transform(X)
        return self.online.predict_proba(Xs)[:, 1], self.batch.predict_proba(Xs)[:, 1]

    def weights(self):
        z = -WEIGHT_SHARPNESS * self.loss
        e = np.exp(z - z.max())
        return e / e.sum()

    def predict(self, X):
        p_online, p_batch = self.components(X)
        w = self.weights()
        return w[0] * p_online + w[1] * p_batch

    def update(self, X, y, sw):
        """Learn from rows this ensemble has never seen; returns its predictions made before learning."""
        p_online, p_batch = self.components(X)
        w = self.weights()
        before = w[0] * p_online + w[1] * p_batch
        batch_losses = np.array([_log_loss(y, p_online, sw), _log_loss(y, p_batch, sw)])
        self.loss = (1 - LOSS_EMA) * self.loss + LOSS_EMA * batch_losses
        self.online.partial_fit(self._transform(X), y, sample_weight=sw)
        return before


class Learner:
    def __init__(self, learning, genome, feature_names):
        self.learning, self.genome = dict(learning), dict(genome)
        self.feature_names = list(feature_names)
        self.champion = None
        self.version = 0
        self.X = np.empty((0, len(self.feature_names)))
        self.y = np.empty(0)
        self.w = np.empty(0)
        self.bars_since_retrain = 0
        self.recent_hits = collections.deque(maxlen=int(learning["drift_window"]))
        self.events = []

    @property
    def ready(self):
        return self.champion is not None

    @property
    def recent_accuracy(self):
        return float(np.mean(self.recent_hits)) if self.recent_hits else None

    def _prepare(self, X, y, w):
        X = X.reindex(columns=self.feature_names).to_numpy(float)
        y = np.asarray(y, float)
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        keep = np.isfinite(X).all(axis=1) & ~np.isnan(y)
        return X[keep], y[keep].astype(int), w[keep]

    def _recency_weights(self, n):
        age = np.arange(n)[::-1]
        return 0.5 ** (age / float(self.genome["recency_half_life"]))

    def _train(self, X, y, w):
        if len(y) < 100 or len(np.unique(y)) < 2:
            return None
        return _Ensemble(self.learning, self.genome, self.learning["seed"]).fit(X, y, w * self._recency_weights(len(y)))

    def fit(self, X, y, w=None):
        """Initial training on history."""
        self.X, self.y, self.w = self._prepare(X, y, w)
        self._trim()
        self.retrain("initial")
        return self

    def observe(self, X, y, w=None):
        """Feed bars whose outcome just became known: adapt online, track accuracy, retrain if needed."""
        X, y, w = self._prepare(X, y, w)
        if not len(y):
            return
        if self.ready:
            before = self.champion.update(X, y, w)
            self.recent_hits.extend(((before >= 0.5).astype(int) == y).tolist())
        self.X, self.y, self.w = np.vstack([self.X, X]), np.concatenate([self.y, y]), np.concatenate([self.w, w])
        self._trim()
        self.bars_since_retrain += len(y)

        drifting = (
            len(self.recent_hits) == self.recent_hits.maxlen
            and self.recent_accuracy < self.learning["drift_threshold"]
            and self.bars_since_retrain >= self.learning["retrain_every_bars"] // 4
        )
        if not self.ready:
            self.retrain("initial")
        elif drifting:
            self.retrain(f"drift (recent accuracy {self.recent_accuracy:.3f})")
        elif self.bars_since_retrain >= self.learning["retrain_every_bars"]:
            self.retrain("scheduled")

    def _trim(self):
        keep = int(self.learning["train_window_bars"])
        self.X, self.y, self.w = self.X[-keep:], self.y[-keep:], self.w[-keep:]

    def retrain(self, reason):
        self.bars_since_retrain = 0
        n_val = int(self.learning["validation_bars"])
        if not self.ready or len(self.y) < 3 * n_val:
            candidate = self._train(self.X, self.y, self.w)
            if candidate is not None:
                self.champion, self.version = candidate, self.version + 1
                self.recent_hits.clear()
                self.events.append({"reason": reason, "promoted": True, "version": self.version})
            return

        X_tr, y_tr, w_tr = self.X[:-n_val], self.y[:-n_val], self.w[:-n_val]
        X_val, y_val, w_val = self.X[-n_val:], self.y[-n_val:], self.w[-n_val:]
        candidate = self._train(X_tr, y_tr, w_tr)
        if candidate is None:
            return
        old_loss = _log_loss(y_val, self.champion.predict(X_val), w_val)
        new_loss = _log_loss(y_val, candidate.predict(X_val), w_val)
        promoted = new_loss <= old_loss - self.learning["promote_min_improvement"]
        if promoted:
            candidate.update(X_val, y_val, w_val)
            self.champion, self.version = candidate, self.version + 1
            self.recent_hits.clear()
        self.events.append(
            {"reason": reason, "promoted": promoted, "version": self.version,
             "champion_loss": round(old_loss, 4), "challenger_loss": round(new_loss, 4)}
        )

    def predict_proba(self, X):
        """Probability that price moves up first, for each row of X."""
        if not self.ready:
            return np.full(len(X), 0.5)
        return self.champion.predict(X.reindex(columns=self.feature_names).to_numpy(float))

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
