import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.neighbors import KernelDensity
from hpo.configuration import ConfigurationSpace

class KDESurrogateLevel:
    def __init__(self, bandwidth: float = 1.0):
        self.bandwidth = float(bandwidth)
        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        self.kde_good: Optional[KernelDensity] = None
        self.kde_bad: Optional[KernelDensity] = None

    def add(self, x: np.ndarray, y: float):
        self.X.append(x)
        self.y.append(y)

    def fit(self, top_k_ratio: float = 0.25):
        if len(self.y) == 0:
            return
        X = np.vstack(self.X)
        ys = np.array(self.y)
        idx = np.argsort(ys)  # minimize
        n = len(ys)
        k = max(1, int(np.ceil(n * top_k_ratio)))
        good_idx = idx[:k]
        bad_idx = idx[k:]
        if len(good_idx) > 0:
            self.kde_good = KernelDensity(bandwidth=self.bandwidth).fit(X[good_idx])
        if len(bad_idx) > 0:
            self.kde_bad = KernelDensity(bandwidth=self.bandwidth).fit(X[bad_idx])

    def score(self, x: np.ndarray, meta_score: float = 1.0) -> float:
        lg = self.kde_good.score_samples(x.reshape(1, -1))[0] if (self.kde_good is not None) else -1e9
        lb = self.kde_bad.score_samples(x.reshape(1, -1))[0] if (self.kde_bad is not None) else -1e9
        return float(lg - lb + np.log(max(meta_score, 1e-12)))


class BOHB_KDE:
    """
    BOHB-style controller that uses KDESurrogate per-budget.
    This is intended as a component inside a scheduler/orchestrator.
    """
    def __init__(self, cs: ConfigurationSpace, budgets: List[float], bandwidth: float = 1.0):
        self.cs = cs
        self.levels = {b: KDESurrogateLevel(bandwidth=bandwidth) for b in budgets}

    def observe(self, config: dict, budget: float, value: float):
        x = self.cs.to_array([config])[0]
        level = self.levels.get(budget)
        if level is None:
            # create if needed
            level = KDESurrogateLevel(bandwidth=1.0)
            self.levels[budget] = level
        level.add(x, float(value))

    def fit_all(self):
        for lvl in self.levels.values():
            lvl.fit()

    def propose(self, candidates: List[Dict[str, Any]], top_k: int, meta_scores: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Scores a list of candidate configurations and returns the top_k best.
        If no KDE models are trained, it returns a random subset.
        """
        if not candidates:
            return []

        # Find the highest budget level that has a trained KDE model
        budgets = sorted(self.levels.keys(), reverse=True)
        best_level = None
        for b in budgets:
            lvl = self.levels.get(b)
            if lvl and lvl.kde_good and lvl.kde_bad:
                best_level = lvl
                break

        # If no model is ready, return a random subset of candidates
        if best_level is None:
            np.random.shuffle(candidates)
            return candidates[:top_k]

        Xc = self.cs.to_array(candidates)
        scores = []
        for i, x in enumerate(Xc):
            meta = meta_scores[i] if (meta_scores is not None and i < len(meta_scores)) else 1.0
            try:
                sc = best_level.score(x, meta_score=meta)
            except Exception:
                sc = -1e9  # Assign a very low score on error
            scores.append(sc)

        # Get the indices of the top_k scores
        num_to_return = min(top_k, len(candidates))
        top_indices = np.argsort(scores)[-num_to_return:]

        # Return the corresponding candidates in descending order of score
        return [candidates[i] for i in top_indices[::-1]]