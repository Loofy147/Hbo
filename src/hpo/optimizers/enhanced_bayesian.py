import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from typing import List, Dict, Any

class EnhancedBayesianOptimizer:
    def __init__(self, config_space, acq='ei', xi=0.01, kappa=2.576, n_init=5):
        self.cs = config_space
        self.acq = acq
        self.xi = xi
        self.kappa = kappa
        self.n_init = n_init
        self.X=[]
        self.y=[]
        self.gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True)

    def _config_to_vec(self, cfg):
        # reuse config_space._configs_to_array if available
        if hasattr(self.cs,'_configs_to_array'):
            return self.cs._configs_to_array([cfg])[0]
        # This is a placeholder and will need a real implementation
        # based on the config_space object used.
        # For now, let's assume a simple dict vectorization.
        vec = []
        for p in self.cs.get_hyperparameters():
             vec.append(cfg.get(p.name))
        return np.array(vec)


    def suggest(self):
        if len(self.X) < self.n_init:
            return self.cs.sample_configuration()
        X_arr = np.vstack(self.X)
        self.gp.fit(X_arr, np.array(self.y))
        # acquisition optimization: random candidate scan
        best=None; best_val=-1e9
        for _ in range(200):
            cand = self.cs.sample_configuration()
            x = self._config_to_vec(cand).reshape(1,-1)
            mu, sigma = self.gp.predict(x, return_std=True)
            if sigma <= 0:
                val = 0.0
            else:
                if self.acq=='ei':
                    fbest = min(self.y)
                    gamma = (fbest - mu - self.xi)/sigma
                    from scipy.stats import norm
                    val = ((fbest - mu - self.xi)*norm.cdf(gamma) + sigma*norm.pdf(gamma))[0]
                else: # ucb
                    val = -(mu - self.kappa*sigma)[0]
            if val>best_val:
                best_val=val; best=cand
        return best

    def tell(self,cfg, y):
        v = self._config_to_vec(cfg)
        self.X.append(v)
        self.y.append(y)