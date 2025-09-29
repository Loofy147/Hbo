import math
from typing import Any, Dict, Tuple, List

class BOHBOptimizer:
    def __init__(self, config_space, min_budget=1, max_budget=27, eta=3):
        self.cs = config_space
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.s_max = int(math.log(max_budget/min_budget)/math.log(eta))
        self.B = (self.s_max+1)*max_budget
        # per-budget surrogate models (could be a KDE / GP)
        self.models = {}

    def _generate_bracket(self,s):
        n = int(math.ceil(self.B / self.max_budget / (s+1) * self.eta**s))
        r = self.max_budget * self.eta**(-s)
        return n,r

    def suggest(self):
        # generate based on current bracket
        return self.cs.sample_configuration(), self.min_budget

    def tell(self, config, budget, value):
        # record observation for budget
        pass