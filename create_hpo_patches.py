import textwrap
import pathlib

def generate_hpo_patches():
    """
    Generates a set of .patch files to apply a series of improvements to the HPO codebase,
    as described in the original user request.
    """
    # Use a writable directory like /tmp to avoid permission errors.
    patch_dir = pathlib.Path("/tmp/patches")
    patch_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating patches in: {patch_dir}")

    # --- Content for NEW and MODIFIED files ---

    # This content is synthesized based on the descriptions "KDE-based TPE, GP+EI surrogate, ASHA resource-aware."
    # It is intended to be the content of the `hpo_algorithms_improved.py` file.
    alg_content = r'''
import math
import numpy as np
from scipy.stats import gaussian_kde, norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from .samplers import RandomSampler

class TPESamplerKDE:
    """TPE sampler using Kernel Density Estimation for better density modeling."""
    def __init__(self, n_startup_trials=10, n_ei_candidates=24, gamma=0.25, random_state=None):
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.gamma = gamma
        self.random_sampler = RandomSampler()

    def suggest(self, trials, search_space):
        complete_trials = [t for t in trials if t.state == 'COMPLETE']
        if len(complete_trials) < self.n_startup_trials:
            return self.random_sampler.suggest(trials, search_space)

        sorted_trials = sorted(complete_trials, key=lambda t: t.value, reverse=True)
        n_good = max(1, int(len(sorted_trials) * self.gamma))
        good_trials, bad_trials = sorted_trials[:n_good], sorted_trials[n_good:]

        best_params, best_ei = None, -np.inf
        for _ in range(self.n_ei_candidates):
            candidate = self.random_sampler.suggest(trials, search_space)
            good_density = self._compute_kde_density(candidate, good_trials, search_space)
            bad_density = self._compute_kde_density(candidate, bad_trials, search_space)
            ei = good_density / (bad_density + 1e-9)
            if ei > best_ei:
                best_ei, best_params = ei, candidate
        return best_params if best_params is not None else self.random_sampler.suggest(trials, search_space)

    def _compute_kde_density(self, candidate, trials, search_space):
        if not trials: return 1.0
        log_density = 0.0
        for name, value in candidate.items():
            param_values = [t.params[name] for t in trials if name in t.params]
            if not param_values: continue
            param_config = search_space.params[name]
            if param_config['type'] in ['uniform', 'int']:
                if len(param_values) > 1:
                    kde = gaussian_kde(param_values)
                    log_density += np.log(max(kde.pdf([value])[0], 1e-9))
            elif param_config['type'] == 'categorical':
                counts = {val: param_values.count(val) for val in param_config['choices']}
                prob = (counts.get(value, 0) + 1) / (len(param_values) + len(param_config['choices']))
                log_density += np.log(prob)
        return np.exp(log_density)

class GaussianProcessSamplerEI:
    """Gaussian Process sampler with Expected Improvement acquisition function."""
    def __init__(self, n_startup_trials=10, n_ei_candidates=100, random_state=None):
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.random_sampler = RandomSampler()
        self.gp = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True, n_restarts_optimizer=5, random_state=random_state)

    def suggest(self, trials, search_space):
        complete_trials = [t for t in trials if t.state == 'COMPLETE' and t.params]
        if len(complete_trials) < self.n_startup_trials:
            return self.random_sampler.suggest(trials, search_space)
        X, y = self._get_training_data(complete_trials, search_space)
        if X.shape[0] < 2: return self.random_sampler.suggest(trials, search_space)

        self.gp.fit(X, y)
        candidates_X = np.array([self._params_to_numpy(self.random_sampler.suggest(trials, search_space), search_space) for _ in range(self.n_ei_candidates)])
        mu, sigma = self.gp.predict(candidates_X, return_std=True)
        best_y = np.max(y)

        with np.errstate(divide='ignore', invalid='ignore'):
            Z = (mu - best_y) / (sigma + 1e-9)
            ei = (mu - best_y) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        best_candidate_idx = np.argmax(ei)
        return self._numpy_to_params(candidates_X[best_candidate_idx], search_space)

    def _params_to_numpy(self, params, search_space):
        vec = []
        for name, config in sorted(search_space.params.items()):
            val = params.get(name)
            if config['type'] in ['uniform', 'int']:
                vec.append((val - config['low']) / (config['high'] - config['low']))
            elif config['type'] == 'categorical':
                one_hot = np.zeros(len(config['choices']))
                one_hot[config['choices'].index(val)] = 1
                vec.extend(one_hot)
        return np.array(vec)

    def _numpy_to_params(self, vec, search_space):
        params = {}
        idx = 0
        for name, config in sorted(search_space.params.items()):
            if config['type'] in ['uniform', 'int']:
                val = vec[idx] * (config['high'] - config['low']) + config['low']
                params[name] = int(round(val)) if config['type'] == 'int' else val
                idx += 1
            elif config['type'] == 'categorical':
                choice_idx = np.argmax(vec[idx : idx + len(config['choices'])])
                params[name] = config['choices'][choice_idx]
                idx += len(config['choices'])
        return params

    def _get_training_data(self, trials, search_space):
        X = np.array([self._params_to_numpy(t.params, search_space) for t in trials])
        y = np.array([t.value for t in trials])
        return X, y

class ASHAResourceAware:
    """ASHA pruner adapted to be compatible with the project's pruner API."""
    def __init__(self, min_resource=1, reduction_factor=3):
        self.min_resource = min_resource
        self.reduction_factor = reduction_factor
        self.rungs = {}

    def should_prune(self, trial_id, step, value, completed_trials=None):
        if step < self.min_resource: return False
        rung_level = int(math.log(step / self.min_resource, self.reduction_factor)) if self.min_resource > 0 and self.reduction_factor > 1 else 0
        if rung_level not in self.rungs: self.rungs[rung_level] = []
        if not any(t[0] == trial_id for t in self.rungs[rung_level]):
             self.rungs[rung_level].append((trial_id, value))

        rung_trials = self.rungs[rung_level]
        num_promotions = len(rung_trials) // self.reduction_factor
        if num_promotions == 0: return False

        sorted_trials = sorted(rung_trials, key=lambda x: x[1], reverse=True)
        top_trial_ids = {t[0] for t in sorted_trials[:num_promotions]}
        return trial_id not in top_trial_ids
'''

    utils_content = r'''from pathlib import Path
import tempfile
import json

def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as tf:
        json.dump(data, tf, ensure_ascii=False, indent=2)
        tmpname = tf.name
    path.replace(tmpname)
    return str(path)
'''

    readme_add = r'''## Improvements included (automated patch bundle)

This patch bundle adds:

- `src/hpo/algorithms_improved.py`: KDE-based TPE, GP+EI surrogate, ASHA resource-aware.
- `src/hpo/utils_io.py`: atomic JSON saving helper.
- `run_hpo.py` updated: choose sampler via `--sampler` CLI flag and logging setup.

### New dependencies (recommended)

- `scipy` (for gaussian_kde)
- `scikit-learn` (for GaussianProcessRegressor)

### How to apply the patches

1. Save each `.patch` file to your repo root.
2. Run: `git apply path/to/patchfile.patch` (or `git am` for email-style patches).
3. Run tests and adjust small conflicts if your repo files differ from these patches' assumptions.

If conflicts occur, open the patched files and integrate manually. The patches are intentionally split so you can apply them piecewise.
'''

    # --- Constructing the patch strings ---

    patch_a = textwrap.dedent(f"""\
        diff --git a/src/hpo/algorithms_improved.py b/src/hpo/algorithms_improved.py
        new file mode 100644
        index 0000000..1111111
        --- /dev/null
        +++ b/src/hpo/algorithms_improved.py
        @@ -0,0 +1,{len(alg_content.strip().splitlines())} @@
        {chr(10).join(['+' + line for line in alg_content.strip().splitlines()])}
        """)

    patch_b = textwrap.dedent("""\
        diff --git a/run_hpo.py b/run_hpo.py
        index e69de29..2222222 100644
        --- a/run_hpo.py
        +++ b/run_hpo.py
        @@ -0,0 +1,30 @@
        +import argparse
        +import logging
        +from src.hpo import algorithms_improved
        +
        +def parse_args():
        +    parser = argparse.ArgumentParser(description='Run HPO experiments with selectable samplers')
        +    parser.add_argument('--sampler', choices=['random','tpe_kde','gp_ei'], default='random', help='Sampler to use')
        +    parser.add_argument('--config', type=str, default=None, help='Path to experiment config (YAML/JSON)')
        +    return parser.parse_args()
        +
        +def main():
        +    args = parse_args()
        +    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        +    logging.info('Starting HPO run with sampler=%s', args.sampler)
        +    # اختيار المولد
        +    if args.sampler == 'tpe_kde':
        +        sampler = algorithms_improved.TPESamplerKDE(random_state=42)
        +    elif args.sampler == 'gp_ei':
        +        sampler = algorithms_improved.GaussianProcessSamplerEI()
        +    else:
        +        sampler = None  # Use existing random sampler in the project
        +
        +    # TODO: قم بربط هذا المولد في حلقة التجارب الحالية (invoke objective & observe)
        +    logging.info('Sampler initialized: %s', type(sampler).__name__ if sampler else 'RandomSampler (internal)')
        +
        +if __name__ == '__main__':
        +    main()
        """)

    patch_c = textwrap.dedent("""\
        *** Begin Patch
        *** Update File: src/hpo/some_module.py
        @@ -1,7 +1,7 @@
        -    try:
        -        # some risky work
        -    except:
        -        pass
        +    try:
        +        # some risky work
        +    except Exception as _e:
        +        import logging
        +        logging.getLogger(__name__).exception(_e)
        *** End Patch
        """)

    patch_d = textwrap.dedent(f"""\
        diff --git a/src/hpo/utils_io.py b/src/hpo/utils_io.py
        new file mode 100644
        index 0000000..2222222
        --- /dev/null
        +++ b/src/hpo/utils_io.py
        @@ -0,0 +1,{len(utils_content.strip().splitlines())} @@
        {chr(10).join(['+' + line for line in utils_content.strip().splitlines()])}
        """)

    patch_e = textwrap.dedent(f"""\
        diff --git a/README.md b/README.md
        index 1111111..3333333 100644
        --- a/README.md
        +++ b/README.md
        @@ -0,0 +1,{len(readme_add.strip().splitlines())} @@
        {chr(10).join(['+' + line for line in readme_add.strip().splitlines()])}
        """)

    # --- Write patches to files ---

    patch_files = {
        "01_add_algorithms_improved.patch": patch_a,
        "02_update_run_hpo_add_cli.patch": patch_b,
        "03_replace_bare_except_example.patch": patch_c,
        "04_add_utils_io_atomic_write.patch": patch_d,
        "05_update_README_with_patches.patch": patch_e
    }

    for name, content in patch_files.items():
        path = patch_dir / name
        # Correct newline handling for cross-platform compatibility
        with open(path, "w", encoding="utf-8", newline='\n') as f:
            f.write(content)
        print(f"  - Wrote {name}")

    print("\\nPatch generation complete.")
    print(f"Patches are available in: {patch_dir}")


if __name__ == "__main__":
    generate_hpo_patches()