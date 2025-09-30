"""SpokNASOptimizer: vectorized surrogate training, islands, checkpoints, robust logging."""
from __future__ import annotations
import os
import pickle
import random
import warnings
from copy import deepcopy
from typing import List
import numpy as np
from types import SimpleNamespace
from .utils import as_f32_contiguous

class SpokNASOptimizer:
    def __init__(self, layer_lib: List[str], fitness_fn,
                 population_size: int = 24, elitism: int = 2,
                 mutation_rate: float = 0.06, crossover_rate: float = 0.6,
                 surrogate_enabled: bool = False, surrogate_model=None):
        self.layer_lib = layer_lib
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.elitism = elitism
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.best_individual = None
        self.surrogate_enabled = surrogate_enabled
        self.surrogate_model = surrogate_model
        self.surrogate_data_X = []
        self.surrogate_data_y = []

    def _random_genome(self):
        arch = [random.choice(self.layer_lib) for _ in range(random.randint(3, 10))]
        if not any(t.startswith('fc') for t in arch):
            arch += ['flatten', 'fc-128']
        return {'genome': {'architecture': arch,
                           'lr': 10 ** random.uniform(-4, -2),
                           'wd': 10 ** random.uniform(-6, -3),
                           'batch_size': random.choice([64, 128])},
                'fitness': None}

    def _random_individual(self):
        return self._random_genome()

    def _genome_to_feature_vector(self, genome_dict):
        arch = genome_dict['architecture']
        length = len(arch)
        num_conv = sum(1 for t in arch if t.startswith('conv') or t.startswith('sep_conv'))
        num_fc = sum(1 for t in arch if t.startswith('fc'))
        counts = []
        for t in arch:
            if '-' in t:
                try:
                    counts.append(float(t.split('-')[1]))
                except:
                    pass
        avg_channels = float(np.mean(counts)) if counts else 0.0
        return [length, num_conv, num_fc, avg_channels,
                genome_dict['lr'], genome_dict['wd'], genome_dict['batch_size']]

    def _crossover(self, g1, g2):
        a1 = g1['architecture']; a2 = g2['architecture']
        if len(a1) < 2 or len(a2) < 2:
            child_arch = a1
        else:
            p1 = random.randint(1, len(a1)-1); p2 = random.randint(1, len(a2)-1)
            child_arch = a1[:p1] + a2[p2:]
        child = {'architecture': child_arch[:16],
                 'lr': (g1['lr'] + g2['lr']) / 2.0,
                 'wd': g1['wd'],
                 'batch_size': random.choice([64,128])}
        return child

    def _mutate(self, genome_dict):
        arch = genome_dict['architecture'][:]
        if random.random() < self.mutation_rate:
            op = random.choice(['add', 'del', 'swap'])
            if op == 'add' and len(arch) < 16:
                arch.insert(random.randint(0, len(arch)), random.choice(self.layer_lib))
            elif op == 'del' and len(arch) > 3:
                del arch[random.randint(0, len(arch)-1)]
            elif op == 'swap' and len(arch) > 1:
                i, j = random.sample(range(len(arch)), 2); arch[i], arch[j] = arch[j], arch[i]
        lr = genome_dict['lr'] * (10 ** random.uniform(-0.1, 0.1)) if random.random() < self.mutation_rate else genome_dict['lr']
        return {'architecture': arch, 'lr': lr, 'wd': genome_dict['wd'], 'batch_size': genome_dict.get('batch_size', 64)}

    def _evaluate_individual(self, individual, *args, **kwargs):
        try:
            genome = individual['genome']
            res = self.fitness_fn(genome, *args, **kwargs)
            individual.update(res)
            return individual
        except Exception as e:
            warnings.warn(f"eval failure: {e}")
            individual.update({'fitness': 0.0, 'val_acc': 0.0, 'params': 0})
            return individual

    def _batch_feature_matrix(self, individuals: List[dict]) -> np.ndarray:
        feats = [self._genome_to_feature_vector(ind['genome']) for ind in individuals]
        return as_f32_contiguous(np.vstack(feats))

    def run_with_controller(self, generations=20, num_islands=4, migrate_every=5, migration_k=2, controller=None, *args, **kwargs):
        population = [self._random_individual() for _ in range(self.population_size)]
        islands = [[] for _ in range(num_islands)]
        for i, ind in enumerate(population):
            islands[i % num_islands].append(ind)

        # initial evaluation
        for isl in islands:
            for ind in isl:
                self._evaluate_individual(ind, *args, **kwargs)
        # store surrogate data
        all_inds = [ind for isl in islands for ind in isl]
        X = self._batch_feature_matrix(all_inds)
        y = np.array([ind.get('fitness', 0.0) for ind in all_inds], dtype=np.float32)
        self.surrogate_data_X.append(X)
        self.surrogate_data_y.append(y)

        history = []
        best = None
        for gen in range(1, generations + 1):
            for i_idx, isl in enumerate(islands):
                isl.sort(key=lambda x: x.get('fitness', -1e9), reverse=True)
                elites = isl[:self.elitism]
                new_pop = elites[:]
                while len(new_pop) < len(isl):
                    p1 = max(random.sample(isl, min(3, len(isl))), key=lambda x: x.get('fitness', -1e9))
                    p2 = max(random.sample(isl, min(3, len(isl))), key=lambda x: x.get('fitness', -1e9))
                    child_genome = self._crossover(p1['genome'], p2['genome'])
                    child_genome = self._mutate(child_genome)
                    child = {'genome': child_genome}
                    child = self._evaluate_individual(child, *args, **kwargs)
                    new_pop.append(child)
                islands[i_idx] = new_pop

            # migration
            if migrate_every and gen % migrate_every == 0 and num_islands > 1:
                migrants = [sorted(isl, key=lambda x: x.get('fitness', -1e9), reverse=True)[:migration_k] for isl in islands]
                for i_idx in range(len(islands)):
                    dest = (i_idx + 1) % len(islands)
                    islands[dest][-migration_k:] = [deepcopy(m) for m in migrants[i_idx]]

            # update surrogate dataset every few gens
            all_inds = [ind for isl in islands for ind in isl]
            X = self._batch_feature_matrix(all_inds)
            y = np.array([ind.get('fitness', 0.0) for ind in all_inds], dtype=np.float32)
            self.surrogate_data_X.append(X)
            self.surrogate_data_y.append(y)

            if self.surrogate_enabled and self.surrogate_model is not None and len(self.surrogate_data_X) > 0:
                Xall = np.vstack(self.surrogate_data_X)
                yall = np.concatenate(self.surrogate_data_y)
                try:
                    self.surrogate_model.fit(Xall, yall)
                except Exception:
                    pass

            # global snapshot
            all_inds = [ind for isl in islands for ind in isl]
            all_inds.sort(key=lambda x: x.get('fitness', -1e9), reverse=True)
            best = deepcopy(all_inds[0])
            history.append(best.get('fitness', 0.0))

        self.best_individual = best
        return best, history