"""Entry point: builds components, runs NAS optimizer OR trainer experiments. This script demonstrates integration and how to run end-to-end. """
import os
import yaml
import argparse
import numpy as np
import torch
from types import SimpleNamespace
from transformers import TrainingArguments, Trainer
from grid_analysis.callbacks import GridAnalysisCallback
from spoknas.optimizer import SpokNASOptimizer
from spoknas.fitness import evaluate_multifidelity
from spoknas.controller import ExperimentController
from sklearn.ensemble import RandomForestRegressor

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument('--profile', default=None)
    p.add_argument('--run_nas', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()

def main():
    args = build_args()
    cfg = yaml.safe_load(open('config.yaml'))

    if args.profile:
        profile_name = args.profile
    else:
        profile_name = cfg['defaults']['profile']

    profile = cfg['profiles'][profile_name]

    training_args = TrainingArguments(
        output_dir=profile.get('out_dir', './results'),
        per_device_train_batch_size=profile.get('batch_size', 64),
        num_train_epochs=3,
        logging_steps=100,
        save_steps=500,
        save_strategy='steps',
        logging_dir='./logs',
        seed=args.seed,
        report_to=['tensorboard']
    )

    grid_cb = GridAnalysisCallback(raw_csv='grid_raw_results.csv', summary_csv='grid_summary.csv', run_on='train_end')

    layer_lib = cfg['optimizer']['layer_library']
    surrogate = RandomForestRegressor(n_estimators=cfg['surrogate']['n_estimators'])
    optimizer = SpokNASOptimizer(
        layer_lib,
        fitness_fn=evaluate_multifidelity,
        population_size=profile.get('pop_size', 24),
        elitism=cfg['optimizer']['elitism'],
        mutation_rate=cfg['optimizer']['mutation_rate'],
        surrogate_enabled=True,
        surrogate_model=surrogate
    )
    controller = ExperimentController()

    # Data manager with synthetic data
    class DataManager:
        def __init__(self, num_samples=200, num_classes=10, img_size=32, num_channels=3):
            self.x_full = np.random.randn(num_samples, num_channels, img_size, img_size).astype(np.float32)
            self.y_full = np.random.randint(0, num_classes, (num_samples,)).astype(int)

    data_manager = DataManager()

    if args.run_nas:
        best, history = optimizer.run_with_controller(
            generations=profile.get('generations', 1), # Reduced for a quick test
            num_islands=profile.get('num_islands', 2), # Reduced
            migrate_every=profile.get('migrate_every', 5),
            migration_k=profile.get('migration_k', 1),
            controller=controller,
            data_manager=data_manager,
            train_idx=list(range(150)),
            val_idx=list(range(150, 200))
        )
        print('NAS done, best:', best)
    else:
        print('Use Trainer for supervised experiments — example code omitted.')

if __name__ == '__main__':
    main()