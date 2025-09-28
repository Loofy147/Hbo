from hpo.optimizers.advanced_optimizers import MultiObjectiveOptimizer, MockSearchSpace, MockHPOSystem

if __name__ == "__main__":
    opt = MultiObjectiveOptimizer(rng_seed=42)
    ss = MockSearchSpace()
    ss.add_int('model_complexity', 1, 10)
    ss.add_categorical('batch_size', [16, 32, 64])
    ss.add_uniform('learning_rate', 1e-5, 1e-1, log=True)
    ss.add_uniform('regularization', 0.0, 0.1)
    best, hpo = opt.optimize_model_tradeoffs(MockHPOSystem, lambda: ss, n_trials=20)
    print("Best:", best.value, best.params)