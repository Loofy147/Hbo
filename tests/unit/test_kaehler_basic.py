import numpy as np
import sys
import os

# Adjust the path to import from the root 'src' directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from unified_quantum_hpo.kaehler_optimizer import KaehlerHPOOptimizer, ConfigSpaceShim

def toy_obj(cfg):
    x = cfg['x']; y = cfg['y']
    return (x-0.1)**2 + (y+0.2)**2

def test_complexify_and_potential_diag_metric():
    cs = ConfigSpaceShim(['x','y'])
    opt = KaehlerHPOOptimizer(cs, toy_obj, use_complex_step=True)
    cfg = {'x': 0.0, 'y': 0.0}
    z = opt.complexify_config(cfg)
    K = opt.compute_kahler_potential(z)
    g = opt.compute_kahler_metric(z, method='diag')
    assert np.isfinite(K)
    assert g.shape == (2,2)
    assert np.all(np.isfinite(g))

def test_seed_reproducibility():
    cs = ConfigSpaceShim(['x','y'])
    opt1 = KaehlerHPOOptimizer(cs, toy_obj, use_complex_step=False)
    opt2 = KaehlerHPOOptimizer(cs, toy_obj, use_complex_step=False)
    opt1.seed = 123
    np.random.seed(opt1.seed)
    z1 = opt1.complexify_config({'x':0.1,'y':0.2})

    opt2.seed = 123
    np.random.seed(opt2.seed)
    z2 = opt2.complexify_config({'x':0.1,'y':0.2})

    assert np.allclose(z1, z2)