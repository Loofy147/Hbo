import numpy as np
from spoknas.utils import pairwise_sq_dists

def test_pairwise():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 1.0]])
    d2 = pairwise_sq_dists(a, b)
    assert d2.shape == (2,2)
    assert d2[0,0] == 0.0
    assert d2[1,1] == 2.0