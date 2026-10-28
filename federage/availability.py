"""Client availability profiles p_1, ..., p_N used by the random access model.

Every profile first draws `np.random.random(N)`, even when it does not use the
draw, so that the global NumPy random stream matches the published runs.
"""

import numpy as np


def _tiered(start, num_users):
    """Exponentially decaying weights, tripled for the first fifth and doubled for the middle three fifths."""
    K = num_users
    p = np.exp(np.linspace(start, 1, num=K))
    p[:K // 5], p[K // 5:4 * (K // 5)] = 3 * p[:K // 5], 2 * p[K // 5:4 * (K // 5)]
    return p / np.sum(p)


def availability_profile(name, num_users):
    draw = np.random.random(num_users)
    if name == 'uni':
        # "Uniform" regime of the paper.
        return draw / np.sum(draw)
    if name == 'lin':
        return _tiered(1, num_users)
    if name == 'exp':
        return _tiered(2, num_users)
    if name == 'supexp':
        return _tiered(3, num_users)
    if name == 'superexp':
        # "Non-uniform" regime of the paper (Fig. 5).
        return _tiered(4, num_users)
    if name == 'blobs':
        # Five-client example of Fig. 1.
        return [.33, .3, .27, .05, .05]
    raise ValueError(f'unknown availability profile {name}')
