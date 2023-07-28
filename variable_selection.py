from itertools import combinations
import numpy as np

total_list = list(np.arange(12))

res = [com for com in combinations(total_list, 3)]

rng = np.random.default_rng()
rng.integers(0, 11, size=3)

breakpoint()