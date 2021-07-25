from collections import deque
import gym
from matplotlib import pyplot as plt
import numpy as np
import random
import time
from sum_tree import SumTree


class Memory(object):
    """
    stored as ( s, a, r, s_ ) in SumTree
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4 # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = self.tree.max_p
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), [[] for i in range(n)], np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = min(1., self.beta + self.beta_increment_per_sampling)  # max = 1
        min_prob = self.tree.min_p / self.tree.total_p  # for later calculate ISweight
        # sample n times
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = pow(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        """
        update the node tree_idx to abs_errors
        """
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


