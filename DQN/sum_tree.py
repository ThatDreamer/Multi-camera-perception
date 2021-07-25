from collections import deque
import gym
from matplotlib import pyplot as plt
import numpy as np
import random
import time


class SumTree(object):
    """
    the SumTree Class
    Attributes:
        capacity: the size of memory
        max_p: the max priority in the tree
        min_p: the min priority in the tree
        tree: the tree node which stores priorities and their sum
        data: store transitions of the memory
    """
    data_pointer = 0
    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.max_p = 0
        self.min_p = np.inf
        self.tree = np.zeros(2 * capacity - 1)  # all nodes
        self.data = np.zeros(capacity, dtype=object)  # for all transitions

    def add(self, p, data):
        """
        add node
        :param p: the priority
        :param data: the transition
        :return: null
        """
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        """
        update priority of the node tree_idx
        """
        self.max_p = max(self.max_p, p)
        self.min_p = min(self.min_p, p)
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        # this method is faster than the recursive loop in the reference code
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        get the index, priority and transition by priority v
        """
        parent_idx = 0
        # the while loop is faster than the method in the reference code
        while True:
            # this leaf's left and right kids
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            # reach bottom, end search
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            # downward search, always search for a higher priority node
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root



if __name__ == '__main__':
    batch_size = 32
    unit = 1
    steps = 10
    n = batch_size * unit
    sumTree = SumTree(n)
    data = [0 for i in range(n)]
    for i in range(n):
        data[i] = random.uniform(0, 1)
    print('test add:')
    t = time.time()
    for i in range(n):
        sumTree.add(data[i], i)
    print('time of add:', time.time() - t)

    print('test sample:')
    t = time.time()
    for i in range(steps):
        for j in range(batch_size):
            v = random.uniform(0, sumTree.total_p)
            print(v)
            print(sumTree.get_leaf(v))
    print('time of sample', time.time() - t)
