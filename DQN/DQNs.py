from collections import deque
import gym
from matplotlib import pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, Input
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
import time
# np.random.seed(1)
# tf.set_random_seed(1)
tf.compat.v1.disable_eager_execution()


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
        self.max_p = np.max([self.max_p, p])
        self.min_p = np.min([self.min_p, p])
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        # this method is faster than the recursive loop in the reference code
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    # it only applies to trees of size 2**n, which is a perfect binary tree 
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


class Memory(object):
    """
    stored as ( s, a, r, s_ ) in SumTree
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.4  # [0~1] convert the importance of TD error to priority
    beta = 0.6  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        # max_p = np.max(self.tree.tree[-self.tree.capacity:])
        max_p = self.tree.max_p
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), [[] for i in range(n)], np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        min_prob = self.tree.min_p / self.tree.total_p  # for later calculate ISweight
        # sample n times
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            # print('a, b', a, b)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i] = idx, data
        # print('self.tree.tree[-self.tree.capacity:]', self.tree.tree[-self.tree.capacity:])
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

class DQN(object):
    def __init__(
        self,
        n_actions=4,
        n_features=2,
        learning_rate=0.005,
        reward_decay=0.9,
        e_greedy=0.9,
        replace_target_iter=500,
        memory_size=10000,
        batch_size=32,
        e_greedy_increment=None,
        prioritized=False,
        dueling=False,
        double_q=False,
    ):
        self.step = 0
        self.n_actions = n_actions
        self.n_features = n_features
        self.factor = reward_decay
        self.update_freq = replace_target_iter  # 模型更新频率
        self.replay_size = memory_size  # 训练集大小
        self.lr = learning_rate
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment  # epsilon increase with training steps
        self.epsilon = 0.0 if e_greedy_increment is not None else self.epsilon_max
        self.batch_size = batch_size
        self.prioritized = prioritized
        if self.prioritized:
            print('prioritized experience replay')
            self.replay_queue = Memory(self.replay_size)
            self.ISWeights = np.zeros((self.batch_size, 1))
            # print(self.ISWeights)
        else:
            self.replay_queue = deque(maxlen=self.replay_size)
        self.dueling = dueling
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.double_q = double_q
        
        

    def create_model(self):
        """创建一个隐藏层为10的神经网络"""
        if self.dueling:
            inputA = Input(shape=(self.n_features,))
            inputB = Input(shape=(1,))
            x = layers.Dense(self.batch_size, activation='relu')(inputA)
            x1 = layers.Dense(1, activation='linear')(x)
            x2 = layers.Dense(self.n_actions, activation='linear')(x)
            y = x1 + (x2 - tf.reduce_mean(x2, axis=1, keepdims=True))
            model = models.Model(inputs=[inputA, inputB], outputs=y)
        else:
            # model = models.Sequential([
            #     layers.Dense(10, input_dim=self.n_features, activation='relu',
            #                  kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.3, seed=None),
            #                  bias_initializer=initializers.Constant(value=0.1)),
            #     layers.Dense(self.n_actions, activation="linear",
            #                  kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.3, seed=None),
            #                  bias_initializer=initializers.Constant(value=0.1))
            # ])
            inputA = Input(shape=(self.n_features,))
            inputB = Input(shape=(1,))
            x = layers.Dense(self.batch_size, activation='relu',
                             kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.3, seed=None),
                             bias_initializer=initializers.Constant(value=0.1))(inputA)
            y = layers.Dense(self.n_actions, activation='linear',
                              kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.3, seed=None),
                              bias_initializer=initializers.Constant(value=0.1))(x)
            model = models.Model(inputs=[inputA, inputB], outputs=y)

        # Pay attention to this part !!!
        # need to customize the loss function
        def my_loss_wrapper(ISWeights):
            def my_loss(y_true, y_pred):
                return K.mean(ISWeights * K.square(y_pred - y_true), -1)
            return my_loss
        if self.prioritized:
            model.compile(loss=my_loss_wrapper(inputB),
                          optimizer=optimizers.RMSprop(self.lr))
        else:
            model.compile(loss='mean_squared_error',
                          optimizer=optimizers.RMSprop(self.lr))
        return model

    def act(self, s):
        """predict action"""
        data = np.array([s])
        if np.random.uniform() < self.epsilon:
            temp = self.model.predict([data, self.ISWeights])[0]
            a = np.argmax(temp)
        else:    
            a = np.random.randint(0, self.n_actions)
        return a

    def save_model(self, file_path='multiCamerasSensing-v0-dqn.h5'):
        print('model saved')
        self.model.save(file_path)

    def remember(self, s, a, next_s, reward):
        """ store the transition"""
        if self.prioritized:
            self.replay_queue.store((s, a, next_s, reward))
        else:
            self.replay_queue.append((s, a, next_s, reward))

    def train(self):
        # copy the parameters of the real network to the target network
        if self.step % self.update_freq == 0:
            self.target_model.set_weights(self.model.get_weights())

        # sample from the experience buffer/memory
        if self.prioritized:
            tree_idx, replay_batch, self.ISWeights = self.replay_queue.sample(self.batch_size)
        else:
            replay_batch = random.sample(self.replay_queue, self.batch_size)

        # predict the q value
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[2] for replay in replay_batch])
        Q = self.model.predict([s_batch, self.ISWeights])
        Q_eval4next = self.model.predict([next_s_batch, self.ISWeights])
        Q_next = self.target_model.predict([next_s_batch, self.ISWeights])
        
        Q_target = Q.copy()
        # update the target q value by formula
        for i, replay in enumerate(replay_batch):
            _, a, _, reward = replay
            if self.double_q:
                max_act4next = np.argmax(Q_eval4next[i])
                selected_q_next = Q_next[i, max_act4next]
            else:
                selected_q_next = np.amax(Q_next[i])

            Q_target[i][a] = reward + self.factor * selected_q_next

        if self.prioritized:
            abs_errors = np.sum(np.abs(Q - Q_target), axis=1)
            self.replay_queue.batch_update(tree_idx, abs_errors)  # update priority

        # train the network
        self.model.fit([s_batch, self.ISWeights],  Q_target, batch_size=self.batch_size, verbose=0)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.step += 1