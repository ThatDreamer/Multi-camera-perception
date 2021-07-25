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
from memory import Memory
tf.compat.v1.disable_eager_execution()


class DQN(object):
    def __init__(
            self,
            n_actions=4,
            n_features=2,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            memory_size=2048,
            batch_size=32,
            e_greedy_increment=None,
            prioritized=False,
            dueling=False,
            double_q=False,
            forest=False,
            bPT=False,
    ):
        self.step = 0
        self.n_actions = n_actions
        self.n_features = n_features
        self.factor = reward_decay
        self.update_freq = replace_target_iter  # the update frequency of model
        self.replay_size = memory_size  # the size of training data
        self.lr = learning_rate
        self.epsilon_max = e_greedy  
        self.epsilon_increment = e_greedy_increment  # epsilon increase with training steps
        self.epsilon = 0.0 if e_greedy_increment is not None else self.epsilon_max
        self.batch_size = batch_size
        self.prioritized = prioritized
        self.ISWeights = np.zeros((self.batch_size, 1))
        if self.prioritized:
            print('prioritized experience replay')
            self.replay_queue = Memory(self.replay_size)
        else:
            self.replay_queue = deque(maxlen=self.replay_size)
        self.dueling = dueling
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.double_q = double_q
        self.loss = [] 

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

    def act(self, s, flag=False):
        """predict action"""
        data = np.array([s])
        if np.random.uniform() < self.epsilon and flag:

            temp = self.model.predict([data, self.ISWeights])[0]
            a = np.argmax(temp)
        else:
            a = np.random.randint(0, self.n_actions)
        return a

    def save_model(self, file_path='multiCamerasSensing-v0-dqn.h5'):
        print('model saved')
        self.model.save(file_path)

    def save_model_weights(self, file_path='dqn_weights.h5'):
        print('model weights saved')
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
        history = self.model.fit([s_batch, self.ISWeights], Q_target, batch_size=self.batch_size, verbose=0)
        self.loss.append(history.history['loss'][0])
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.step += 1
