import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, Input
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from dqn import DQN


class Predict():

    def __init__(self):
        self.model = DQN(
            n_actions=4, n_features=2, memory_size=2048,
            e_greedy_increment=0.0001, prioritized=True, double_q=True, dueling=False, forest=False, bPT=False
        ).model
        self.model.load_weights("dqn_new_weights.h5")

    def getAction(self, observation):
        q_val = self.model.predict([observation, np.ones((32, 1))])[0]
        print('q_val', q_val)
        action = np.argmax(q_val)
        print('action', action)
        return action


if __name__ == "__main__":
    PredictAction(1)
