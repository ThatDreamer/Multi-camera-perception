from collections import deque
import random
import gym
from maze_env import Maze
import time
import numpy as np
import tensorflow as tf
from DQNs import DQN
import matplotlib.pyplot as plt

# env = gym.make('MountainCar-v0')
episode = 100  # 训练次数
MEMORY_SIZE = 2048  # it needs to be set to 2**n because of the query way in the sumtree

def update():
    
    total_steps = 0
    steps = []
    episodes = []
    for i in range(episode):
        s = env.reset()
        # print('s', s)
        step = 0
        while True:
            env.render()
            a = agent.act(s)
            next_s, reward, done = env.step(a)
            agent.remember(s, a, next_s, reward)
            
            if total_steps > MEMORY_SIZE:
                agent.train()
            s = next_s
    
            if done:
                print('episode ', i, ' finished')
                steps.append(step)
                episodes.append(i)
                break
            step += 1
            total_steps += 1
    
    # agent.save_model()
    # draw the change chart of the number of steps to reach the target with the number of training steps
    his_prio = np.vstack((episodes, steps))
    print('his_prio', his_prio)
    # plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
    plt.plot(his_prio[0, :], his_prio[1, :], c='r', label='DQN with prioritized replay')
    plt.legend(loc='best')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.show()
if __name__ == "__main__":
    env = Maze()
    agent = DQN(
        n_actions=4, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.0001, prioritized=True, double_q=True, dueling=False
    )
    # update()
    env.after(100, update)
    env.mainloop()
