import gym.spaces
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import os
import sys
from itertools import count
import time
import argparse

from Agent import Agent
###################################################################################################

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

###################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_const', const=True,default=False,help='train Flag')
parser.add_argument('--test', action='store_const', const=True,default=False,help='test Flag')
opt = parser.parse_args()
###################################################################################################
random_seed = 2

env = gym.make('MountainCarContinuous-v0')
env.seed(random_seed)

# size of each action
action_size = env.action_space.shape[0]
print('Size of each action:', action_size)

# examine the state space
#state_size = env.observation_space.shape[0]
state_size = 4
print('Size of state:', state_size)

action_low = env.action_space.low
print('Action low:', action_low)

action_high = env.action_space.high
print('Action high: ', action_high)

agent = Agent(state_size=state_size, action_size=action_size, random_seed=random_seed)

def save_model():
    print("Model Save...")
    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

def ddpg(n_episodes=100000, max_t=1500, print_every=1, save_every=20):
    scores_deque = deque(maxlen=10)
    scores = []

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        goal = [0,0]
        state = np.hstack((state,goal))
        score = 0
        timestep = time.time()
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = state[0].item()
            next_state = np.hstack((next_state,goal))
            agent.step(state, action, reward, next_state, done, t)  #add normal goal
            alts = state.copy()
            alts[2:] = next_state[:2]
            agent.memory.push(alts,action,1,next_state, True)
            score += reward
            state = next_state.copy()
            if done:
                break

        scores_deque.append(score)
        scores.append(score)
        score_average = np.mean(scores_deque)

        if i_episode % print_every == 0:
            print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Time: {:.2f}'\
                  .format(i_episode, score_average, np.max(scores), np.min(scores), time.time() - timestep), end="\n")

        if np.mean(scores_deque) >= 90.0:
            save_model()
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, score_average))

    return scores

if __name__ == '__main__':

    if opt.train:
        scores = ddpg()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(1, len(scores)+1), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

    elif opt.test:
        agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
        agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

        for _ in range(1):
            state = env.reset()
            for t in range(1200):
                action = agent.act(state, add_noise=False)
                env.render()
                state, reward, done, _ = env.step(action)
                if done:
                    break
        env.close()
    else:
        print('no args')






