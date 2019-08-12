#BUILTIN PYTHON PACKAGES
import os
import sys
from itertools import count
import time
import argparse
import random

#OPEN SOURCE PYTHON PACKAGES
import gym.spaces
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

#CUSTOM PYTHON PACKAGES
from Agent import Agent

###################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_const', const=True,default=False,help='train Flag')
parser.add_argument('--test', action='store_const', const=True,default=False,help='test Flag')
opt = parser.parse_args()
###################################################################################################
#INITIALIZE ENVIRONMENT AND AGENT

#initialize openaigym environment
random_seed = 2
env = gym.make('MountainCar-v0')
env.seed(random_seed)

# size of each action
action_size = env.action_space.n
print('Size of each action:', action_size)

# examine the state space
state_size = env.observation_space.shape[0]
print('Size of state:', state_size)

#INITIALIZE AGENT
agent = Agent(state_size=state_size, action_size=action_size)

###################################################################################################
###################################################################################################

def train(n_episodes=100000, max_t=1500, print_every=1, save_every=20,obs=None):
    scores_deque = deque(maxlen=100)
    for i_episode in range(1, n_episodes+1):

        #LEARN SOME IDEAS STARTING FROM A RANDOM GOAL
        state = env.reset()
        goal = [random.uniform(0.5,0.6),random.uniform(-0.07,0.07)]
        env.state = np.array(goal)
        print(env)
        quit()
        for t in range(max_t):
            print(env.state)
            print(env)
            quit()
            action = agent.act_dumb(env.state)
            next_state,reward,done,_ = env.step(action)
            quit()

            #LEARN FROM THE EXPERIENCE
            experience = (state,action,reward,next_state,done)
            agent.learn(experience)
            state = next_state.copy()

        #LEARN SOME IDEAS STARTING FROM THE BEGINNING
        state = env.reset()
        score = 0
        timestep = time.time()
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            #LEARN FROM THE EXPERIENCE
            experience = (state,action,reward,next_state,done)
            loss = agent.learn(experience)

            #LEARN FROM THE GOAL
            state = next_state.copy()
            score += reward
            if done: break

        scores_deque.append(score)
        score_average = np.mean(scores_deque)

        #observer to plot the metrics
        if obs:
            obs.plot({"Loss": loss})

        if i_episode % print_every == 0:
            test()
            print('\rEpisode {}, Loss: {:.3f}, Average Score: {:.2f}, Reward: {:.2f}, Time: {:.2f}'\
                  .format(i_episode, loss, score_average, score, time.time() - timestep), end="\n")

        if np.mean(scores_deque) >= 90.0:
            save_model()
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, score_average))

    return scores

def test(n_episodes=1, max_steps=1000):

    for ep in range(n_episodes):
        state = env.reset()

    state = env.reset()
    for t in range(1200):
        action = agent.act_greedy(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done: break

class Metrics:
    def __init__(self):
        plt.ion()
        plt.show()

    def plot(self,items):
        for key,vals in items.items():
            plt.plot(vals)
            plt.draw()
        plt.pause(0.001)

###################################################################################################
###################################################################################################

if __name__ == '__main__':

    if opt.train:
        observer = Metrics()
        scores = train(obs=observer)
        plt.ioff()
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






