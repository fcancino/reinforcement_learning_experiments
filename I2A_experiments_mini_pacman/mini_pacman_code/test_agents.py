import gym

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

import wrapper

from actor_critic_02 import ActorCritic
from i2a_02 import *
from environment_model import *
import common 

mode = "regular"
USE_CUDA = torch.cuda.is_available()
FRAMES_COUNT = 2


env_name = "Breakout-v0"


env = common.make_env(test=True)

state_shape = env.observation_space.shape
action_space = env.action_space.n

##RANDOM TEST
rewards_per_episode = 0
list_of_rewards = []
for i_episode in range(20):
    rewards_per_episode = 0
    observation = env.reset()
    done = False
    t = 0
    while not done:
        t += 1
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        rewards_per_episode += reward

    list_of_rewards.append(rewards_per_episode)
print("random actions training")
print(list_of_rewards)
print("RANDOM: One Million Updates MEAN", np.mean(list_of_rewards))

#A2C TEST
actor_critic = ActorCritic(state_shape, action_space)


if USE_CUDA:
    actor_critic  = actor_critic.cuda()

actor_critic.load_state_dict(torch.load("pacman_one_million1"))

device = torch.device("cuda" if USE_CUDA else "cpu")
list_of_rewards = []
total_reward = 0.0
total_steps = 0
agent = common.PolicyAgent(lambda x: actor_critic(x)[0], device=device, apply_softmax=True)
done = False    


for _ in range(20):
    obs = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        env.render()    
        action = agent([obs])[0][0]
        obs, r, done, _ = env.step(action)
        total_reward += r

    list_of_rewards.append(total_reward)
print("One Million Updates", list_of_rewards)
print("One Million Updates MEAN", np.mean(list_of_rewards))

#I2A test
# actor_critic = ActorCritic(state_shape, action_space)


# if USE_CUDA:
#     actor_critic  = actor_critic.cuda()

# #actor_critic.load_state_dict(torch.load("pacman_one_million1"))

# device = torch.device("cuda" if USE_CUDA else "cpu")
# list_of_rewards = []
# total_reward = 0.0
# total_steps = 0


# net_em = EnvironmentModel(state_shape, action_space)
# net_em.load_state_dict(torch.load("pacman_one_million_env_model"))
# net_em = net_em.to(device)
# ROLLOUTS_STEPS = 3
# net_i2a = I2A(state_shape, action_space, net_em, actor_critic, ROLLOUTS_STEPS).to(device)
# net_em.load_state_dict(torch.load("pacman_one_million_i2a"))
# print(net_i2a)


# agent = PolicyAgent(lambda x: net_i2a(x)[0], device=device, apply_softmax=True)
# done = False    


# for _ in range(20):
#     obs = env.reset()
#     total_reward = 0.0
#     done = False
#     while not done:
#         action = agent([obs])[0][0]
#         obs, r, done, _ = env.step(action)
#         total_reward += r
#     list_of_rewards.append(total_reward)
# print("I2A: One Million Updates", list_of_rewards)
# print("I2A: One Million Updates MEAN", np.mean(list_of_rewards))
