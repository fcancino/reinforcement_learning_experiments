import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import wrapper

from actor_critic import ActorCritic
from multiprocessing_env import SubprocVecEnv
from minipacman import MiniPacman
from environment_model import *



from IPython.display import clear_output
#import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)


mode = "regular"
num_envs = 16

env_name = "Breakout-v0"
def make_env():
    def _thunk():
        env = wrapper.make_env(env_name)
        return env

    return _thunk

env = make_env()
envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

state_shape = env().observation_space.shape
action_space = env().action_space.n
num_pixels = 210
print("state shape", state_shape)
print("action_space", action_space)

env_model = EnvModel(state_shape, num_pixels, num_rewards=6)#len(mode_rewards["regular"]))
actor_critic = ActorCritic(state_shape, action_space)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(env_model.parameters())

if USE_CUDA:
    env_model    = env_model.cuda()
    actor_critic = actor_critic.cuda()

actor_critic.load_state_dict(torch.load("breakout_cien_mil"))

def pix_to_target(next_states):
    target = []
    for pixel in next_states.transpose(0, 2, 3, 1).reshape(-1, 3):
        target.append(pixel_to_categorical[tuple([np.ceil(pixel[0]), np.ceil(pixel[1]), np.ceil(pixel[2])])])
    return target

def target_to_pix(imagined_states):
    pixels = []
    to_pixel = {value: key for key, value in pixel_to_categorical.items()}
    for target in imagined_states:
        pixels.append(list(to_pixel[target]))
    return np.array(pixels)

def rewards_to_target(mode, rewards):
    target = []
    for reward in rewards:
        target.append(reward_to_categorical[mode][reward])
    return target


def get_action(state):
    if state.ndim == 4:
        state = torch.FloatTensor(np.float32(state))
    else:
        state = torch.FloatTensor(np.float32(state)).unsqueeze(0)
        
    action = actor_critic.act(Variable(state, volatile=True))
    action = action.data.cpu().squeeze(1).numpy()
    return action


def play_games(envs, frames):
    states = envs.reset()
    
    for frame_idx in range(frames):
        actions = get_action(states)
        next_states, rewards, dones, _ = envs.step(actions)
        
        yield frame_idx, states, actions, rewards, next_states, dones
        
        states = next_states

reward_coef = 0.1
num_updates = 10000

losses = []
all_rewards = []

for frame_idx, states, actions, rewards, next_states, dones in play_games(envs, num_updates):
 
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)

    batch_size = states.size(0)
    
    onehot_actions = torch.zeros(batch_size, action_space, *state_shape[1:])
    onehot_actions[range(batch_size), actions] = 1
    inputs = Variable(torch.cat([states, onehot_actions], 1))
    # print()
    # print("INSIDE LOOP")
    # print("states_size",states.size())
    # print("actions_size",actions.size())
    # print("one_hot_action_size",onehot_actions.size())
    # print("inputs_size",inputs.size())
    
    if USE_CUDA:
        inputs = inputs.cuda()

    
    # print()
    # print("INSIDE NET")
    imagined_state, imagined_reward = env_model(inputs)#states, onehot_actions)


    target_state = pix_to_target(next_states)
    target_state = Variable(torch.LongTensor(target_state))
    
    target_reward = rewards_to_target(mode, rewards)
    target_reward = Variable(torch.LongTensor(target_reward))

    optimizer.zero_grad()
    image_loss  = criterion(imagined_state, target_state)
    reward_loss = criterion(imagined_reward, target_reward)
    loss = image_loss + reward_coef * reward_loss
    loss.backward()
    optimizer.step()
    
    losses.append(loss.data[0])
    all_rewards.append(np.mean(rewards))
    
    if frame_idx % 100 == 0:
        print("frame", frame_idx)
        # plot(frame_idx, all_rewards, losses)

torch.save(env_model.state_dict(), "breakout_model")
