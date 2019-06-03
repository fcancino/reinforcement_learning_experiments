import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from actor_critic import ActorCritic
from multiprocessing_env import SubprocVecEnv
from minipacman import MiniPacman
from environment_model import *


from IPython.display import clear_output
#import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

#7 different pixels in MiniPacman
pixels = (
    (0.0, 1.0, 1.0),
    (0.0, 1.0, 0.0), 
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 0.0), 
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
)
pixel_to_categorical = {pix:i for i, pix in enumerate(pixels)} 
num_pixels = len(pixels)
print("Num pixels", num_pixels)
#For each mode in MiniPacman there are different rewards
mode_rewards = {
    "regular": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "avoid":   [0.1, -0.1, -5, -10, -20],
    "hunt":    [0, 1, 10, -20],
    "ambush":  [0, -0.1, 10, -20],
    "rush":    [0, -0.1, 9.9]
}
reward_to_categorical = {mode: {reward:i for i, reward in enumerate(mode_rewards[mode])} for mode in mode_rewards.keys()}

def pix_to_target(next_states):
  #  print("size next_states", next_states.transpose(0,2,3,1).reshape(-1,3))
  #  print("size next_states", next_states)
    target = []
    for pixel in next_states.transpose(0, 2, 3, 1).reshape(-1, 3):
        #print(pixel)
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

def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('loss %s' % losses[-1])
    plt.plot(losses)
    plt.show()
    
def displayImage(image, step, reward):
    s = str(step) + " " + str(reward)
    plt.title(s)
    plt.imshow(image)
    plt.show()


print(len(mode_rewards["regular"]))

mode = "regular"
num_envs = 16

def make_env():
    def _thunk():
        env = MiniPacman(mode, 1000)
        return env

    return _thunk

env = make_env()
envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs)

state_shape = env().observation_space.shape
num_actions = env().action_space.n

print("state shape",state_shape)
print("num actions", num_actions)
print("num pixeles", num_pixels)
env_model = EnvModel(state_shape, num_pixels, len(mode_rewards["regular"]))
actor_critic = ActorCritic(state_shape, num_actions)

print("len mode rewards regular",   len(mode_rewards["regular"]))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(env_model.parameters())

if USE_CUDA:
    env_model    = env_model.cuda()
    actor_critic = actor_critic.cuda()

actor_critic.load_state_dict(torch.load("actor_critic_" + mode))

def get_action(state):
    if state.ndim == 4:
        state = torch.FloatTensor(np.float32(state))
    else:
        state = torch.FloatTensor(np.float32(state)).unsqueeze(0)
        
    action = actor_critic.act(Variable(state, volatile=True))
    action = action.data.cpu().squeeze(1).numpy()
    return action


def  play_games(envs, frames):
    states = envs.reset()
    
    for frame_idx in range(frames):
        actions = get_action(states)
        next_states, rewards, dones, _ = envs.step(actions)
        
        yield frame_idx, states, actions, rewards, next_states, dones
        
        states = next_states

reward_coef = 0.1
num_updates = 10000
SAVE_EVERY_BATCH = 1000

losses = []
all_rewards = []
best_loss = np.inf
for frame_idx, states, actions, rewards, next_states, dones in play_games(envs, num_updates):

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    batch_size = states.size(0)
    
    onehot_actions = torch.zeros(batch_size, num_actions, *state_shape[1:])

    onehot_actions[range(batch_size), actions] = 1
    inputs = Variable(torch.cat([states, onehot_actions], 1))

    if USE_CUDA:
        inputs = inputs.cuda()

    imagined_state, imagined_reward = env_model(inputs)


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


    if loss < best_loss:
            print("SAVING NETWORK: Best loss updated: %.4e -> %.4e" % (best_loss, loss))
            best_loss = loss
            torch.save(env_model.state_dict(), "env_model_" + mode)

    frame_idx += 1
    if frame_idx + 1 % SAVE_EVERY_BATCH == 0:
        print("Saving Network on step: ", frame_idx)
        torch.save(env_model.state_dict(), "env_model_" + mode)

    if frame_idx % 400 == 0:
        print("STEP", frame_idx, "Loss updated: %.4e -> %.4e" % (best_loss, loss))


torch.save(env_model.state_dict(), "env_model_" + mode)