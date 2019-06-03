
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import wrapper
import gym
import torch.nn.utils as nn_utils
import time
import csv

from actor_critic_02 import ActorCritic
from multiprocessing_env import SubprocVecEnv
from minipacman import MiniPacman
from environment_model import *
from i2a_02 import *
import common


USE_CUDA = torch.cuda.is_available()
ROLLOUTS_STEPS = 3
LEARNING_RATE = 1e-4
POLICY_LR = 1e-4
TEST_EVERY_BATCH = 1000
NUM_ENVS = 16
NUM_OF_EPISODES = 100000
REWARD_STEPS = 5
GAMMA = 0.99
CLIP_GRAD = 0.5
ENTROPY_BETA = 0.01
VALUE_LOSS_COEF = 0.5
BATCH_SIZE = REWARD_STEPS * 16
SAVE_EVERY_BATCH = 10000

def set_seed(seed, envs=None, cuda=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    if envs:
        for idx, env in enumerate(envs):
            env.seed(seed + idx)



if __name__ == "__main__":
    device = torch.device("cuda" if USE_CUDA else "cpu")
