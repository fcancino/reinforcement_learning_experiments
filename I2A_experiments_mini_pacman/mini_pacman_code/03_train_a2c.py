import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import wrapper
import gym
import time
import torch.nn.utils as nn_utils

from actor_critic_02 import ActorCritic
from multiprocessing_env import SubprocVecEnv
from minipacman import MiniPacman
from environment_model import *
import common 


LEARNING_RATE = 1e-4
TEST_EVERY_BATCH = 1000
USE_CUDA = torch.cuda.is_available()
NUM_ENVS = 16
REWARD_STEPS = 5
GAMMA = 0.99
BATCH_SIZE = REWARD_STEPS * 16
CLIP_GRAD = 0.5
ENTROPY_BETA = 0.01
VALUE_LOSS_COEF = 0.5
OBS_WEIGHT = 10.0
REWARD_WEIGHT = 1.0

FRAMES_COUNT = 2
IMG_SHAPE = (FRAMES_COUNT, 84, 84)
EM_OUT_SHAPE = (1, ) + IMG_SHAPE[1:]
NUM_OF_EPISODES = 300000

SAVE_EVERY_BATCH = 5000

atari_games = [
        'CarnivalNoFrameskip-v4',
        'AlienNoFrameskip-v4',
        'AmidarNoFrameskip-v4',
        'BankHeistNoFrameskip-v4',
        'MsPacmanNoFrameskip-v4',
        'TutankhamNoFrameskip-v4',
        'VentureNoFrameskip-v4',
        'WizardOfWorNoFrameskip-v4',
        'AssaultNoFrameskip-v4',
        'AsteroidsNoFrameskip-v4',
        'BeamRiderNoFrameskip-v4',
        'CentipedeNoFrameskip-v4',
        'ChopperCommandNoFrameskip-v4',
        'CrazyClimberNoFrameskip-v4',
#        'DemonAttackNoFrameskip-v4',
#        'AtlantisNoFrameskip-v4',
#        'GravitarNoFrameskip-v4',
#        'PhoenixNoFrameskip-v4',
#        'PooyanNoFrameskip-v4',
#        'RiverraidNoFrameskip-v4',
#        'SeaquestNoFrameskip-v4',
#        'SpaceInvadersNoFrameskip-v4',
#        'StarGunnerNoFrameskip-v4',
#        'TimePilotNoFrameskip-v4',
#        'ZaxxonNoFrameskip-v4',
#        'YarsRevengeNoFrameskip-v4',
#        'AsterixNoFrameskip-v4',
#        'ElevatorActionNoFrameskip-v4',
#        'BerzerkNoFrameskip-v4',
#        'FreewayNoFrameskip-v4',
#        'FrostbiteNoFrameskip-v4',
#        'JourneyEscapeNoFrameskip-v4',
#        'KangarooNoFrameskip-v4',
#        'KrullNoFrameskip-v4',
#        'PitfallNoFrameskip-v4',
#        'SkiingNoFrameskip-v4',
#        'UpNDownNoFrameskip-v4',
#        'QbertNoFrameskip-v4',
#        'RoadRunnerNoFrameskip-v4',
#        'DoubleDunkNoFrameskip-v4',
#        'IceHockeyNoFrameskip-v4',
#        'MontezumaRevengeNoFrameskip-v4',
#        'GopherNoFrameskip-v4',
#        'BreakoutNoFrameskip-v4',
#        'PongNoFrameskip-v4',
#        'PrivateEyeNoFrameskip-v4',
#        'TennisNoFrameskip-v4',
#        'VideoPinballNoFrameskip-v4',
#        'FishingDerbyNoFrameskip-v4',
#        'NameThisGameNoFrameskip-v4',
#        'BowlingNoFrameskip-v4',
#        'BattleZoneNoFrameskip-v4',
#        'BoxingNoFrameskip-v4',
#        'JamesbondNoFrameskip-v4',
#        'RobotankNoFrameskip-v4',
#        'SolarisNoFrameskip-v4',
#        'EnduroNoFrameskip-v4',
#        'KungFuMasterNoFrameskip-v4',
        ]


if __name__ == "__main__":

    device = torch.device("cuda" if USE_CUDA else "cpu")

    envs = [common.make_env() for _ in range(NUM_ENVS)]
    

    test_env = common.make_env(test=True)

    net = ActorCritic(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)
    optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, eps=1e-5)

    step_idx = 0
    total_steps = 1000
    best_reward = None
    ts_start = time.time()
    best_test_reward = None

    for mb_obs, mb_rewards, mb_actions, mb_values, _, done_rewards, done_steps in \
            common.iterate_batches(envs, net, device=device):
        if len(done_rewards) > 0:
            total_steps += sum(done_steps)
            speed = total_steps / (time.time() - ts_start)
            if best_reward is None:
                best_reward = done_rewards.max()
            elif best_reward < done_rewards.max():
                best_reward = done_rewards.max()
            

        common.train_a2c(net, mb_obs, mb_rewards, mb_actions, mb_values,
                         optimizer, step_idx, device=device)
        step_idx += 1
      
        # if step_idx % 1000 == 0:
        #     print("step", step_idx)

        if step_idx % SAVE_EVERY_BATCH == 0:
            fname = "breakout_a2c.pth"
            torch.save(net.state_dict(), fname)
   
        if step_idx % TEST_EVERY_BATCH == 0:
            test_reward, test_steps = common.test_model(test_env, net, device=device)
            if best_test_reward is None or best_test_reward < test_reward:
                if best_test_reward is not None:
                    fname = "breakout_a2c.pth"
                    torch.save(net.state_dict(), fname)
                    print("Saving Net")
                best_test_reward = test_reward
            print("%d: test reward=%.2f, steps=%.2f, best_reward=%.2f" % (
                step_idx, test_reward, test_steps, best_test_reward))

fname = "breakout_a2c.pth"
torch.save(net.state_dict(), fname)
