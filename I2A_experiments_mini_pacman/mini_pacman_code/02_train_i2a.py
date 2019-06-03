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

    envs = [common.make_env() for _ in range(NUM_ENVS)]
    test_env = common.make_env(test=True)

    obs_shape = envs[0].observation_space.shape
    act_n = envs[0].action_space.n

    net_policy = ActorCritic(obs_shape, act_n).to(device)

    net_em = EnvironmentModel(obs_shape, act_n)
    net_em.load_state_dict(torch.load("breakout.env"))
    net_em = net_em.to(device)

    net_i2a = I2A(obs_shape, act_n, net_em, net_policy, ROLLOUTS_STEPS).to(device)
    print(net_i2a)

    obs = envs[0].reset()
    obs_v = common.default_states_preprocessor([obs]).to(device)
    res = net_i2a(obs_v)

    optimizer = optim.RMSprop(net_i2a.parameters(), lr=LEARNING_RATE, eps=1e-5)
    policy_opt = optim.Adam(net_policy.parameters(), lr=POLICY_LR)

    step_idx = 0
    total_steps = 0
    ts_start = time.time()
    best_reward = None
    best_test_reward = None
    for mb_obs, mb_rewards, mb_actions, mb_values, mb_probs, done_rewards, done_steps in \
            common.iterate_batches(envs, net_i2a, device):
        if len(done_rewards) > 0:
            total_steps += sum(done_steps)
            speed = total_steps / (time.time() - ts_start)
            if best_reward is None:
                best_reward = done_rewards.max()
            elif best_reward < done_rewards.max():
                best_reward = done_rewards.max()
           
        obs_v = common.train_a2c(net_i2a, mb_obs, mb_rewards, mb_actions, mb_values,
                                 optimizer, step_idx, device=device)
        # policy distillation
        probs_v = torch.FloatTensor(mb_probs).to(device)
        policy_opt.zero_grad()
        logits_v, _ = net_policy(obs_v)
        policy_loss_v = -F.log_softmax(logits_v, dim=1) * probs_v.view_as(logits_v)
        policy_loss_v = policy_loss_v.sum(dim=1).mean()
        policy_loss_v.backward()
        policy_opt.step()

        step_idx += 1

        if step_idx % SAVE_EVERY_BATCH == 0:
            fname = "breakout.i2a"
            torch.save(net_em.state_dict(), fname)
            torch.save(net_policy.state_dict(), fname + ".policy")

        if step_idx % TEST_EVERY_BATCH == 0:
            test_reward, test_steps = common.test_model(test_env, net_i2a, device=device)
            
            append_to_file = [step_idx, test_reward]
            with open('i2a_performance.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(append_to_file)

            if best_test_reward is None or best_test_reward < test_reward:
                if best_test_reward is not None:
                    fname = "breakout.i2a"
                    torch.save(net_i2a.state_dict(), fname)
                    torch.save(net_policy.state_dict(), fname + ".policy")
                    print("Save I2A NET at step", step_idx)
                else:
                    fname = "breakout.env.i2a"
                    torch.save(net_em.state_dict(), fname)
                    print("Save I2A ENV NET at step", step_idx)
                best_test_reward = test_reward
            print("%d: test reward=%.2f, steps=%.2f, best_reward=%.2f" % (
                step_idx, test_reward, test_steps, best_test_reward))

fname = "breakout.i2a"
torch.save(net_i2a.state_dict(), fname)
torch.save(net_policy.state_dict(), fname + ".policy")
torch.save(net_em.state_dict(), fname + ".env.dat")
