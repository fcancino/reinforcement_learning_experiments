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
from environment_model_02 import *
import common



LEARNING_RATE = 1e-3
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
NUM_OF_EPISODES = 50000
SAVE_EVERY_BATCH = 1000


def get_obs_diff(prev_obs, cur_obs):
    prev = np.array(prev_obs)[-1]
    cur = np.array(cur_obs)[-1]
    prev = prev.astype(np.float32) / 255.0
    cur = cur.astype(np.float32) / 255.0
    return cur - prev

#different iterate_batches than the one in common.py
def iterate_batches(envs, net, device="cpu"):
    act_selector = common.ProbabilityActionSelector()
    mb_obs = np.zeros((BATCH_SIZE, ) + IMG_SHAPE, dtype=np.uint8)
    mb_obs_next = np.zeros((BATCH_SIZE, ) + EM_OUT_SHAPE, dtype=np.float32)
    mb_actions = np.zeros((BATCH_SIZE, ), dtype=np.int32)
    mb_rewards = np.zeros((BATCH_SIZE, ), dtype=np.float32)
    obs = [e.reset() for e in envs]
    total_reward = [0.0] * NUM_ENVS
    total_steps = [0] * NUM_ENVS
    batch_idx = 0
    done_rewards = []
    done_steps = []

    for _ in range(500000):
        obs_v = common.default_states_preprocessor(obs).to(device)
        logits_v, values_v = net(obs_v)
        probs_v = F.softmax(logits_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = act_selector(probs)

        for e_idx, e in enumerate(envs):
            o, r, done, _ = e.step(actions[e_idx])
            mb_obs[batch_idx] = obs[e_idx]
            mb_obs_next[batch_idx] = get_obs_diff(obs[e_idx], o)
            mb_actions[batch_idx] = actions[e_idx]
            mb_rewards[batch_idx] = r

            total_reward[e_idx] += r
            total_steps[e_idx] += 1

            batch_idx = (batch_idx + 1) % BATCH_SIZE
            if batch_idx == 0:
                yield mb_obs, mb_obs_next, mb_actions, mb_rewards, done_rewards, done_steps
                done_rewards.clear()
                done_steps.clear()
            if done:
                o = e.reset()
                done_rewards.append(total_reward[e_idx])
                done_steps.append(total_steps[e_idx])
                total_reward[e_idx] = 0.0
                total_steps[e_idx] = 0
            obs[e_idx] = o



if __name__ == "__main__":
    device = torch.device("cuda" if USE_CUDA else "cpu")

    envs = [common.make_env() for _ in range(NUM_ENVS)]

    net = ActorCritic(envs[0].observation_space.shape, envs[0].action_space.n)
    net_em = EnvironmentModel(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    net.load_state_dict(torch.load("pacman_a2c.net"))
    net = net.to(device)
    print(net_em)
    optimizer = optim.Adam(net_em.parameters(), lr=LEARNING_RATE)

    step_idx = 0
    best_loss = np.inf

    for mb_obs, mb_obs_next, mb_actions, mb_rewards, done_rewards, done_steps in iterate_batches(envs, net, device):
        # if len(done_rewards) > 0:
        #     m_reward = np.mean(done_rewards)
        #     m_steps = np.mean(done_steps)
        #     print("%d: done %d episodes, mean reward=%.2f, steps=%.2f" % (
        #         step_idx, len(done_rewards), m_reward, m_steps))
        #     tb_tracker.track("total_reward", m_reward, step_idx)
        #     tb_tracker.track("total_steps", m_steps, step_idx)

        obs_v = torch.FloatTensor(mb_obs).to(device)
        obs_next_v = torch.FloatTensor(mb_obs_next).to(device)
        actions_t = torch.LongTensor(mb_actions.tolist()).to(device)
        rewards_v = torch.FloatTensor(mb_rewards).to(device)

        optimizer.zero_grad()
        out_obs_next_v, out_reward_v = net_em(obs_v.float()/255, actions_t)
        loss_obs_v = F.mse_loss(out_obs_next_v.squeeze(-1), obs_next_v)
        loss_rew_v = F.mse_loss(out_reward_v.squeeze(-1), rewards_v)
        loss_total_v = OBS_WEIGHT * loss_obs_v + REWARD_WEIGHT * loss_rew_v
        loss_total_v.backward()
        optimizer.step()

        loss = loss_total_v.data.cpu().numpy()
        if loss < best_loss:
            print("Best loss updated: %.4e -> %.4e" % (best_loss, loss))
            best_loss = loss
            fname = "pacman_env.net"
            print("Saving Network")
            print("STEP", step_idx)
            torch.save(net_em.state_dict(), fname)

        step_idx += 1
        if step_idx % SAVE_EVERY_BATCH == 0:
            fname = "breakout.env"
            print("Saving Network")
            print("STEP", step_idx)
            torch.save(net_em.state_dict(), fname)

        if step_idx % 200 == 0:
            print("STEP", step_idx, "Loss updated: %.4e -> %.4e" % (best_loss, loss))


