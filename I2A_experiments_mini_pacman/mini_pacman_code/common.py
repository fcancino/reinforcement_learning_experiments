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
NUM_OF_EPISODES = 1000000
ACT_PONDER_PENALTY = .1
class ActionSelector:
    """
    Abstract class which converts scores to the actions
    """
    def __call__(self, scores):
        raise NotImplementedError

class ProbabilityActionSelector(ActionSelector):
    """
    Converts probabilities of actions into action by sampling them
    """
    def __call__(self, probs):
        assert isinstance(probs, np.ndarray)
        actions = []
        for prob in probs:
            actions.append(np.random.choice(len(prob), p=prob))
        return np.array(actions)

def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def iterate_batches(envs, net, act=False, device="cpu"):
    n_actions = envs[0].action_space.n
    act_selector = ProbabilityActionSelector()
    obs = [e.reset() for e in envs]
    batch_dones = [[False] for _ in range(NUM_ENVS)]
    total_reward = [0.0] * NUM_ENVS
    total_steps = [0] * NUM_ENVS
    mb_obs = np.zeros((NUM_ENVS, REWARD_STEPS) + IMG_SHAPE, dtype=np.uint8)
    mb_rewards = np.zeros((NUM_ENVS, REWARD_STEPS), dtype=np.float32)
    mb_values = np.zeros((NUM_ENVS, REWARD_STEPS), dtype=np.float32)
    mb_actions = np.zeros((NUM_ENVS, REWARD_STEPS), dtype=np.int32)
    mb_probs = np.zeros((NUM_ENVS, REWARD_STEPS, n_actions), dtype=np.float32)

    for _ in range(NUM_OF_EPISODES):
        batch_dones = [[dones[-1]] for dones in batch_dones]
        done_rewards = []
        done_steps = []
        for n in range(REWARD_STEPS):
            obs_v = default_states_preprocessor(obs).to(device)
            mb_obs[:, n] = obs_v.data.cpu().numpy()
            if act:
                logits_v, values_v, _ = net(obs_v)
            else:
                logits_v, values_v = net(obs_v)

            probs_v = F.softmax(logits_v, dim=1)
            probs = probs_v.data.cpu().numpy()
            actions = act_selector(probs)
            mb_probs[:, n] = probs
            mb_actions[:, n] = actions
            mb_values[:, n] = values_v.squeeze().data.cpu().numpy()
            for e_idx, e in enumerate(envs):
                o, r, done, _ = e.step(actions[e_idx])
                total_reward[e_idx] += r
                total_steps[e_idx] += 1
                if done:
                    o = e.reset()
                    done_rewards.append(total_reward[e_idx])
                    done_steps.append(total_steps[e_idx])
                    total_reward[e_idx] = 0.0
                    total_steps[e_idx] = 0
                obs[e_idx] = o
                mb_rewards[e_idx, n] = r
                batch_dones[e_idx].append(done)
        # obtain values for the last observation
        obs_v = default_states_preprocessor(obs).to(device)
        _, values_v, _ = net(obs_v)
        values_last = values_v.squeeze().data.cpu().numpy()

        for e_idx, (rewards, dones, value) in enumerate(zip(mb_rewards, batch_dones, values_last)):
            rewards = rewards.tolist()
            if not dones[-1]:
                rewards = discount_with_dones(rewards + [value], dones[1:] + [False], GAMMA)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones[1:], GAMMA)
            mb_rewards[e_idx] = rewards

        out_mb_obs = mb_obs.reshape((-1,) + IMG_SHAPE)
        out_mb_rewards = mb_rewards.flatten()
        out_mb_actions = mb_actions.flatten()
        out_mb_values = mb_values.flatten()
        out_mb_probs = mb_probs.flatten()
        yield out_mb_obs, out_mb_rewards, out_mb_actions, out_mb_values, out_mb_probs, \
              np.array(done_rewards), np.array(done_steps)


def train_a2c(net, mb_obs, mb_rewards, mb_actions, mb_values, optimizer,  step_idx, act=False ,device="cpu"):
    optimizer.zero_grad()
    mb_adv = mb_rewards - mb_values
    adv_v = torch.FloatTensor(mb_adv).to(device)
    obs_v = torch.FloatTensor(mb_obs).to(device)
    rewards_v = torch.FloatTensor(mb_rewards).to(device)
    actions_t = torch.LongTensor(mb_actions).to(device)
    logits_v, values_v, ponder_dict = net(obs_v)
    log_prob_v = F.log_softmax(logits_v, dim=1)
    log_prob_actions_v = adv_v * log_prob_v[range(len(mb_actions)), actions_t]

    loss_policy_v = -log_prob_actions_v.mean()
    loss_value_v = F.mse_loss(values_v.squeeze(-1), rewards_v)

    prob_v = F.softmax(logits_v, dim=1)
    entropy_loss_v = (prob_v * log_prob_v).sum(dim=1).mean()
    loss_v = ENTROPY_BETA * entropy_loss_v + VALUE_LOSS_COEF * loss_value_v + loss_policy_v
    
    if ponder_dict:
        loss_v += (
            ACT_PONDER_PENALTY * ponder_dict["ponder_cost"].mean()
        )    
    loss_v.backward()
    nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
    optimizer.step()

    return obs_v

def make_env(test=False, clip=True):
    if test:
        args = {'reward_clipping': False,
                'episodic_life': False}
    else:
        args = {'reward_clipping': clip}
    return wrapper.wrap_dqn(gym.make('BreakoutNoFrameskip-v4'),
                                         stack_frames=FRAMES_COUNT,
                                         **args)

class BaseAgent:
    """
    Abstract Agent interface
    """
    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError

class PolicyAgent(BaseAgent):
    """
    Policy agent gets action probabilities from the model and samples actions from it
    """
    # TODO: unify code with DQNAgent, as only action selector is differs.
    def __init__(self, model, action_selector=ProbabilityActionSelector(), device="cpu",
                 apply_softmax=False, preprocessor=default_states_preprocessor):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    def __call__(self, states, agent_states=None):
        """
        Return actions from given list of states
        :param states: list of states
        :return: list of actions
        """
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
        probs_v = self.model(states)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        actions = self.action_selector(probs)
        return np.array(actions), agent_states


def test_model(env, net, rounds=3, device="cpu"):
    total_reward = 0.0
    total_steps = 0
    agent = PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)

    for _ in range(rounds):
        obs = env.reset()
        while True:
            action = agent([obs])[0][0]
            obs, r, done, _ = env.step(action)
            total_reward += r
            total_steps += 1
            if done:
                break
    return total_reward / rounds, total_steps / rounds
