import torch
import torch.optim as optim

import common
from actor_critic_02 import ActorCritic

LEARNING_RATE = 1e-4
TEST_EVERY_BATCH = 300

if __name__ == "__main__":
	device = torch.device("cuda")
	envs = [common.make_env() for _ in range(common.NUM_ENVS)]
	test_env = common.make_env(test=True)
	net = ActorCritic(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
	print(net)
	optimizer = optim.RMSprop(net.parameters(), lr=LEARNING_RATE, eps=1e-5)
	step_idx = 0
	total_steps = 0
	best_reward = None
	best_test_reward = None
	
	for mb_obs, mb_rewards, mb_actions, mb_values, _, done_rewards, done_steps in common.iterate_batches(envs, net, device=device):
		common.train_a2c(net, mb_obs, mb_rewards, mb_actions, mb_values,
                             optimizer, step_idx, device=device)
		step_idx += 1

		if step_idx % TEST_EVERY_BATCH == 0:
			test_reward, test_steps = common.test_model(test_env, net, device=device)
			if best_test_reward is None or best_test_reward < test_reward:
				if best_test_reward is not None:
					fname = "pacman_a2c.net"
					torch.save(net.state_dict(), fname)
				best_test_reward = test_reward
			print("%d: test reward=%.2f, steps=%.2f, best_reward=%.2f" % (step_idx, test_reward, test_steps, best_test_reward))
		
		if step_idx > 100000:
			break
