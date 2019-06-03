import gym
from gym import spaces

from deepmind import PillEater, observation_as_rgb

# class DiscreteOneHotWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         super(DiscreteOneHotWrapper, self).__init__(env)
#         assert isinstance(env.observation_space, gym.spaces.Discrete)
#         self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)

#     def observation(self, observation):
#         res = np.copy(self.observation_space.low)
#         res[observation] = 1.0
#         return res


class AtariGame:
    def __init__(self, mode, frame_cap):
        self.mode      = mode
        self.frame_cap = frame_cap
        
        self.env = gym.make('Breakout-v0')
        
        self.action_space      =  env.action_space #spaces.Discrete(5)
        self.observation_space = env.observation_space #spaces.Box(low=0, high=1.0, shape=(3, 15, 19))

    def step(self, action):
        self.env.step(action)
        env_reward, env_pcontinue, env_frame = self.env.observation()
        self.done = env_pcontinue != 1
        env_frame = env_frame.transpose(2, 0, 1)
        return env_frame, env_reward, self.done, {}

    def reset(self):
        image, _, _ = self.env.start()
        image = observation_as_rgb(image)
        self.done = False
        image = image.transpose(2, 0, 1)
        return image

# def main():
#     env = gym.make('MsPacman-v0')
#     for i_episode in range(2000):
#         observation = env.reset()
#         for t in range(200):
            
#             env.render()
#             #print(observation)
#             action = env.action_space.sample()
#             observation, reward, done, info = env.step(action)
#             if done:
#                 print("Episode finished after {} timesteps".format(t+1))
#                 break    



# if __name__ == '__main__':
#     main()