import tensorflow as tf
import torch
import gym
import kinova_sim
import numpy as np

import torch
from ppo import ppo
import gym


def main():
    what_to_do = 'train' #to train new agent put 'train' to run trained agent put 'play'

    if what_to_do == 'train':
        env_fn = lambda: gym.make('kinova-v0') #make env

        ac_kwargs = dict(hidden_sizes=[64, 128, 64], activation=torch.nn.ReLU) #neural network for actor-critic network

        logger_kwargs = dict(output_dir='path/to/output_dir3imprecise', exp_name='kinova') #output_dir is filepath for
                                                                                # saving agent and logging progress

        ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=200, steps_per_epoch=1000, epochs=1000,
            logger_kwargs=logger_kwargs)  #runs the training process with specified hyperparameters

    elif what_to_do == 'play':
        ac = torch.load('path/to/output_dir2/ppo_model.pt') # loads the agent model
        ac.pi.eval()
        env = gym.make('kinova-v0') #make env

        for _ in range(5): #runs agent for 5 episodes
            o = env.reset() #reset env after an episode
            d = False
            while not d:
                action = ac.act(torch.tensor(o, dtype=torch.float32)) #collects action from agent model
                o, r, d, _ = env.step(action) #applys action to environment

    else:
        # This else is used for testing environment by making what_to_do = ''
        env = gym.make('kinova-v0')
        ob = env.reset()
        t = 0.000001
        while True:
            t += 0.0000001
            # action = np.array([np.random.uniform(-1, 1),
            #                    np.random.uniform(-1, 1),
            #                    np.random.uniform(-1, 1),
            #                    np.random.uniform(-1, 1)])
            action = np.array([0+t, 0+t, 0+t])
            ob, _, done, _ = env.step(action)


if __name__ == '__main__':
    main()
