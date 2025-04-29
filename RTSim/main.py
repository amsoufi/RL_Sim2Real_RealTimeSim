import tensorflow as tf
import torch
import gym
import kinova_sim
import numpy as np

import torch
from ppo import ppo

def main():
    what_to_do = 'train' #to train new agent put 'train' to run trained agent put 'play'

    if what_to_do == 'train':
        netarc = [128, 128, 64] #network architecture for actor-critic neural network
        repeat = 1 #training cycles

        for i in range(repeat):
            env_fn = lambda: gym.make('kinova-v0') #env function

            ac_kwargs = dict(hidden_sizes=netarc, activation=torch.nn.ReLU) #create actor-critic network

            logger_kwargs = dict(output_dir='data2/to/netarc' + str(i + 1), exp_name='kinova') #filepath for saving
                                                                                        #agents and logging progress

            ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=200, steps_per_epoch=1000, epochs=1000,
                logger_kwargs=logger_kwargs) #runs the training process with specified hyperparameters

    elif what_to_do == 'play':
        ac = torch.load('data3/to/netarc1/ppo_model.pt') # loads the agent model
        ac.pi.eval()
        env = gym.make('kinova-v0', render=True) # make env with rendering

        for _ in range(5): #runs agent for 5 episodes
            o = env.reset()
            d = False
            while not d:
                action = ac.act(torch.tensor(o, dtype=torch.float32))  #collects action from agent model
                o, r, d, _ = env.step(action) #applys action to environment
                # print(o)

    else:
        # this is for testing the env
        env = gym.make('kinova-v0')
        for _ in range(5):
            ob = env.reset()
            t = 0
            while t < 400:
                t += 1
                action = np.array([1, 0.1, 0])

                # action = np.array([0+t, 0+t, 0+t])

                ob, _, done, _ = env.step(action)


if __name__ == '__main__':
    main()
