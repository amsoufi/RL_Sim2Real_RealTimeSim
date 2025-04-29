import torch
import gym
import kinova_sim
import numpy as np
import torch
from ppo import ppo
import gym


def main():
    what_to_do = 'play'

    if what_to_do == 'play':
        ac = torch.load('Agents/ppo_model_RTR.pt')
        ac.pi.eval()
        env = gym.make('kinova-v0')

        for _ in range(5):
            o = env.reset()
            d = False
            while not d:
                action = ac.act(torch.tensor(o, dtype=torch.float32))
                o, r, d, _ = env.step(action)

    else:
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
