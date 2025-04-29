import pybullet as p
import os


class Block:
    def __init__(self, client, bbase):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'cube.urdf')
        # self.block = self.client.loadURDF(fileName=f_name,
        #                                   basePosition=[bbase[0], bbase[1], bbase[2]])
        self.block = self.client.loadURDF(fileName=f_name,
                                          basePosition=[bbase[0], bbase[1], bbase[2]])  # mass is also increased from 0.05 to 0.5 in the .urdf