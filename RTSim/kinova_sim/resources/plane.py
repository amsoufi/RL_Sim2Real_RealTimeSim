import pybullet as pl
import pybullet_data

class Plane:
    def __init__(self, client):
        self.client = client
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.loadURDF(fileName="plane.urdf",
                   basePosition=[0, 0, 0])
                   #physicsClientId=client)