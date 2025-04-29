import pybullet as g
import os


class Goal:
    def __init__(self, client, base):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'simplegoal.urdf')
        self.goal = self.client.loadURDF(fileName=f_name,
                                         basePosition=[base[0], base[1], base[2]])
        #                                physicsClientId=client)

        # self.client.resetBaseVelocity(self.goal,
        #                               linearVelocity=[rvel[0], rvel[1], rvel[2]])
        #                               physicsClientId=client)

    # def get_observation(self):
    #     ls = self.client.getBasePositionAndOrientation(self.goal)
    #
    #     observation = ls[0]
    #
    #     return observation
