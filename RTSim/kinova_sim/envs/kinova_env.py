import gym
import numpy as np
import math
import pybullet as p
import time
from pybullet_utils import bullet_client
from kinova_sim.resources.robot import Robot
from kinova_sim.resources.plane import Plane
from kinova_sim.resources.goal import Goal
from kinova_sim.resources.block import Block
# from kinova_sim.resources.gripper import Gripper
import os

import matplotlib.pyplot as plt


class KinovaEnvDM(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render=False):

        if render:
            self.client = bullet_client.BulletClient(connection_mode=p.GUI)  # p.connect(p.DIRECT)
        else:
            self.client = bullet_client.BulletClient(connection_mode=p.DIRECT)

        self.client.setPhysicsEngineParameter(enableFileCaching=0)
        # p.setRealTimeSimulation(1)
        self.np_random, _ = gym.utils.seeding.np_random()
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(-np.inf, np.inf, np.shape(self.reset()), dtype="float32")
        # self.observation_space = gym.spaces.box.Box(-np.inf, np.inf, np.shape(12,), dtype="float32")

        # Reduce length of episodes for RL algorithms
        self.client.setTimeStep(1/1000)

        self.robot = None
        self.goal = None
        self.done = False
        # self.gripper = None
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None

    def step(self, action):
        # Feed action to the car and get observation of car's state

        # print(action)
        action = np.clip(action, -1, 1)
        self.robot.apply_action(action)
        # time.sleep(0.025) #use in real-time mode
        for _ in range(25): #run for 25 steps, comment when in real-time mode

            # self.client.setTimeStep((1 / 1000) + np.random.uniform(-0.0001, 0.0001))
            self.client.stepSimulation()
        # p.stepSimulation()
        robot_ob = self.robot.get_observation()
        # goal_ob = self.goal.get_observation()

        # Compute reward as L2 change in distance to goal
        dist_to_goal = math.sqrt(((robot_ob[0] - self.goal[0]) ** 2 +
                                  (robot_ob[1] - self.goal[1]) ** 2 +
                                  (robot_ob[2] - self.goal[2]) ** 2))
        # _, misalignment = p.getAxisAngleFromQuaternion(p.getDifferenceQuaternion(robot_ob[3:7], [1, 0, 0, 0]))
        # misalignment *= 180.0 / math.pi
        reward = -dist_to_goal * 0.00002 - ((robot_ob[12]**2 + robot_ob[13]**2 + robot_ob[14]**2) * 0.000001)  # -dist_to_goal * 0.001
        # reward = - dist_to_goal * 0.0005 - misalignment * 0.000005
        # self.prev_dist_to_goal = dist_to_goal

        # Done if number of steps is exceeded
        self.i += 1
        if self.i > 200:
            self.done= True
        # Done by reaching goal
        if dist_to_goal < 0.05:
            self.done = True
            reward = 20

        # ob = np.array(robot_ob + t + gpu + cpu + mem, dtype=np.float32)
        ob = np.array(robot_ob + self.goal, dtype=np.float32)
        # print(ob)
        return ob, reward, self.done, {}

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.client.resetSimulation()
        self.client.setGravity(0, 0, -9.8)

        # Reload the plane and car
        # Plane(self.client)

        # delta_ori = 3.1416 * 0.02
        # ori1 = self.np_random.uniform(-delta_ori, delta_ori)
        # ori2 = self.np_random.uniform(-delta_ori, delta_ori) + 1.0
        # ori3 = self.np_random.uniform(-delta_ori, delta_ori) - 1.0
        # ori4 = self.np_random.uniform(-delta_ori, delta_ori) - 1.0
        # ori5 = self.np_random.uniform(-delta_ori, delta_ori)
        # ori6 = self.np_random.uniform(-delta_ori, delta_ori)

        # self.ori = (ori1, ori2, ori3)

        radius = 0.4
        # # angle_polar = self.np_random.uniform(0.05 * 3.1416, 0.45 * 3.1416)
        angle_azimuth = self.np_random.uniform(-3.1416 / 2.0, 3.1416 / 2.0)
        #
        x = 0.4 #radius * math.cos(angle_azimuth)
        y = 0.2 #radius * math.sin(angle_azimuth)
        z = 0.5

        # x_ob = 0.54
        # y_ob = 0.0
        # z_ob = 0.0

        # x_push = 0.54
        # y_push = 0.0
        # z_push = 0.05
        # self.push = (x_push, y_push, z_push)

        # v = (-np.sign(x)*self.np_random.uniform(0, 0.005),
        #      -np.sign(y)*self.np_random.uniform(0, 0.005),
        #      self.np_random.uniform(-0.002, 0.002))

        self.goal = (x, y, z)  # make goalp when dynamic
        # self.block = (x_ob, y_ob, z_ob)
        # self.goalv = v
        self.done = False
        self.robot = Robot(self.client)
        self.robot.reset_gripper()

        # Visual element of the goal
        # Goal(self.client, self.goal, self.goalv)
        Goal(self.client, self.goal)
        # Block(self.client, self.block)

        # Get observation to return
        robot_ob = self.robot.get_observation()

        # goal_ob = self.goal.get_observation()
        self.i = 0

        # robot_id, _ = self.robot.get_ids()
        # self.gripper = Gripper(self.client, robot_id)
        # self.gripper.close_gripper()

        return np.array(robot_ob + self.goal, dtype=np.float32)

    def render(self, mode='human'):
        return

        # if self.rendered_img is None:
        #     self.rendered_img = plt.imshow(np.zeros((1000, 1000, 4)))
        #
        # # Base information
        # robot_id, client_id = self.robot.get_ids()
        # proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
        #                                            nearVal=0.01, farVal=100, physicsClientId=client_id)
        # pos, ori = [list(l) for l in
        #             p.getBasePositionAndOrientation(robot_id, client_id)]
        # newpos = [1,2.5,1]
        # pos1 = []
        # for j in range(3):
        #     pos1.append(pos[j]+newpos[j])
        #
        # # Rotate camera direction
        # rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        # camera_vec = np.matmul(rot_mat, [0, 0, 0])
        # up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        # view_matrix = p.computeViewMatrix(pos1, pos + camera_vec, up_vec, physicsClientId=client_id)
        #
        # # Display image
        # frame = p.getCameraImage(1000, 1000, view_matrix, proj_matrix, physicsClientId=client_id)[2]
        # frame = np.reshape(frame, (1000, 1000, 4))
        # self.rendered_img.set_data(frame)
        # plt.draw()
        # plt.pause(.00001)

    def close(self):
        self.client.disconnect()
