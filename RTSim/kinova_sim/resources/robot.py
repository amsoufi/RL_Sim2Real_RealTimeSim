import numpy as np
import pybullet as r
import os
from numpy import inf
import math
from collections import namedtuple


class Robot:
    def __init__(self, client):
        self.client = client

        J1 = '0 0 {:.6f}'.format(np.random.uniform(0.1563, 0.1565)) #0.15643 # randomize J1 position
        J2 = '0 0.005375 {:.6f}'.format(np.random.uniform(-0.1283, -0.1285)) #-0.12838 # randomize J2 position

        f_name = os.path.join(os.path.dirname(__file__), 'kinovaGen3.urdf')
        f_name_temp = os.path.join(os.path.dirname(__file__), 'kinovaGen3_temp.urdf')

        with open(f_name, 'r') as file:
            filedata = file.read()
            filedata = filedata.replace('J1', J1)
            filedata = filedata.replace('J2', J2)

        with open(f_name_temp, 'w') as file:
            file.write(filedata)

        self.robot = self.client.loadURDF(fileName=f_name_temp,
                                          basePosition=[0, 0, 0],
                                          useFixedBase=1,
                                          flags=r.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)#|r.URDF_USE_SELF_COLLISION)
        # physicsClientId=client)
        os.remove(f_name_temp)

        # Joint indices as found by p.getJointInfo()
        self.arm_num_dofs = 3
        self.eef_id = 7

        # self.client.resetJointState(self.robot, 0, rori[0])  # , physicsClientId=client)
        # self.client.resetJointState(self.robot, 1, rori[1])  # , physicsClientId=client)
        # self.client.resetJointState(self.robot, 2, rori[2])  # , physicsClientId=client)
        # self.client.resetJointState(self.robot, 3, rori[3])  # , physicsClientId=client)
        # self.client.resetJointState(self.robot, 4, rori[4])  # , physicsClientId=client)
        # self.client.resetJointState(self.robot, 5, rori[5])  # , physicsClientId=client)

        #Unlock force constraints for Torque Control
        # self.client.setJointMotorControlArray(
        #     bodyUniqueId=self.robot,
        #     jointIndices=[1, 2, 3],
        #     controlMode=r.VELOCITY_CONTROL,
        #     forces=[0, 0, 0])
        # physicsClientId=self.client)

        # get joint info
        numJoints = self.client.getNumJoints(self.robot)
        jointInfo = namedtuple('jointInfo',
                               ['id', 'name', 'type', 'damping', 'friction', 'lowerLimit', 'upperLimit', 'maxForce',
                                'maxVelocity', 'controllable'])
        self.joints = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = self.client.getJointInfo(self.robot, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]  # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            jointDamping = info[6]
            jointFriction = info[7]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = (jointType != r.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(jointID)
                self.client.setJointMotorControl2(self.robot, jointID, r.VELOCITY_CONTROL, targetVelocity=0, force=0)
            info = jointInfo(jointID, jointName, jointType, jointDamping, jointFriction, jointLowerLimit,
                             jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            self.joints.append(info)

        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        self.arm_lower_limits = [info.lowerLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_upper_limits = [info.upperLimit for info in self.joints if info.controllable][:self.arm_num_dofs]
        self.arm_joint_ranges = [info.upperLimit - info.lowerLimit for info in self.joints if info.controllable][
                                :self.arm_num_dofs]

        for i in range(8, 19):
            self.client.changeDynamics(self.robot, i, lateralFriction=1.0, spinningFriction=1.0,
                                       rollingFriction=0.0001, frictionAnchor=True)

        # pos = (rori[0], rori[1], rori[2] + 0.16)
        # orn = r.getQuaternionFromEuler((np.pi, 0, np.pi))
        # self.joint_poses = self.client.calculateInverseKinematics(self.robot, self.eef_id, pos, orn,
        #                                                           self.arm_lower_limits,
        #                                                           self.arm_upper_limits, self.arm_joint_ranges,
        #                                                           [0, 0, 0, 0, 0, 0],
        #                                                           maxNumIterations=1000, residualThreshold=0.0001)
        # # self.rori = rori
        # # self.orn = orn
        #
        # # self.client.resetJointState(self.robot, 1, 0)  # , physicsClientId=client)
        # # self.client.resetJointState(self.robot, 2, 15*np.pi/180)  # , physicsClientId=client)
        # # self.client.resetJointState(self.robot, 3, -130*np.pi/180)  # , physicsClientId=client)
        # # self.client.resetJointState(self.robot, 4, 0)  # , physicsClientId=client)
        # # self.client.resetJointState(self.robot, 5, 55*np.pi/180)  # , physicsClientId=client)
        # # self.client.resetJointState(self.robot, 6, np.pi/2)  # , physicsClientId=client)
        #
        # self.client.resetJointState(self.robot, 1, self.joint_poses[0])
        # self.client.resetJointState(self.robot, 2, self.joint_poses[1])
        # self.client.resetJointState(self.robot, 3, self.joint_poses[2])
        # self.client.resetJointState(self.robot, 4, self.joint_poses[3])
        # self.client.resetJointState(self.robot, 5, self.joint_poses[4])
        # self.client.resetJointState(self.robot, 6, self.joint_poses[5])

        # gripper finger range
        # self.gripper_range = [0.0400, 0.085]
        self.gripper_range = [0.04, 0.085]
        # to control gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)

    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint.id: mimic_children_names[joint.name] for joint in self.joints if
                                       joint.name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = self.client.createConstraint(self.robot, self.mimic_parent_id,
                                             self.robot, joint_id,
                                             jointType=r.JOINT_GEAR,
                                             jointAxis=[0, 1, 0],
                                             parentFramePosition=[0, 0, 0],
                                             childFramePosition=[0, 0, 0])
            self.client.changeConstraint(c, gearRatio=-multiplier, maxForce=100,
                                         erp=1)  # Note: the mysterious `erp` is of EXTREME importance

    def move_gripper(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # if open_length == 0:
        #     self.client.setJointMotorControl2(self.robot, self.mimic_parent_id, r.VELOCITY_CONTROL, force=0)
        #     self.client.setJointMotorControl2(self.robot, self.mimic_parent_id, r.TORQUE_CONTROL,
        #                                       force=100)
        # else:
        #     self.client.setJointMotorControl2(self.robot, self.mimic_parent_id, r.POSITION_CONTROL,
        #                                       targetPosition=open_angle,
        #                                       force=self.joints[self.mimic_parent_id].maxForce,
        #                                       maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)
        # Control the mimic gripper joint(s)

        # self.client.setJointMotorControl2(self.robot, self.mimic_parent_id, r.VELOCITY_CONTROL,
        #                                   targetVelocity=self.joints[self.mimic_parent_id].maxVelocity,
        #                                   force=10 * self.joints[self.mimic_parent_id].maxForce)

        self.client.setJointMotorControl2(self.robot, self.mimic_parent_id, r.POSITION_CONTROL,
                                          targetPosition=open_angle,
                                          force=10 * self.joints[self.mimic_parent_id].maxForce,
                                          maxVelocity=self.joints[self.mimic_parent_id].maxVelocity)

    def open_gripper(self):
        self.move_gripper(self.gripper_range[1])

    def close_gripper(self):
        self.move_gripper(self.gripper_range[0])

    def reset_gripper(self):
        self.open_gripper()

    def get_ids(self):
        return self.robot, self.client

    def apply_action(self, action):
        # Expects action to be 6 dimensional
        # t1, t2, t3, t4, t5, t6 = action
        t1, t2, t3 = action

        # Clip torque values to reasonable values
        t1 *= 1.39
        t2 *= 1.39
        t3 *= 1.22
        # t4 *= 1.22
        # t5 *= 1.22
        # t6 *= 1.22
        # t1 = self.joint_poses[0] + t1
        # t2 = self.joint_poses[1] + t2
        # t3 = self.joint_poses[2] + t3
        #

        # pos = (self.rori[0] + t1, self.rori[1], self.rori[2])
        # joint_poses_new = self.client.calculateInverseKinematics(self.robot, self.eef_id, pos, self.orn,
        #                                                          self.arm_lower_limits,
        #                                                          self.arm_upper_limits, self.arm_joint_ranges,
        #                                                          [0, 0, 0, 0, 0, 0],
        #                                                          maxNumIterations=1000, residualThreshold=0.0001)
        # t1 = joint_poses_new[0]
        # t2 = joint_poses_new[1]
        # t3 = joint_poses_new[2]
        # t4 = joint_poses_new[3]
        # t5 = joint_poses_new[4]
        # t6 = joint_poses_new[5]

        # t1 = 0 + t1
        # t2 = 15*np.pi/180 + t2
        # t3 = -130*np.pi/180 + t3
        # t4 = 0 + t4
        # t5 = 55*np.pi/180 + t5
        # t6 = np.pi/2 + t6

        # Set the torque of the joints directly
        # self.client.setJointMotorControlArray(
        #     bodyUniqueId=self.robot,
        #     jointIndices=[1, 2, 3, 4, 5, 6],
        #     controlMode=r.TORQUE_CONTROL,
        #     forces=[t1, t2, t3, t4, t5, t6])
        self.client.setJointMotorControl2(self.robot, 1, controlMode=r.VELOCITY_CONTROL,
                                          targetVelocity=t1, force=39)

        self.client.setJointMotorControl2(self.robot, 3, controlMode=r.VELOCITY_CONTROL,
                                          targetVelocity=t2, force=39)

        self.client.setJointMotorControl2(self.robot, 5, controlMode=r.VELOCITY_CONTROL,
                                          targetVelocity=t3, force=9)

        # self.client.setJointMotorControl2(self.robot, 4, controlMode=r.POSITION_CONTROL,
        #                                   targetPosition=t4, force=9, maxVelocity=1.22)

        # self.client.setJointMotorControl2(self.robot, 5, controlMode=r.VELOCITY_CONTROL,
        #                                   targetVelocity=t5, force=9)

        # self.client.setJointMotorControl2(self.robot, 6, controlMode=r.POSITION_CONTROL,
        #                                   targetPosition=t6, force=9, maxVelocity=1.22)

        # print(t1, t2, t3,t4,t5,t6)
        # physicsClientId=self.client)

    def get_observation(self):

        # Get cartesian position of end effector

        # physicsClientId=self.client)
        ls = self.client.getLinkState(self.robot,
                                      linkIndex=7,
                                      computeLinkVelocity=1)

        # js = self.client.getJointStates(self.robot,
        #                                 jointIndices=[1, 2, 3])
        #joint info
        js = self.client.getJointStates(self.robot,
                                        jointIndices=[1, 3, 5])

        js_angle = (js[0][0], js[1][0], js[2][0])
        angle_sin = tuple([math.sin(j) for j in js_angle])
        angle_cos = tuple([math.cos(j) for j in js_angle])
        js_vel = (js[0][1], js[1][1], js[2][1])
        js_vel_rand = (js[0][1] * np.random.uniform(0.95, 1.05), js[1][1] * np.random.uniform(0.95, 1.05), js[2][1] *
                  np.random.uniform(0.95, 1.05))
        # js_torque = (js[0][3], js[1][3], js[2][3])
        # print('raw vel==', js[0][1], js[1][1], js[2][1])
        observation = ls[4] + js_angle + angle_sin + angle_cos + js_vel #use js_vel_rand to randomize velocity readings

        return observation
