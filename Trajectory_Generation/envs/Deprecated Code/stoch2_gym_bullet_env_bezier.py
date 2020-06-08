import sys, os
sys.path.append(os.path.realpath('../..'))
import numpy as np
import math
import gym
import os
from gym import utils, spaces
import pdb
import pybRL.envs.walking_controller as walking_controller
import time

import pybullet
import pybRL.envs.bullet_client as bullet_client
import pybullet_data
import matplotlib.pyplot as plt
INIT_POSITION = [0, 0, 0.29] 
INIT_ORIENTATION = [0, 0, 0, 1]
LEG_POSITION = ["fl_", "bl_", "fr_", "br_"]
KNEE_CONSTRAINT_POINT_RIGHT = [0.014, 0, 0.076] #hip
KNEE_CONSTRAINT_POINT_LEFT = [0.0,0.0,-0.077] #knee
RENDER_HEIGHT = 720 #360
RENDER_WIDTH = 960 #480 
PI = math.pi

class Stoch2Env(gym.Env):
    
    def __init__(self,
                 render = False,
                 on_rack = False,
                 gait = 'trot',
                 phase = [0,PI,PI,0],
                 action_dim = 10,
                 stairs = True):
        
        self._is_stairs = stairs
        
        self._is_render = render
        self._on_rack = on_rack
        if self._is_render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient()
        
        self._theta = 0
        self._theta0 = 0
        self._update_action_every = 1.  # update is every 50% of the step i.e., theta goes from 0 to pi/2
        self._frequency = 2.8 #change back to 1
        self._kp = 20
        self._kd = 2
        self.dt = 0.001
        self._frame_skip = 5
        self._n_steps = 0
        self._action_dim = action_dim

        # self._obs_dim = 7
        self._obs_dim = 4
     
        self.action = np.zeros(self._action_dim)
        
        self._last_base_position = [0, 0, 0]
        self._distance_limit = float("inf")

        self._xpos_previous = 0.0
        self._walkcon = walking_controller.WalkingController(gait_type=gait,
                                                             planning_space = 'polar_task_space',
                                                             left_to_right_switch = True,
                                                             frequency=self._frequency,
                                                             phase=phase)
        
        self._cam_dist = 1.0
        self._cam_yaw = 0.0
        self._cam_pitch = 0.0
    
        ## Gym env related mandatory variables
        observation_high = np.array([10.0] * self._obs_dim)
        observation_low = -observation_high
        self.observation_space = spaces.Box(observation_low, observation_high)
        
        action_high = np.array([1] * self._action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.hard_reset()
        if(self._is_stairs):
            boxHalfLength = 0.06
            boxHalfWidth = 2.5
            boxHalfHeight = 0.02
            sh_colBox = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,halfExtents=[boxHalfLength,boxHalfWidth,boxHalfHeight])
            boxOrigin = 0.15
            n_steps = 30
            for i in range(n_steps):
                block=self._pybullet_client.createMultiBody(baseMass=0,baseCollisionShapeIndex = sh_colBox,basePosition = [boxOrigin + i*2*boxHalfLength,0,boxHalfHeight + i*2*boxHalfHeight],baseOrientation=[0.0,0.0,0.0,1])
            x = 1
            
        
    def hard_reset(self):
        self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=int(300))
        self._pybullet_client.setTimeStep(self.dt/self._frame_skip)
        
        plane = self._pybullet_client.loadURDF("%s/plane.urdf" % pybullet_data.getDataPath())
        self._pybullet_client.changeVisualShape(plane,-1,rgbaColor=[1,1,1,0.9])
        self._pybullet_client.setGravity(0, 0, -9.8)
        
        #Change this to suit your path
        model_path = os.path.realpath('../..')+'/pybRL/envs/stoch_two_urdf/urdf/stoch_two_urdf.urdf'
        self.stoch2 = self._pybullet_client.loadURDF(model_path, INIT_POSITION)
        
        self._joint_name_to_id, self._motor_id_list, self._motor_id_list_obs_space = self.BuildMotorIdList()

        num_legs = 4
        for i in range(num_legs):
            self.ResetLeg(i, add_constraint=True)
    
        if self._on_rack:
            self._pybullet_client.createConstraint(
                self.stoch2, -1, -1, -1, self._pybullet_client.JOINT_FIXED,
                [0, 0, 0], [0, 0, 0], [0, 0, 0.3])
            
        self._pybullet_client.resetBasePositionAndOrientation(self.stoch2, INIT_POSITION, INIT_ORIENTATION)
        self._pybullet_client.resetBaseVelocity(self.stoch2, [0, 0, 0], [0, 0, 0])
      
        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        
    def reset(self):
        self._pybullet_client.resetBasePositionAndOrientation(self.stoch2, INIT_POSITION, INIT_ORIENTATION)
        self._pybullet_client.resetBaseVelocity(self.stoch2, [0, 0, 0], [0, 0, 0])
      
        num_legs = 4
        for i in range(num_legs):
            self.ResetLeg(i, add_constraint=False)

        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        self._n_steps = 0
              
        return self.GetObservationReset()
    
    def step(self, action, callback=None):
        action = np.clip(action, -1, 1)
        energy_spent_per_step, cost_reference, ang_data = self.do_simulation(action, n_frames = self._frame_skip, callback=callback)
        ob = self.GetObservation()
        reward,done,penalty = self._get_reward(action,energy_spent_per_step,cost_reference) 
        if done:
            self.reset()
        return ob, reward, done, ang_data

    def do_simulation(self, action, n_frames, callback=None):
        omega = 2 * math.pi * self._frequency
        self._theta = self._theta0
        energy_spent_per_step = 0
        self.action = action
        cost_reference = 0
        ii = 0
        angle_data = []
        while(self._theta - self._theta0 <= math.pi * self._update_action_every and not self._theta >= 2 * math.pi):
            theta = self._theta
            spine_des, leg_m_angle_cmd, d_spine_des, leg_m_vel_cmd= self._walkcon.transform_action_to_motor_joint_command3(theta,action)   
            self._theta = (omega * self.dt + theta)
            qpos_act = np.array(self.GetMotorAngles())
            m_angle_cmd_ext = np.array(leg_m_angle_cmd)
            m_vel_cmd_ext = np.zeros(8)
            for _ in range(n_frames):
                current_angle_data = np.concatenate(([ii],self.GetMotorAngles()))
                angle_data.append(current_angle_data)
                ii = ii + 1
                applied_motor_torque = self._apply_pd_control(m_angle_cmd_ext, m_vel_cmd_ext)
                self._pybullet_client.stepSimulation()
                joint_power = np.multiply(applied_motor_torque, self.GetMotorVelocities()) # Power output of individual actuators
                joint_power[ joint_power < 0.0] = 0.0 # Zero all the negative power terms
                energy_spent = np.sum(joint_power) * self.dt/n_frames
                energy_spent_per_step += energy_spent           
        self._theta0 = self._theta % (2* math.pi)
        self._n_steps += 1
        return energy_spent_per_step, cost_reference, angle_data
  
    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        
        base_pos, _ = self.GetBasePosAndOrientation()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=self._cam_dist,
                yaw=self._cam_yaw,
                pitch=self._cam_pitch,
                roll=0,
                upAxisIndex=2)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(
                fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
                nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
                width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
                projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        
        rgb_array = np.array(px).reshape(RENDER_WIDTH, RENDER_HEIGHT, 4)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self, pos, orientation):
        done = False
        penalty = 0
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]

        # stop episode after ten steps
        if self._n_steps >= 1000:
            done = True
            print('%s steps finished. Terminated' % self._n_steps)
            penalty = 0
        else:
            if np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.3:
                print('Oops, Robot about to fall! Terminated')
                done = True
                penalty = penalty + 0.1
            if pos[2] < 0.04:
                print('Robot was too low! Terminated')
                done = True
                penalty = penalty + 0.5
            if pos[2] > 0.3:
                print('Robot was too high! Terminated')
                done = True
                penalty = penalty + 0.6

        if done and self._n_steps <= 2:
            penalty = 3
            
        return done, penalty

    def _get_reward(self,action,energy_spent_per_step,cost_reference):
        current_base_position, current_base_orientation = self.GetBasePosAndOrientation()

        rpy = self._pybullet_client.getEulerFromQuaternion(current_base_orientation)
       # print(rpy)
        roll_penalty = np.abs(rpy[0])
        yaw_penalty = np.abs(rpy[1])
        pitch_penalty = np.abs(rpy[2])
        distance_travelled = current_base_position[0] - self._last_base_position[0] # added negative reward for staying still
        # distance_travelled = np.clip(forward_reward, -0.1, 0.1)

#         walking_velocity_reward = 10 * np.exp(-10*(0.6 - xvel)**2)
        walking_height_reward = 0.5 * np.exp(-10*(0.23 - current_base_position[2])**2)
        #print("Current base position" + str(current_base_position[2]))
        #print("Walking height" + str(walking_height_reward))

        done, penalty = self._termination(current_base_position, current_base_orientation)
#         if forward_reward >= 0:
#             reward = (distance_travelled - 0.01 * energy_spent_per_step - penalty + 0.01 * walking_height_reward - 0.0*(pitch_penalty)) 
#         else:
#             reward = forward_reward
#         print('forward reward',forward_reward)
#         print('energy per step', energy_spent_per_step)
#         print('reward',reward)
        
#         rt_start = self._walkcon.transform_action_to_rt(0, action)
#         rt_mid = self._walkcon.transform_action_to_rt(math.pi/2, action)
#         print(rt_start,rt_mid)
        
#         foot_clearance_reward = 0.5 * np.exp(-10*(0.04 - (rt_mid[0] - rt_mid[2]))**2)
#         stride_length_reward = 0.1 * np.exp(-10*(0.25 - (rt_start[1] - rt_start[3]))**2)
        # distance_travelled = np.array(current_base_position) - np.array(self._xpos_previous)
        self._xpos_previous = current_base_position[0]
        penalty = penalty + roll_penalty + yaw_penalty +pitch_penalty
#         walking_velocity_reward = 10 * np.exp(-10*(0.6 - xvel)**2)
#         walking_height_reward = 2 * np.exp(-2*(0.22 - zpos)**2)
        costreference_reward = 10 * np.exp(-2*(0 - cost_reference)**2)
#         print('height',zpos)
#         print(walking_height_reward)
        
#         rt_start, _ = self._walkcon.transform_action_to_rt(0, action)
#         rt_mid, _ = self._walkcon.transform_action_to_rt(math.pi/2, action)
#         print('r and theta start',rt_start,'r and theta mid',rt_mid)
        
#         foot_clearance_reward = 0.5 * np.exp(-10*(0.05 - (rt_mid[0] - rt_mid[2]))**2)
#         foot_clearance_reward = 0.5 * np.exp(-2*(0.23 - rt_mid[0])**2) + 0.5 * np.exp(-2*(0.18 - rt_mid[2])**2)
#         stride_length_reward = 0.5 * np.exp(-10*(1.0 - (rt_start[1] - rt_start[3]))**2) + 0.1 * np.exp(-10*(0.0 - (rt_mid[1] - rt_mid[3]))**2)
        
        # reward = distance_travelled - penalty - 0.01 * energy_spent_per_step + 0.5 * costreference_reward #+ walking_height_reward + foot_clearance_reward + stride_length_reward# + walking_velocity_reward
        # print('reward being returned in function: ', reward)
        #REMOVED PENALTIES TO LEARN OTHER GAITS
        if self._is_stairs:
            reward = distance_travelled - 0.01 * energy_spent_per_step + walking_height_reward #+ foot_clearance_reward + stride_length_reward# + walking_velocity_reward
        else:
            reward = distance_travelled - 0.01 * energy_spent_per_step #+ walking_height_reward + foot_clearance_reward + stride_length_reward# + walking_velocity_reward
        return reward, done, penalty

    def _apply_pd_control(self, motor_commands, motor_vel_commands):
        qpos_act = self.GetMotorAngles()
        qvel_act = self.GetMotorVelocities()
        applied_motor_torque = self._kp * (motor_commands - qpos_act) + self._kd * (motor_vel_commands - qvel_act)

        for motor_id, motor_torque in zip(self._motor_id_list, applied_motor_torque):
            self.SetMotorTorqueById(motor_id, motor_torque)

        return applied_motor_torque
    
    def GetObservation(self):
        pos, ori = self.GetBasePosAndOrientation()
        return np.concatenate([ori]).ravel()

    
    def GetObservationReset(self):
        """
        Resets the robot and returns the base position and Orientation with a random error
        :param : None, should be called in the reset function if an error in initial pos is desired
        :return : Initial state with an error.
        Robot starts in the same position, only it's readings have some error. 
        """
        pos, ori = self.GetBasePosAndOrientation()
        return np.concatenate([ori]).ravel()



    def GetMotorAngles(self):
        motor_ang = [self._pybullet_client.getJointState(self.stoch2, motor_id)[0] for motor_id in self._motor_id_list]
        return motor_ang
    
    def GetMotorAnglesObs(self):
        motor_ang = [self._pybullet_client.getJointState(self.stoch2, motor_id)[0] for motor_id in self._motor_id_list_obs_space]
        return motor_ang

    def GetMotorVelocities(self):
        motor_vel = [self._pybullet_client.getJointState(self.stoch2, motor_id)[1] for motor_id in self._motor_id_list]
        return motor_vel

    def GetMotorTorques(self):
        motor_torq = [self._pybullet_client.getJointState(self.stoch2, motor_id)[3] for motor_id in self._motor_id_list]
        return motor_torq
    
    def GetBasePosAndOrientation(self):
        position, orientation = (self._pybullet_client.getBasePositionAndOrientation(self.stoch2))
        return position, orientation

    def GetDesiredMotorAngles(self):
        _, leg_m_angle_cmd, _, _ = self._walkcon.transform_action_to_motor_joint_command(self._theta,self.action)
        
        return leg_m_angle_cmd


    def SetMotorTorqueById(self, motor_id, torque):
        self._pybullet_client.setJointMotorControl2(
                  bodyIndex=self.stoch2,
                  jointIndex=motor_id,
                  controlMode=self._pybullet_client.TORQUE_CONTROL,
                  force=torque)

    def BuildMotorIdList(self):
        num_joints = self._pybullet_client.getNumJoints(self.stoch2)
        joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(self.stoch2, i)
            joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]
        
        MOTOR_NAMES = [ "motor_fl_upper_hip_joint",
                        "motor_fl_upper_knee_joint", 
                        "motor_fr_upper_hip_joint",
                        "motor_fr_upper_knee_joint", 
                        "motor_bl_upper_hip_joint",
                        "motor_bl_upper_knee_joint", 
                        "motor_br_upper_hip_joint", 
                        "motor_br_upper_knee_joint"]
        
        #   WITHOUT SPINE
        # MOTOR_NAMES = [ "motor_fl_upper_hip_joint",
        #                 "motor_fl_upper_knee_joint", 
        #                 "motor_fr_upper_hip_joint",
        #                 "motor_fr_upper_knee_joint", 
        #                 "motor_bl_upper_hip_joint",
        #                 "motor_bl_upper_knee_joint", 
        #                 "motor_br_upper_hip_joint", 
        #                 "motor_br_upper_knee_joint",]
        # Even smaller workspace
        
        MOTOR_NAMES2 = [ "motor_fl_upper_hip_joint",
                        "motor_fl_upper_knee_joint",  
                        "motor_bl_upper_hip_joint",
                        "motor_bl_upper_knee_joint"]
        motor_id_list = [joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]
        motor_id_list_obs_space = [joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES2]

        return joint_name_to_id, motor_id_list, motor_id_list_obs_space
    
    def ResetLeg(self, leg_id, add_constraint):
        leg_position = LEG_POSITION[leg_id]
        self._pybullet_client.resetJointState(
                  self.stoch2,
                  self._joint_name_to_id["motor_" + leg_position + "upper_knee_joint"], # motor
                  targetValue = 0, targetVelocity=0)
        self._pybullet_client.resetJointState(
                  self.stoch2,
                  self._joint_name_to_id[leg_position + "lower_knee_joint"],
                  targetValue = 0, targetVelocity=0)
        self._pybullet_client.resetJointState(
                  self.stoch2,
                  self._joint_name_to_id["motor_" + leg_position + "upper_hip_joint"], # motor
                  targetValue = 0, targetVelocity=0)
        self._pybullet_client.resetJointState(
                  self.stoch2,
                  self._joint_name_to_id[leg_position + "lower_hip_joint"],
                  targetValue = 0, targetVelocity=0)

        if add_constraint:
            c = self._pybullet_client.createConstraint(
                  self.stoch2, self._joint_name_to_id[leg_position + "lower_hip_joint"],
                  self.stoch2, self._joint_name_to_id[leg_position + "lower_knee_joint"],
                  self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0],
                  KNEE_CONSTRAINT_POINT_RIGHT, KNEE_CONSTRAINT_POINT_LEFT)

            self._pybullet_client.changeConstraint(c, maxForce=200)

        # set the upper motors to zero
        self._pybullet_client.setJointMotorControl2(
                              bodyIndex=self.stoch2,
                              jointIndex=(self._joint_name_to_id["motor_" + leg_position + "upper_knee_joint"]),
                              controlMode=self._pybullet_client.VELOCITY_CONTROL,
                              targetVelocity=0,
                              force=0)
        self._pybullet_client.setJointMotorControl2(
                              bodyIndex=self.stoch2,
                              jointIndex=(self._joint_name_to_id["motor_"+ leg_position + "upper_hip_joint"]),
                              controlMode=self._pybullet_client.VELOCITY_CONTROL,
                              targetVelocity=0,
                              force=0)

        # set the lower joints to zero
        self._pybullet_client.setJointMotorControl2(
                              bodyIndex=self.stoch2,
                              jointIndex=(self._joint_name_to_id[leg_position + "lower_hip_joint"]),
                              controlMode=self._pybullet_client.VELOCITY_CONTROL,
                              targetVelocity=0,
                              force=0)
        self._pybullet_client.setJointMotorControl2(
                              bodyIndex=self.stoch2,
                              jointIndex=(self._joint_name_to_id[leg_position + "lower_knee_joint"]),
                              controlMode=self._pybullet_client.VELOCITY_CONTROL,
                              targetVelocity=0,
                              force=0)
        # set the spine motors to zero
        self._pybullet_client.setJointMotorControl2(
                  bodyIndex=self.stoch2,
                  jointIndex=(self._joint_name_to_id["motor_front_body_spine_joint"]),
                  controlMode=self._pybullet_client.VELOCITY_CONTROL,
                  targetVelocity=0)
        self._pybullet_client.setJointMotorControl2(
                  bodyIndex=self.stoch2,
                  jointIndex=(self._joint_name_to_id["motor_back_body_spine_joint"]),
                  controlMode=self._pybullet_client.VELOCITY_CONTROL,
                  targetVelocity=0)

    def CostReferenceGait(self,theta,q):
        return 0
    
    def GetXYTrajectory(self,action):
        rt = np.zeros((4,100))
        rtvel = np.zeros((4,100))
        xy = np.zeros((4,100))
        xyvel = np.zeros((4,100))
        
        for i in range(100):
            theta = 2*math.pi/100*i
            rt[:,i], rtvel[:,i] = self._walkcon.transform_action_to_rt(theta, action)
            
            r_ac1 = rt[0,i] 
            the_ac1 = rt[1,i] 
            r_ac2 = rt[2,i] 
            the_ac2 = rt[3,i] 
            
            xy[0,i] =  r_ac1*math.sin(the_ac1)
            xy[1,i] = -r_ac1*math.cos(the_ac1)
            xy[2,i] =  r_ac2*math.sin(the_ac2)
            xy[3,i] = -r_ac2*math.cos(the_ac2)
            
        return xy


if(__name__ == "__main__"):
    env = Stoch2Env(render=True, stairs = True)
    for i in range(20):
        # env.step(np.array([0,0,0,0,0,0,0,0,0,0]))
        env.step(np.zeros(18))
#         env.step(np.array( [ 0.06778296, -0.01940124, -0.01924977, -0.00751148, -0.03500922,  0.01891797,
#  -0.02483966, -0.01901164, -0.01536581,  0.01925358]))
        #Normalize action space between -0.024 to +0.024
#         env.step(np.array([ 0.04409455,  0.01223679, -0.04060704, -0.01334077, -0.02117415,  0.01420131,
#   0.02825101, -0.02903829,  0.02508816, -0.00322808]))
