import sys, os
sys.path.append(os.path.realpath('../'))
import numpy as np
import gym
import os
from gym import utils, spaces
import pdb
import envs.walking_controller as walking_controller
import time
import math

import pybullet
import envs.bullet_client as bullet_client
import pybullet_data
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from collections import deque

INIT_POSITION = [0, 0, 0.29]
INIT_ORIENTATION = [0, 0, 0, 1]
LEG_POSITION = ["fl_", "bl_", "fr_", "br_"]
KNEE_CONSTRAINT_POINT_RIGHT = [0.014, 0, 0.0434] #hip
KNEE_CONSTRAINT_POINT_LEFT = [0.0,0.0,-0.0434] #knee
RENDER_HEIGHT = 720 #360
RENDER_WIDTH = 960 #480
PI = np.pi
no_of_points = 100

def constrain_theta(theta):
	theta = np.fmod(theta, 2*no_of_points)
	if(theta < 0):
		theta = theta + 2*no_of_points
	return theta
class Stoch2Env(gym.Env):

	def __init__(self,
				 render = False,
				 on_rack = False,
				 gait = 'trot',
				 phase = [0,no_of_points,no_of_points,0],  # what is phase?
				 action_dim = 10,
				 obs_dim = 14,
				 scale = 1.0,
				 roc = 0.3,
				 stairs = True):   #stairs?

		self._is_stairs = stairs
		self.scale = scale #INITIAL ONE FOR TRAINING
		self._is_render = render
		self._on_rack = on_rack
		if self._is_render:
			self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
		else:
			self._pybullet_client = bullet_client.BulletClient()

		self._theta = 0
		self._theta0 = 0
		self._update_action_every = 1/20  # update is every 50% of the step i.e., theta goes from 0 to pi/2
		self._frequency = 2.5 #change back to 1 .??
		# self._frequency = 2.8 #change back to 1
		self.frequency_weight = 1
		self.prev_yaw = 0
		self._kp = 150
		self._kd = 5
		self.dt = 0.005 # LET ME CHANGE, was 0.001
		self._frame_skip = 25 # Working ratio is 5* self._dt
		self._n_steps = 0
		self._action_dim = action_dim

		self._obs_dim = obs_dim

		self.action = np.zeros(self._action_dim)

		self._last_base_position = [0, 0, 0]
		self._distance_limit = float("inf")

		self._xpos_previous = 0.0
		self._walkcon = walking_controller.WalkingController(gait_type=gait, phase=phase, scale = self.scale)

		self._cam_dist = 1.0  #cam?
		self._cam_yaw = 0.0
		self._cam_pitch = 0.0

		self.avg_vel_per_step = 0
		self.avg_omega_per_step = 0

		self.linearV = 0
		self.angV = 0

		self.prev_rpy = np.array([0,0,0])

		self.obs_queue = deque([0]*10, maxlen=10) 			#observation queue

		self.termination_steps = 200

		self.radius = roc
		## Gym env related mandatory variables

		observation_high = np.array([10.0] * self._obs_dim) #obs high?
		observation_low = -observation_high
		self.observation_space = spaces.Box(observation_low, observation_high)

		action_high = np.array([1] * self._action_dim)
		self.action_space = spaces.Box(-action_high, action_high)
		self.hard_reset()
		if(self._is_stairs):  #stairs?
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

		model_path = os.path.realpath('../')+'/envs/stoch_two__abduction_spring_urdf/urdf/stoch_two__abduction_spring_urdf.urdf'
		self.stoch2 = self._pybullet_client.loadURDF(model_path, INIT_POSITION)

		self._joint_name_to_id, self._motor_id_list, self._motor_id_list_obs_space = self.BuildMotorIdList()

		num_legs = 4
		for i in range(num_legs):
			self.ResetLeg(i, add_constraint=True)
		self.ResetPoseForAbd()
		if self._on_rack:                                  # ??
			self._pybullet_client.createConstraint(
				self.stoch2, -1, -1, -1, self._pybullet_client.JOINT_FIXED,
				[0, 0, 0], [0, 0, 0], [0, 0, 0.3])

		self._pybullet_client.resetBasePositionAndOrientation(self.stoch2, INIT_POSITION, INIT_ORIENTATION)
		self._pybullet_client.resetBaseVelocity(self.stoch2, [0, 0, 0], [0, 0, 0])

		self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
		#self.SetFootFriction(0.5)

	def reset_standing_position(self):
		num_legs = 4

		for i in range(num_legs):
			self.ResetLeg(i, add_constraint=False, standstilltorque=10)
		self.ResetPoseForAbd()

		# Conditions for standstill
		for i in range(1):
			self._pybullet_client.stepSimulation()

		for i in range(num_legs):
			self.ResetLeg(i, add_constraint=False, standstilltorque=0)

	def reset(self):
		self._theta = 0
		self._last_base_position = [0, 0, 0]
		self._pybullet_client.resetBasePositionAndOrientation(self.stoch2, INIT_POSITION, INIT_ORIENTATION)
		self._pybullet_client.resetBaseVelocity(self.stoch2, [0, 0, 0], [0, 0, 0])               #????
		self.reset_standing_position()
		self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
		self._n_steps = 0

		return self.GetObservationReset()

	def transform_action(self, action):

		action = np.clip(action, -1, 1) #Clip (limit) the values in an array.
		action = (action + 1)/2					#weights are always possitive

		for i in range(action.shape[0]):
			if action[i] < 0.01:
				action[i] = 0.01

		return action

	def step(self, action, callback=None):
		action = self.transform_action(action)
		energy_spent_per_step, cost_reference, ang_data = self.do_simulation(action, n_frames = self._frame_skip, callback=callback)
		ob = self.GetObservation()  #####
		reward = self._get_reward(action,energy_spent_per_step,cost_reference) ########
		return ob, reward,False, ang_data

	def CurrentVelocities(self):
	
		current_w = self.GetBaseAngularVelocity()[2]
		current_v = self.GetBaseLinearVelocity()
		radial_v = math.sqrt(current_v[0]**2 + current_v[1]**2)
		return radial_v, current_w

	

	def do_simulation(self, action, n_frames, callback=None):
		omega = 2 * no_of_points * self._frequency  #Maybe remove later
		energy_spent_per_step = 0
		self.action = action
		cost_reference = 0  ### cost ref??
		ii = 0
		angle_data = []      #??
		counter = 0
		sum_V = 0
		sum_W = 0
		current_theta = self._theta
		while(np.abs(self._theta - current_theta) <= no_of_points * self._update_action_every):
			current_angle_data = np.concatenate(([self._theta],self.GetMotorAngles()))
			angle_data.append(current_angle_data)
			abd_m_angle_cmd, leg_m_angle_cmd, d_spine_des, leg_m_vel_cmd= self._walkcon.transform_action_to_motor_joint_command_bezier(self._theta,action, self.radius)
			self._theta = constrain_theta(omega * self.dt + self._theta)
			# print(self._theta)
			qpos_act = np.array(self.GetMotorAngles())
			m_angle_cmd_ext = np.array(leg_m_angle_cmd + abd_m_angle_cmd)
			m_vel_cmd_ext = np.zeros(12)
			counter = counter+1
			current_v, current_w = self.CurrentVelocities()
			sum_V = sum_V + current_v
			sum_W = sum_W + current_w
			for _ in range(n_frames): #n_frame??
				ii = ii + 1
				applied_motor_torque = self._apply_pd_control(m_angle_cmd_ext, m_vel_cmd_ext)
				self._pybullet_client.stepSimulation()
				joint_power = np.multiply(applied_motor_torque, self.GetMotorVelocities()) # Power output of individual actuators
				joint_power[ joint_power < 0.0] = 0.0 # Zero all the negative power terms
				energy_spent = np.sum(joint_power) * self.dt/n_frames
				energy_spent_per_step += energy_spent

		self.avg_vel_per_step = sum_V/counter
		self.avg_omega_per_step = sum_W/counter
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

	def _termination(self, pos, orientation, RPY):
		done = False
		penalty = 0
		rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
		local_up = rot_mat[6:]

		# stop episode after ten steps
		if self._n_steps >= self.termination_steps:
			done = True
			# print('%s steps finished. Terminated' % self._n_steps)
			penalty = 0
		else:
			if abs(RPY[0]) > 45 * PI / 180:
				print('Oops, Robot going to fall sideways! Terminated')
				done = True
				penalty = penalty + 0.1

			if abs(RPY[1]) > 35 * PI / 180:
				print('Oops, Robot doing wheely! Terminated')
				done = True
				penalty = penalty + 0.1

			if pos[2] > 0.5:
				print('Robot was too high! Terminated')
				done = True
				penalty = penalty + 0.6

		if done and self._n_steps <= 2:
			penalty = 3

		return done, penalty

	def _get_reward(self, action, energy_spent_per_step, cost_reference):#   cost refernece

		pos, ori = self.GetBasePosAndOrientation()

		RPY = self._pybullet_client.getEulerFromQuaternion(ori)
		RPY = np.round(RPY, 4)


		roll_reward = np.exp(-50 * (RPY[0] ** 2))
		pitch_reward = np.exp(-50 * (RPY[1] ** 2))

		x = pos[0]
		x_l = self._last_base_position[0]
		self._last_base_position = pos

		step_distance_x = (x - x_l)

		done, penalty = self._termination(pos, ori, RPY)
		if done:
			reward = 0
		else:
			reward = round(pitch_reward, 4) + round(roll_reward, 4) + 50 * round(step_distance_x, 4) - 0.1*energy_spent_per_step
			#print(0.01*energy_spent_per_step)
		return reward

	def _apply_pd_control(self, motor_commands, motor_vel_commands):
		qpos_act = self.GetMotorAngles()
		qvel_act = self.GetMotorVelocities()
		applied_motor_torque = self._kp * (motor_commands - qpos_act) + self._kd * (motor_vel_commands - qvel_act)
#
		applied_motor_torque = np.clip(np.array(applied_motor_torque), -7, 7)
		applied_motor_torque = applied_motor_torque.tolist()#list?

		for motor_id, motor_torque in zip(self._motor_id_list, applied_motor_torque):
			self.SetMotorTorqueById(motor_id, motor_torque)
		return applied_motor_torque

	def GetObservation(self):

		pos, ori = self.GetBasePosAndOrientation()
		RPY = self._pybullet_client.getEulerFromQuaternion(ori)
		RPY = np.around(RPY, decimals = 4)


		Ang_Vel = self.GetBaseAngularVelocity()
		Ang_Vel = np.around(Ang_Vel, decimals=4)

		motor_angles = self.GetMotorAngles()

		obs = np.concatenate((motor_angles,RPY[:2])).ravel()
		self.prev_rpy = RPY
		return obs

	# def GetObservation(self):
	# 	pos, ori = self.GetBasePosAndOrientation()
	# 	RPY = self._pybullet_client.getEulerFromQuaternion(ori)
	# 	RPY = np.round(RPY, 5)
	#
	# 	self.obs_queue.append(RPY[0])	#appending current roll
	# 	self.obs_queue.append(RPY[1])	#appending current pitch
	#
	# 	obs = np.array(list(self.obs_queue))
	# 	return obs

	# def GetObservationReset(self):
	# 	pos, ori = self.GetBasePosAndOrientation()
	# 	RPY = self._pybullet_client.getEulerFromQuaternion(ori)
	# 	RPY = np.round(RPY, 5)
	#
	# 	self.obs_queue.append(RPY[0])  # appending current roll
	# 	self.obs_queue.append(RPY[1])  # appending current pitch
	#
	# 	obs = np.array(list(self.obs_queue))
	# 	return obs

	def GetObservationReset(self):

		pos, ori = self.GetBasePosAndOrientation()
		RPY = self._pybullet_client.getEulerFromQuaternion(ori)
		RPY = np.around(RPY, decimals = 4)


		Ang_Vel = self.GetBaseAngularVelocity()
		Ang_Vel = np.around(Ang_Vel, decimals=4)

		motor_angles = self.GetMotorAngles()

		obs = np.concatenate((motor_angles, RPY[:2])).ravel()
		self.prev_rpy = RPY
		return obs

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

	def GetBaseAngularVelocity(self):
		basevelocity= self._pybullet_client.getBaseVelocity(self.stoch2)
		return basevelocity[1] #world AngularVelocity vec3, list of 3 floats

	def GetBaseLinearVelocity(self):
		basevelocity= self._pybullet_client.getBaseVelocity(self.stoch2)
		return basevelocity[0] #world linear Velocity vec3, list of 3 floats

	def SetFootFriction(self, foot_friction):
		FOOT_LINK_ID = [3,8,14,19]
		for link_id in FOOT_LINK_ID:
			self._pybullet_client.changeDynamics(
			self.stoch2, link_id, lateralFriction=foot_friction)

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

		#adding abduction
		MOTOR_NAMES = [ "motor_fl_upper_hip_joint",
						"motor_fl_upper_knee_joint",
						"motor_fr_upper_hip_joint",
						"motor_fr_upper_knee_joint",
						"motor_bl_upper_hip_joint",
						"motor_bl_upper_knee_joint",
						"motor_br_upper_hip_joint",
						"motor_br_upper_knee_joint",
						"motor_front_left_abd_joint",
						"motor_front_right_abd_joint",
						"motor_back_left_abd_joint",
						"motor_back_right_abd_joint"]
		#assign 4 more motors


		MOTOR_NAMES2 = [ "motor_fl_upper_hip_joint",
						"motor_fl_upper_knee_joint",
						"motor_bl_upper_hip_joint",
						"motor_bl_upper_knee_joint"]
		motor_id_list = [joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]
		motor_id_list_obs_space = [joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES2]

		return joint_name_to_id, motor_id_list, motor_id_list_obs_space

	def ResetLeg(self, leg_id, add_constraint, standstilltorque = 10):
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
				  self._pybullet_client.JOINT_POINT2POINT, [0, 0, 0],                                 #??
				  KNEE_CONSTRAINT_POINT_RIGHT, KNEE_CONSTRAINT_POINT_LEFT)

			self._pybullet_client.changeConstraint(c, maxForce=200)

		# set the upper motors to zero
		self._pybullet_client.setJointMotorControl2(
							  bodyIndex=self.stoch2,
							  jointIndex=(self._joint_name_to_id["motor_" + leg_position + "upper_knee_joint"]),
							  controlMode=self._pybullet_client.VELOCITY_CONTROL,
							  targetVelocity=0,
							  force=standstilltorque)
		self._pybullet_client.setJointMotorControl2(
							  bodyIndex=self.stoch2,
							  jointIndex=(self._joint_name_to_id["motor_"+ leg_position + "upper_hip_joint"]),
							  controlMode=self._pybullet_client.VELOCITY_CONTROL,
							  targetVelocity=0,
							  force=standstilltorque)

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

	def ResetPoseForAbd(self):
		self._pybullet_client.resetJointState(
			self.stoch2,
			self._joint_name_to_id["motor_front_left_abd_joint"],
			targetValue = 0, targetVelocity = 0)
		self._pybullet_client.resetJointState(
			self.stoch2,
			self._joint_name_to_id["motor_front_right_abd_joint"],
			targetValue = 0, targetVelocity = 0)
		self._pybullet_client.resetJointState(
			self.stoch2,
			self._joint_name_to_id["motor_back_left_abd_joint"],
			targetValue = 0, targetVelocity = 0)
		self._pybullet_client.resetJointState(
			self.stoch2,
			self._joint_name_to_id["motor_back_right_abd_joint"],
			targetValue = 0, targetVelocity = 0)

		#Set control mode for each motor and initial conditions
		self._pybullet_client.setJointMotorControl2(
			bodyIndex = self.stoch2,
			jointIndex = (self._joint_name_to_id["motor_front_left_abd_joint"]),
			controlMode = self._pybullet_client.VELOCITY_CONTROL,
			force = 0,
			targetVelocity = 0
		)
		self._pybullet_client.setJointMotorControl2(
			bodyIndex = self.stoch2,
			jointIndex = (self._joint_name_to_id["motor_front_right_abd_joint"]),
			controlMode = self._pybullet_client.VELOCITY_CONTROL,
			force = 0,
			targetVelocity = 0
		)
		self._pybullet_client.setJointMotorControl2(
			bodyIndex = self.stoch2,
			jointIndex = (self._joint_name_to_id["motor_back_left_abd_joint"]),
			controlMode = self._pybullet_client.VELOCITY_CONTROL,
			force = 0,
			targetVelocity = 0
		)
		self._pybullet_client.setJointMotorControl2(
			bodyIndex = self.stoch2,
			jointIndex = (self._joint_name_to_id["motor_back_right_abd_joint"]),
			controlMode = self._pybullet_client.VELOCITY_CONTROL,
			force = 0,
			targetVelocity = 0
		)

	def GetXYTrajectory(self,action):
		rt = np.zeros((4,100))
		rtvel = np.zeros((4,100))
		xy = np.zeros((4,100))
		xyvel = np.zeros((4,100))

		for i in range(100):
			theta = 2*no_of_points/100*i
			rt[:,i], rtvel[:,i] = self._walkcon.transform_action_to_rt(theta, action)

			r_ac1 = rt[0,i]
			the_ac1 = rt[1,i]
			r_ac2 = rt[2,i]
			the_ac2 = rt[3,i]

			xy[0,i] =  r_ac1*np.sin(the_ac1)
			xy[1,i] = -r_ac1*np.cos(the_ac1)
			xy[2,i] =  r_ac2*np.sin(the_ac2)
			xy[3,i] = -r_ac2*np.cos(the_ac2)

		return xy


	def simulate_command(self, m_angle_cmd_ext, m_vel_cmd_ext, callback=None):
		"""
		Provides an interface for testing, you can give external position/ velocity commands and see how the robot behaves
		"""
		omega = 2 * no_of_points * self._frequency
		angle_data = []
		counter = 0
		while(np.abs(omega*self.dt*counter) <= no_of_points * self._update_action_every):
			self._theta = constrain_theta(omega * self.dt + self._theta)
			for _ in range(self._frame_skip):
				applied_motor_torque = self._apply_pd_control(m_angle_cmd_ext, m_vel_cmd_ext)
				self._pybullet_client.stepSimulation()
		return 0

	def apply_trajectory2d(self, fl_traj, fr_traj, bl_traj, br_traj, fl_phi, fr_phi, bl_phi, br_phi):
		"""
		Provides an interface for testing, you can give external xy trajectories and see how the robot behaves, the trajectory should be
		centered at 0 (maybe). Provide trajectory for fl, fr, bl, br in that order
		"""
		self._theta = 0
		omega = 2 * no_of_points * self._frequency
		while True:
			abd_m_angle_cmd, leg_m_angle_cmd, d_spine_des, leg_m_vel_cmd= self._walkcon.run_traj2d(self._theta,
			[fl_traj,fl_phi], [fr_traj, fr_phi], [bl_traj, bl_phi], [br_traj, br_phi])
			self._theta = constrain_theta(omega * self.dt + self._theta)
			qpos_act = np.array(self.GetMotorAngles())
			m_angle_cmd_ext = np.array(leg_m_angle_cmd + abd_m_angle_cmd)
			m_vel_cmd_ext = np.zeros(12)
			for _ in range(self._frame_skip):
				applied_motor_torque = self._apply_pd_control(m_angle_cmd_ext, m_vel_cmd_ext)
				self._pybullet_client.stepSimulation()
		pass
	def update_radius(self, rad):
		self.radius = rad
	def update_scale(self, scale):
		self._walkcon.scale = scale
	def update_comy(self):
		pos, ori = self.GetBasePosAndOrientation()
		yaw = quaternionToEuler(ori)
		self._walkcon.comy = yaw

	def do_trajectory(self ):
		pass

def quaternionToEuler(q):
    siny_cosp = 2 * (q[3]* q[2] + q[0] * q[1])
    cosy_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw

if(__name__ == "__main__"):

	env = Stoch2Env(render=True, stairs = False,on_rack=False, gait = 'trot', roc = 1, scale = 1)
	action = [1.0, 1.0, 1.0, 1.0, 1.0] *2
	states = []
	env.reset()
	angles = []
	t_r = 0
	#env.hard_reset()
	for i in np.arange(200):
		#env._pybullet_client.stepSimulation()
		#print(env.GetMotorAngles())
		obs, r, _, angle = env.step(action)
		t_r += r
		print(obs)
	print("Total Reward is {}".format(t_r))

