import sys, os
sys.path.append(os.path.realpath('../'))
# sys.path.append('/home/sashank/stoch2_gym_env')


from utils.logger import DataLog
import utils.make_train_plots as plotter
import envs.stoch2_gym_bullet_env_bezier as e

import pybullet as p
import numpy as np
import time
PI = np.pi

walk = [0, PI, PI/2, 3*PI/2]
pace = [0, PI, 0, PI]
bound = [0, 0, PI, PI]
trot = [0, PI, PI , 0]
custom_phase = [0, PI, PI+0.1 , 0.1]

env = e.Stoch2Env(render=True, stairs=False, on_rack=False, gait="trot", roc = 10000, scale = 1)
state = env.reset()
policy = np.load("stable_walk_3Jun/iterations/best_policy.npy")
steps = 0
t_r = 0
while steps<200:
  action = policy.dot(state)
  state, r,_,_ = env.step(action)
  t_r +=r
  steps =steps + 1
print("Total_reward "+ str(t_r))
