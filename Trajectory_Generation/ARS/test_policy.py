import sys, os
sys.path.append(os.path.realpath('../..'))
# sys.path.append('/home/sashank/stoch2_gym_env')


from pybRL.utils.logger import DataLog
import pybRL.utils.make_train_plots as plotter 
import pybRL.envs.stoch2_gym_bullet_env_bezier as e

import pybullet as p
import numpy as np
import time

import os



PI = np.pi

walk = [0, PI, PI/2, 3*PI/2]
pace = [0, PI, 0, PI]
bound = [0, 0, PI, PI]
trot = [0, PI, PI , 0]
custom_phase = [0, PI, PI+0.1 , 0.1]
radius = [0.01]
#scale = [-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1]
scale = [ 0.25, 0.5, 0.75, 1]

for s in scale:

  if s >0:
    file_name = "Rad0.01_S" + str(s)
  else:
    file_name = "Rad0.01_Sneg" + str(abs(s))
  folder_name = "OnSpot/" + file_name + "/iterations/best_policy.npy"


  env = e.Stoch2Env(render = False, stairs = False, on_rack=False, gait = "trot" , roc = 0.01, scale = s)
  state = env.reset()

  policy = np.load(folder_name)
  steps = 0
  t_r = 0
  while steps<8:
    action = policy.dot(state)
    action = np.clip(action, -1, 1)
    state, r,_,_ = env.step(action)
    action = np.reshape(action,24)
    if steps == 6:
      for i in range(action.shape[0]):
        action[i] = (action[i]+1)/2
        if action[i] == 0:
          action[i] = 1e-2
      print(r, s)
      action = np.around(action, decimals=3)
      action = action.tolist()
      actionfr = str(action[:6])
      actionfr = actionfr.replace("[", "{")
      actionfr = actionfr.replace("]", "}")

      actionbr = str(action[12:18])
      actionbr = actionbr.replace("[", "{")
      actionbr = actionbr.replace("]", "}")

      actionfl = str(action[6:12])
      actionfl = actionfl.replace("[", "{")
      actionfl = actionfl.replace("]", "}")

      actionbl = str(action[18:24])
      actionbl = actionbl.replace("[", "{")
      actionbl = actionbl.replace("]", "}")

      file_name = file_name.replace(".","")
      with open('./OnSpot_output.txt', 'a') as f1:
        f1.write("float " + file_name + "_fr []  = " + str(actionfr) +";" + os.linesep)
        f1.write("float " + file_name + "_br []  = " + str(actionbr) +";" + os.linesep)
        f1.write("float " + file_name + "_fl []  = " + str(actionfl) +";" + os.linesep)
        f1.write("float " + file_name + "_bl []  = " + str(actionbl) +";" + os.linesep)


    t_r +=r
    steps =steps + 1
  print("Total_reward "+ str(t_r))