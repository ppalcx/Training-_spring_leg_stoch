import sys, os
sys.path.append(os.path.realpath('../..'))
# sys.path.append('/home/sashank/stoch2_gym_env')


from pybRL.utils.logger import DataLog
import pybRL.utils.make_train_plots as plotter 
import pybRL.envs.stoch2_gym_bullet_env_bezier as e

import pybullet as p
import numpy as np
import time
PI = np.pi

radius = [0.01,0.3,0.5,0.7, 1]
scale = [-1, -0.75, -0.50, -0.25, 0.25, 0.50, 0.75, 1]


inputs = []
outputs = []

final_str = ""
for s in scale:
  for r in radius:
    inputs.append(np.array([r,s]))
    if s < 0:
      folder_name = "Rad" + str(r) + "_Sneg" + str(abs(s))
      folder_name_neg = "Radneg" + str(r) + "_Sneg" + str(abs(s))
    else:
      folder_name = "Rad" + str(r) + "_S" + str(abs(s))
      folder_name_neg = "Radneg" + str(r) + "_S" + str(abs(s))


    folder_path = "/home/abhik/ID/ICRA2020_ARS/pybRL/Training Results/" + folder_name

    policy = np.load(folder_path+"/iterations/best_policy.npy")

    env = e.Stoch2Env(render = False, stairs = False, on_rack=False, gait = "trot" , roc = r, scale = s)
    state = env.reset()

    steps = 0
    t_r = 0
    while steps<11:
      action = policy.dot(state)
      action = np.clip(action, -1, 1)

      if steps == 9:
        BR_FL = action
      elif steps == 10:
        FR_BL = action
        outputs.append(np.concatenate((FR_BL,BR_FL),axis = 0))

      state, r,_,_ = env.step(action)
      steps =steps + 1



    FR_action = (FR_BL[:6] + 1)/2 
    BL_action = (FR_BL[6:]+ 1)/2

    BR_action = (BR_FL[:6]+1)/2
    FL_action = (BR_FL[6:]+1)/2

    for i in range (FR_action.shape[0]):
      if FR_action[i] == 0:
        FR_action[i] = 1e-2
    for i in range (FL_action.shape[0]):
      if FL_action[i] == 0:
        FL_action[i] = 1e-2
    for i in range (BR_action.shape[0]):
      if BR_action[i] == 0:
        BR_action[i] = 1e-2
    for i in range (BL_action.shape[0]):
      if BL_action[i] == 0:
        BL_action[i] = 1e-2
    FR_action_neg = FL_action
    FL_action_neg = FR_action
    BL_action_neg = BR_action
    BR_action_neg = BL_action

    FL_Name = folder_name + "_fl"
    FL_Name = FL_Name.replace(".","")

    BL_Name = folder_name + "_bl"
    BL_Name = BL_Name.replace(".","")

    FR_Name = folder_name + "_fr"
    FR_Name = FR_Name.replace(".","")

    BR_Name = folder_name + "_br"
    BR_Name = BR_Name.replace(".","")

    FL_Name_neg = folder_name_neg + "_fl"
    FL_Name_neg = FL_Name_neg.replace(".","")

    BL_Name_neg = folder_name_neg + "_bl"
    BL_Name_neg = BL_Name_neg.replace(".","")

    FR_Name_neg = folder_name_neg + "_fr"
    FR_Name_neg = FR_Name_neg.replace(".","")

    BR_Name_neg = folder_name_neg + "_br"
    BR_Name_neg = BR_Name_neg.replace(".","")


    action_list = [FR_action,FR_action_neg, FL_action,FL_action_neg, BR_action,BR_action_neg, BL_action,BL_action_neg]
    leg_name = [FR_Name,FR_Name_neg, FL_Name,FL_Name_neg, BR_Name,BR_Name_neg, BL_Name,BL_Name_neg]
    idx = 0
    for a in action_list:
      str_action = "float " + leg_name[idx]+"[] = {"
      idx+=1
      for x in a:
        str_action = str_action +str(round(x,3)) + ","
      str_action = str_action[:-1]
      str_action = str_action +"};\n"
      final_str = final_str + str_action
    final_str += "\n"


radius = ["Radneg1","Radneg07","Radneg05", "Radneg03","Radneg001","Rad001", "Rad03", "Rad05", "Rad07", "Rad1"]
scale = ["Sneg1", "Sneg075", "Sneg05", "Sneg025", "S025", "S05", "S075", "S1"]



leg = ["fl", "fr", "bl", "br"]
weight_str = ""
for i in leg:
  weight_str = weight_str+"float* weight_array_"+i+"[]={\n"
  for j in radius:
    for k in scale:
      weight_str = weight_str +j+"_"+k+"_"+i+","
    weight_str+="\n"
  weight_str+= "};\n"
final_str+=weight_str

np.save("inputs3.npy", inputs)
np.save("outputs3.npy", outputs)

with open("allweights_incld1_3.txt","w") as fl:
  fl.write(final_str)