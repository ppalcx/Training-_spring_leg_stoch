import numpy as np
import matplotlib.pyplot as plt

traj1 = np.load("elipse_desired.npy")
traj2 = np.load("elipse_observed.npy")
# traj3 = np.load("highPTraj.npy")
# traj4 = np.load("full_bezier_actual_sat10.npy")
# traj5 = np.load("full_bezier_actual.npy")


traj1  =  np.reshape(traj1,(80,2))
traj2  =  np.reshape(traj2,(80,2))
# traj3  =  np.reshape(traj3,(80,2))
# traj4  =  np.reshape(traj4,(80,2))
# traj5  =  np.reshape(traj5,(80,2))



plt.scatter(traj1[:,0], traj1[:,1], label = "desired_traj")
plt.scatter(traj2[:,0], traj2[:,1], label = "observed_traj")
# plt.scatter(traj3[:,0], traj3[:,1], label = "actual_traj P= 40")
# plt.scatter(traj4[:,0], traj4[:,1], label = "actual_traj P= 150 D = 5 Sat = 10nm")
# plt.scatter(traj5[:,0], traj5[:,1], label = "actual_traj P= 150 D = 5 Sat = 20nm")

plt.legend()
plt.show()
