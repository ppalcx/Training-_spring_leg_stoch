# ### Walking controller
# Written by Shishir Kolathaya shishirk@iisc.ac.in
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for realizing walking controllers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from dataclasses import dataclass
from collections import namedtuple
import os
import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt
PI = np.pi
no_of_points=100
@dataclass
class leg_data:
    name : str
    motor_hip : float = 0.0
    motor_knee : float = 0.0
    motor_abduction : float = 0.0
    x : float = 0.0
    y : float = 0.0
    radius : float = 0.0
    theta : float = 0.0
    phi : float = 0.0
    gamma : float = 0.0
    b: float = 1.0
    step_length: float = 0.0
@dataclass
class robot_data:
    front_right : leg_data = leg_data('fr')
    front_left : leg_data = leg_data('fl')
    back_right : leg_data = leg_data('br')
    back_left : leg_data = leg_data('bl')

class WalkingController():
    def __init__(self,
                 gait_type='trot',
                 leg = [0.12,0.15015,0.04,0.15501,0.11187,0.04,0.2532,2.803],
                 phase = [0,0,0,0],
                 scale = 1.0,
                 comy = 0.0
                 ):     
        ## These are empirical parameters configured to get the right controller, these were obtained from previous training iterations  
        self._phase = robot_data(front_right = phase[0], front_left = phase[1], back_right = phase[2], back_left = phase[3])
        self.front_left = leg_data('fl')
        self.front_right = leg_data('fr')
        self.back_left = leg_data('bl')
        self.back_right = leg_data('br')
        print('#########################################################')
        print('This training is for', gait_type)
        print('#########################################################')
        self._leg = leg
        self.gamma = 0
        self.MOTOROFFSETS = [2.3562,1.2217]
        self.body_width = 0.24
        self.body_length = 0.37
        #New calculation
        #self._pts = np.array([[-0.068,-0.24],[-0.115,-0.24],[-0.065,-0.145],[0.065,-0.145],[0.115,-0.24],[0.068,-0.24]])
        self.swing_points = np.array([[-0.068,-0.22],[-0.066,-0.145],[0.066,-0.145],[0.068,-0.22]])
        self.stance_points = np.array([[0.068,-0.22],[0, -0.243],[-0.068,-0.22]])
        # self.new_pts = np.array([[-0.058  #policy = np.load("/home/abhik/ID/ICRA2020_ARS/pybRL/ARS/21Feb8/iterations/best_policy.npy")
  #policy = np.load("0.5_radius_policy.npy"),-0.24],[-0.105,-0.24],[-0.055,-0.145],[0.075,-0.145],[0.125,-0.24],[0.078,-0.24]])
        self.scale = scale
        self.comy = comy

    def update_leg_theta(self,theta):
        """ Depending on the gait, the theta for every leg is calculated"""
        def constrain_theta(theta):
            theta = np.fmod(theta, 2*no_of_points)
            if(theta < 0):
                theta = theta + 2*no_of_points
            return theta
        self.front_right.theta = constrain_theta(theta+self._phase.front_right)
        self.front_left.theta = constrain_theta(theta+self._phase.front_left)
        self.back_right.theta = constrain_theta(theta+self._phase.back_right)
        self.back_left.theta = constrain_theta(theta+self._phase.back_left)

    def get_swing_stance_weights(self, action):
        # if action[1] < 0.1:
        #     action[1] = 0.1
        #print(action.tolist())
        swing_weights = np.array([action[0], action[1], action[2], action[3]])
        stance_weights = np.array([action[3],action[4],action[0]])
        return  swing_weights, stance_weights

    def transform_action_to_motor_joint_command_bezier(self, theta, action, radius):

        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        legs = Legs(front_right = self.front_right, front_left = self.front_left, back_right = self.back_right, back_left = self.back_left)
        step_length = 0.068*2

        if radius == 1.0:
            radius = 10000

        self._update_leg_phi(radius)
        self._update_leg_step_length(step_length, radius)
        self.update_leg_theta(theta)

        actionf = action[:5]
        actionb = action[5:]

        swing_weightsf, stance_weightsf = self.get_swing_stance_weights(actionf)
        swing_weightsb, stance_weightsb = self.get_swing_stance_weights(actionb)

        for leg in legs:
            
            tau = leg.theta/no_of_points

            if leg.name =="fr" or "fl":
                x,y = self.drawfullBezier(self.swing_points,swing_weightsf, self.stance_points, stance_weightsf, tau)
            else:
                x,y = self.drawfullBezier(self.swing_points,swing_weightsb, self.stance_points, stance_weightsb, tau)

            leg.x, leg.y, leg.z = np.array([[np.cos(leg.phi),0,np.sin(leg.phi)],[0,1,0],[-np.sin(leg.phi),0, np.cos(leg.phi)]])@np.array([x,y,0])

            leg.x = leg.x * self.scale 
            leg.z = leg.z * self.scale 
            
            leg.motor_knee, leg.motor_hip, leg.motor_abduction = self._inverse_3D(leg.x, leg.y, leg.z, self._leg)
            leg.motor_abduction = constrain_abduction(leg.motor_abduction)
            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS[1]
        
        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip, legs.front_right.motor_knee,
        legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip, legs.back_right.motor_knee]
        leg_abduction_angles = [legs.front_left.motor_abduction, legs.front_right.motor_abduction, legs.back_left.motor_abduction,
        legs.back_right.motor_abduction]
        return leg_abduction_angles,leg_motor_angles, np.zeros(2), np.zeros(8)
    
    def footstep_to_motor_joint_command(self, theta, footstep, footstep_last):        
        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        legs = Legs(front_right = self.front_right, front_left = self.front_left, back_right = self.back_right, back_left = self.back_left)
        step_length = 0.068*2
        self._update_leg_transformation_matrix(legs, footstep)
        self._update_leg_step_length_footstep(legs, footstep)
        self.update_leg_theta(theta)
        for leg in legs:
            tau = leg.theta/PI
            weights = np.ones(6)
            x,y = self.drawBezier(self._pts, weights, tau)
            #We need to move it also, so that when tau = 0, it reaches footstep_last, tau = 1 it reaches footstep_current
            leg.x, leg.y, leg.z = np.array([[np.cos(leg.phi),0,np.sin(leg.phi)],[0,1,0],[-np.sin(leg.phi),0, np.cos(leg.phi)]])@np.array([x,y,0])
            leg.motor_knee, leg.motor_hip,leg.motor_abduction = self._inverse_3D(leg.x, leg.y, leg.z, self._leg)
            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS[1]
        
        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip, legs.front_right.motor_knee,
        legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip, legs.back_right.motor_knee]
        leg_abduction_angles = [legs.front_left.motor_abduction, legs.front_right.motor_abduction, legs.back_left.motor_abduction,
        legs.back_right.motor_abduction]
        return leg_abduction_angles,leg_motor_angles, np.zeros(2), np.zeros(8)
    
    def run_traj2d(self, theta, fl_rfunc, fr_rfunc, bl_rfunc, br_rfunc):
        """
        Provides an interface to run trajectories given r as a function of theta and abduction angles
        """
        Legs = namedtuple('legs', 'front_right front_left back_right back_left')
        legs = Legs(front_right = self.front_right, front_left = self.front_left, back_right = self.back_right, back_left = self.back_left)
        legs.front_left.phi = fl_rfunc[1]
        legs.front_right.phi = fr_rfunc[1]
        legs.back_left.phi = bl_rfunc[1]
        legs.back_right.phi = br_rfunc[1]
        self.update_leg_theta(theta)
        for leg in legs:
            y_center = -0.195
            leg.r = leg.rfunc(theta)
            # print(leg.theta)
            x = leg.r * np.cos(leg.theta)
            y = leg.r * np.sin(leg.theta) + y_center
            leg.x, leg.y, leg.z = np.array([[np.cos(leg.phi),0,np.sin(leg.phi)],[0,1,0],[-np.sin(leg.phi),0, np.cos(leg.phi)]])@np.array([x,y,0])
            # leg.z = leg.r * np.cos(leg.gamma)*(1 - leg.b * np.cos(leg.theta))
            # leg.z = np.abs(leg.z)
            leg.motor_knee, leg.motor_hip,leg.motor_abduction = self._inverse_3D(leg.x, leg.y, leg.z, self._leg)
            leg.motor_hip = leg.motor_hip + self.MOTOROFFSETS[0]
            leg.motor_knee = leg.motor_knee + self.MOTOROFFSETS[1]
            #-1 is also due to a coordinate difference
            # leg.motor_abduction = constrain_abduction(np.arctan2(leg.z, -leg.y))
        leg_motor_angles = [legs.front_left.motor_hip, legs.front_left.motor_knee, legs.front_right.motor_hip, legs.front_right.motor_knee,
        legs.back_left.motor_hip, legs.back_left.motor_knee, legs.back_right.motor_hip, legs.back_right.motor_knee]
        leg_abduction_angles = [legs.front_left.motor_abduction, legs.front_right.motor_abduction, legs.back_left.motor_abduction,
        legs.back_right.motor_abduction]
        return leg_abduction_angles,leg_motor_angles, np.zeros(2), np.zeros(8) 
    

    def _inverse_stoch2(self, x,y,Leg):

        l1 =    Leg[0]
        l2 =    Leg[1]
        l4 =    Leg[2]
        l5 =    Leg[3]
        le =    Leg[5]
        tq1 =   Leg[6]
        tq2 =   Leg[7]
        delta = Leg[4]
        xb = [[0,0],[0,0]]
        yb = [[0,0],[0,0]]
        phid = [0,0];psi = [0,0]; theta = [0,0]
        R_base = [[0,0],[0.035,0]]
        xb[0] = R_base[0][0];xb[1] = R_base[1][0]
        yb[0] = R_base[0][1];yb[1] = R_base[1][1]
        l3 = np.sqrt((x-xb[0])**2+(y-yb[0])**2)
        theta[0] = np.arctan2((y-yb[0]),(x-xb[0]))
        zeta = (l3**2 - l1**2 -l2**2)/(2*l1*l2)
        zeta = np.sign(zeta) if abs(zeta) > 1 else zeta
        phid[0] = np.arccos(zeta)
        psi[0] = np.arctan2(l2*np.sin(phid[0]),(l1+l2*np.cos(phid[0])))
        q1 = theta[0] - psi[0]
        q2 = q1 + phid[0]
        xm = l1*np.cos(q1)+l2*np.cos(q2)
        ym = l1*np.sin(q1)+l2*np.sin(q2)
        xi = (xm+xb[0])
        yi = (ym+yb[0])

        xi = xb[0] + l1*np.cos(q1) + 0.04*np.cos(q2-tq1)
        yi = yb[0] + l1*np.sin(q1) + 0.04*np.sin(q2-tq1)
        R = [xi,yi]
        l6 = np.sqrt(((xi-xb[1])**2+(yi-yb[1])**2))
        theta[1] = np.arctan2((yi-yb[1]),(xi-xb[1]))
        Zeta = (l6**2 - l4**2 - l5**2)/(2*l5*l4)
        leg = 'left'
        Zeta = np.sign(Zeta) if abs(Zeta) > 1 else Zeta
        phid[1] = np.arccos(Zeta)
        psi[1] = np.arctan2(l5*np.sin(phid[1]),(l4+l5*np.cos(phid[1])))
        q3 = theta[1]+psi[1]
        q4 = q3-phid[1]
        xm = l4*np.cos(q3)+l5*np.cos(q4)+xb[1]
        ym = l4*np.sin(q3)+l5*np.sin(q4)+yb[1]

        if Zeta == 1:
            [q1, q2] = self._inverse_new(xm,ym,delta,Leg)

        return [q3, q1, q4, q2]

    def _inverse_new(self, xm,ym,delta,Leg):

        l1 = Leg[0]
        l2 = Leg[1]-Leg[4]
        l4 = Leg[2]
        l5 = Leg[3]
        delta = Leg[4]
        xb = [[0,0],[0,0]]
        yb = [[0,0],[0,0]]
        phid = [0,0];psi = [0,0]; theta = [0,0]
        R_base = [[1,0],[-1,0]]
        xb[0] = R_base[0][0];xb[1] = R_base[1][0]
        yb[0] = R_base[0][1];yb[1] = R_base[1][1]
        l3 = np.sqrt((xm-xb[0])**2+(ym-yb[0])**2)
        theta[0] = np.arctan2((ym-yb[0]),(xm-xb[0]))
        zeta = (l3**2 - l1**2 -l2**2)/(2*l1*l2)
        zeta = np.sign(zeta) if abs(zeta) > 1 else zeta
        phid[0] = np.arccos(zeta)
        psi[0] = np.arctan2(l2*np.sin(phid[0]),(l1+l2*np.cos(phid[0])))
        q1 = theta[0] + psi[0]
        q2 = q1 - phid[0]
        xm = l1*np.cos(q1)+l2*np.cos(q2)
        ym = l1*np.sin(q1)+l2*np.sin(q2)

        return [q1,q2]

    def _inverse_3D(self, x, y, z, Leg):
        theta = np.arctan2(z,-y)

        new_coords = np.array([x,-y/np.cos(theta) - 0.035, z])

        motor_knee, motor_hip, _, _ = self._inverse_stoch2(new_coords[0], -new_coords[1], Leg)
        return [motor_knee, motor_hip, theta]

    def _update_leg_phi_val(self, leg_phi):
        self.front_right.phi =  leg_phi[0]
        self.front_left.phi = leg_phi[1]
        self.back_right.phi =    leg_phi[2]
        self.back_left.phi =  leg_phi[3]
    
    def _update_leg_phi(self, radius):
        if(radius >= 0):
            self.front_left.phi =  np.arctan2(self.body_length/2, radius + self.body_width/2)
            self.front_right.phi = -np.arctan2(self.body_length/2, radius - self.body_width/2)
            self.back_left.phi = -np.arctan2(self.body_length/2, radius + self.body_width/2)
            self.back_right.phi =  np.arctan2(self.body_length/2, radius - self.body_width/2)
          
        if(radius<0):
            newr = -1*radius
            self.front_right.phi =  np.arctan2(self.body_length/2, newr + self.body_width/2)
            self.front_left.phi = -np.arctan2(self.body_length/2, newr - self.body_width/2)
            self.back_right.phi = -np.arctan2(self.body_length/2, newr + self.body_width/2)
            self.back_left.phi =  np.arctan2(self.body_length/2, newr - self.body_width/2)

    def _update_leg_step_length_val(self, step_length):
        self.front_right.step_length = step_length[0]
        self.front_left.step_length = step_length[1]
        self.back_right.step_length = step_length[2]
        self.back_left.step_length = step_length[3]


    def _update_leg_step_length(self, step_length, radius):
        if(abs(radius) <= 0.12):
            self.front_right.step_length = step_length
            self.front_left.step_length = step_length 
            self.back_right.step_length = step_length 
            self.back_left.step_length = step_length 
            return

        if(radius >= 0):
            self.front_right.step_length = step_length * (radius - self.body_width/2)/radius
            self.front_left.step_length = step_length * (radius + self.body_width/2)/radius
            self.back_right.step_length = step_length * (radius - self.body_width/2)/radius
            self.back_left.step_length = step_length * (radius + self.body_width/2)/radius
            return

        if(radius < 0):
            newr = radius*-1
            self.front_left.step_length = step_length * (newr- self.body_width/2)/newr
            self.front_right.step_length = step_length * (newr + self.body_width/2)/newr
            self.back_left.step_length = step_length * (newr - self.body_width/2)/newr
            self.back_right.step_length = step_length *(newr + self.body_width/2)/newr
            return 

    def drawSwingBezier(self, points, weights, t):
        newpoints = np.zeros(points.shape)
        def drawCurve(points, weights, t):
            # print("ent1")
            if(points.shape[0]==1):
                return [points[0,0]/weights[0], points[0,1]/weights[0]]
            else:
                newpoints=np.zeros([points.shape[0]-1, points.shape[1]])
                newweights=np.zeros(weights.size)
                for i in np.arange(newpoints.shape[0]):
                    x = (1-t) * points[i,0] + t * points[i+1,0]
                    y = (1-t) * points[i,1] + t * points[i+1,1]
                    w = (1-t) * weights[i] + t*weights[i+1]
                    newpoints[i,0] = x
                    newpoints[i,1] = y
                    newweights[i] = w
                return drawCurve(newpoints, newweights, t)
        for i in np.arange(points.shape[0]):
            newpoints[i]=points[i]*weights[i]

        if(t<1):
            return drawCurve(newpoints, weights, t)
        if(t>=1):
            return [points[-1,0]+ (t-1)*(points[0,0] - points[-1,0]), -0.243]


    def drawfullBezier(self, swing_points, swing_weights, stance_points, stance_weights,  t):

        def drawCurve(points, weights, t):
            if(points.shape[0]==1):
                return [points[0,0]/weights[0], points[0,1]/weights[0]]
            else:
                newpoints=np.zeros([points.shape[0]-1, points.shape[1]])
                newweights=np.zeros(weights.size)
                for i in np.arange(newpoints.shape[0]):
                    x = (1-t) * points[i,0] + t * points[i+1,0]
                    y = (1-t) * points[i,1] + t * points[i+1,1]
                    w = (1-t) * weights[i] + t*weights[i+1]
                    newpoints[i,0] = x
                    newpoints[i,1] = y
                    newweights[i] = w

                return drawCurve(newpoints, newweights, t)

        swing_newpoints = np.zeros(swing_points.shape)
        stance_newpoints = np.zeros(stance_points.shape)

        for i in np.arange(swing_points.shape[0]):
            swing_newpoints[i]=swing_points[i]*swing_weights[i]

        for i in np.arange(stance_points.shape[0]):
            stance_newpoints[i]=stance_points[i]*stance_weights[i]

        if(t<1):
            return drawCurve(swing_newpoints, swing_weights, t)
        if(t>=1):
            #return [stance_points[0,0]+ (t-1)*(stance_points[-1,0] - stance_points[0,0]), -0.21]
            return drawCurve(stance_newpoints, stance_weights, t-1)


    def _update_leg_step_length_footstep(legs, footstep, last_footstep):
        legs.front_left.step_length = ((footstep.front_left.x - last_footstep.front_left.x)**2 + (footstep.front_left.z - last_footstep.front_left.z)**2)**0.5
        legs.front_right.step_length = ((footstep.front_right.x - last_footstep.front_right.x)**2 + (footstep.front_right.z - last_footstep.front_right.z)**2)**0.5
        legs.back_left.step_length = ((footstep.back_left.x - last_footstep.back_left.x)**2 + (footstep.back_left.z - last_footstep.back_left.z)**2)**0.5
        legs.back_right.step_length = ((footstep.back_right.x - last_footstep.back_right.x)**2 + (footstep.back_right.z - last_footstep.back_right.z)**2)**0.5

    def _update_leg_transformation_matrix(legs, footstep, last_footstep):
        
        pass
def constrain_abduction(angle):
    if(angle < -0.1727):
        angle = -0.1727
    elif(angle > 0.35):
        angle = 0.35
    return angle

if(__name__ == "__main__"):
    # action = np.array([ 0.24504616, -0.11582746,  0.71558934, -0.46091432, -0.36284493,  0.00495828, -0.06466855, -0.45247894,  0.72117291, -0.11068088])

    walkcon = WalkingController(phase=[PI,0,0,PI])
    walkcon._update_leg_step_length(0.068*2, 0.4)
    walkcon._update_leg_phi( 0.4)

