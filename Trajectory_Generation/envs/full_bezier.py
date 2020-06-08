import matplotlib.pyplot as plt
import numpy as np
y = np.arange(-0.145, -0.245, -0.001)

x_max = np.zeros(y.size)
x_min = np.zeros(y.size)

trap_pts = []
count = 0

for pt in y:
    x_max[count] = (pt+0.01276)/1.9737
    x_min[count] = -1*(pt+0.01276)/1.9737
    count = count + 1
x_bottom = np.arange(x_min[-1], x_max[-1], -0.001)
x_top = np.arange(x_min[0], x_max[0], -0.001)
final_x = np.concatenate([x_max, np.flip(x_bottom,0), x_min, x_top])
final_y = np.concatenate([y, np.ones(x_bottom.size)*y[-1], y, np.ones(x_top.size)*y[0]])

thetas = np.arange(0, 2*np.pi, 0.001)
tau = thetas/(2*np.pi)
x = np.zeros(thetas.size)
y = np.zeros(thetas.size)
count = 0

x_w = [-0.03, -0.1, 0.1, 0.03]
y_w = [-0.243, -0.2, -0.2, -0.243]

r = [0.01,0.01,0.01,0.01]

for t in tau:
    f = [((1-t)**3)*r[0], 3*t*((1-t)**2)*r[1],3*(1-t)*(t**2)*r[2], (t**3)*r[3]]
    basis = f[0] + f[1] + f[2] + f[3]
    x[count] = (x_w[0]*f[0] + x_w[1]*f[1]+x_w[2]*f[2]+x_w[3]*f[3])/basis
    y[count] = (y_w[0]*f[0]+  y_w[1]*f[1]+y_w[2]*f[2]+y_w[3]*f[3])/basis
    count = count + 1

plt.figure(1)
plt.plot(final_x,final_y,'r', label = 'robot workspace')


def drawBezier(swing_points, swing_weights, stance_points, stance_weights,  t):

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
        return drawCurve(stance_newpoints, stance_weights, t-1)
        #return [stance_points[0,0]+ (t-1)*(stance_points[-1,0] - stance_points[0,0]), -0.21]




x= np.zeros(80)
y =np.zeros(80)


swing_points = np.array([[-0.068,-0.21],[-0.066,-0.145],[0.066,-0.145],[0.068,-0.21]])
stance_points = np.array([[0.068,-0.21],[0, -0.243],[-0.068,-0.21]])

#pts_r = points.copy()
#pts_l = points.copy()
action = [1.0, 1.0, 0.7444449361252171, 1.0, 0.1, 0.5390553688270329]
action = [1.0, 0.3882134224948346, 0.1, 0.1, 1.0, 0.6156509903652028]
action = [1, 0, 1, 0.1, 0.1, 0.1]

action = [0.9621096927675283, 0.1837329065837064, 0.01]
#action = [0.504613100663817, 0.6072727182360427, 0.5694740009141699]

#action = [1, 0, 1]

# def get_swing_stance_weights(action):
#     swing_weights = np.array([action[0], action[1], action[1], action[2]])
#     stance_weights = np.array([action[2],action[3], action[4], action[5],action[0]])
#     return swing_weights, stance_weights

def get_swing_stance_weights(action):
    swing_weights = np.array([action[0], 1, 1, action[2]])
    stance_weights = np.array([action[2],action[1],action[0]])
    return  swing_weights, stance_weights

# swing_weights = np.ones(4)
# stance_weights = np.ones(5)
swing_weights, stance_weights = get_swing_stance_weights(action)
print(swing_weights)
#weightsl = np.ones(4)

count = 0
for t in np.arange(0,2, 0.025):
    x[count], y[count] = drawBezier(swing_points, swing_weights, stance_points, stance_weights, t)
    count = count+1
# count =0
# for t in np.arange(0,2, 0.001):
#     x1[count], y1[count] = drawBezier(pts_l,weightsl, t)
#     count = count+1

plt.plot(x,y,".", label = 'robot trajectoryr')
#plt.plot(x1,y1,'r', label = 'robot trajectoryl')
plt.legend()
plt.show()
