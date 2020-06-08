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
# center = [0, -0.195]
# center = [0,0] # Need to change

thetas = np.arange(0, 2*np.pi, 0.001)
tau = thetas/(2*np.pi)
x = np.zeros(thetas.size)
y = np.zeros(thetas.size)
count = 0 
x_w = [-0.03, -0.06, 0.06, 0.03]
y_w = [-0.243, -0.15, -0.15, -0.243]
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
body_width = 0.24
r = 10000
def _update_leg_step_length(radius, step_length = 0.068*2):
        if(abs(radius) <= 0.12):
            rstep_length = step_length
            lstep_length = step_length 
        if(radius >= 0):
            rstep_length = step_length * (radius - body_width/2)/radius
            lstep_length = step_length * (radius + body_width/2)/radius
           
        if(radius < 0):
            newr = radius*-1
            lstep_length = step_length * (newr- body_width/2)/newr
            rstep_length = step_length * (newr +body_width/2)/newr
        return[rstep_length, lstep_length]

rstep_length, lstep_length = _update_leg_step_length(r)
print(rstep_length, lstep_length)

action = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
action = np.array([0.01, 0.01, 0.131, 1.0, 1.0, 0.952, 1.0, 1.0, 1.0, 0.01, 0.282, 1.0])
#action = np.array([1.0, 0.462, 1.0, 1.0, 1.0, 1.0, 0.754, 1.0, 1.0, 0.01, 0.326, 1.0])
actionr = action[:6]
actionl = action[6:]

weightsr = (actionr+1.0)/2
weightsl = (actionl +1.0)/2


for i in range (weightsr.shape[0]):
    if weightsr[i] == 0:
        weightsr[i] = 1e-1

for i in range (weightsl.shape[0]):
    if weightsl[i] == 0:
        weightsl[i] = 0.1
print(weightsl)

points = np.array([[-0.068,-0.243],[-0.115,-0.243],[-0.065,-0.145],[0.065,-0.145],[0.115,-0.243],[0.068,-0.243]])
pts_r = points.copy()
pts_l = points.copy()

pts_r[0,0] = -1*rstep_length/2
pts_r[-1,0] = rstep_length/2

pts_l[0,0] = -1*lstep_length/2
pts_l[-1,0] = lstep_length/2
print(rstep_length, lstep_length)
print(pts_l, pts_r)
# weights = np.array([0.01,0.01,1,0,1,1])
def drawBezier(points, weights, t):
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


x= np.zeros(2000)
y =np.zeros(2000)
x1= np.zeros(2000)
y1 =np.zeros(2000)
count = 0
for t in np.arange(0,2, 0.001):
    x[count], y[count] = drawBezier(pts_r,weightsr, t)
    count = count+1
count =0
for t in np.arange(0,2, 0.001):
    x1[count], y1[count] = drawBezier(pts_l,weightsl, t)
    count = count+1

plt.plot(x,y,'g', label = 'robot trajectoryr')
plt.plot(x1,y1,'r', label = 'robot trajectoryl')
plt.legend()
plt.show()

