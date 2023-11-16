import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize

cam_poses = np.load('cam_poses.npy')
cam_dirs = np.load('cam_dirs.npy')
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
#ax = Axes3D(fig)

rot_poses = []
rot = []
ups = []
for t,R in zip(cam_poses, cam_dirs):
    #print(R, t)
    #R = R.T
    t = -np.dot(R, t)
    rot_poses.append(t)
    rot.append(R[:, 2])
    ups.append(R[:, 1])
    #ax.quiver(t[0], t[1], t[2], R[0, 0], R[1, 0], R[2, 0], color='r', length=0.4, normalize=True)
    #ax.quiver(t[0], t[1], t[2], R[0, 1], R[1, 1], R[2, 1], color='g', length=0.4, normalize=True)
    #stack translation vectors into an N x 3 matrix
    ax.quiver(t[0], t[1], t[2], R[0, 2], R[1, 2], R[2, 2], color='b', length=0.7, normalize=True)

#take svd of cam_poses
rot_poses = np.array(rot_poses)
rot = np.array(rot)
up_mean = np.array(ups).mean(axis=0)

U, S, V = np.linalg.svd(rot_poses)

def objective_function(x):
    return np.linalg.norm(rot @ x, ord=1)

# Define the constraint (Ax = b)

def norm_constraint(x):
    return np.linalg.norm(x) - 1

# Combine the constraints
constraints = [
    {'type': 'eq', 'fun': norm_constraint}
]

# Initial guess for the solution
x0 = np.zeros(rot_poses.shape[1])

# Solve the problem using L1 minimization

result = minimize(objective_function, x0, constraints=constraints, method='SLSQP')
up_mean = result.x
# print(rot_poses.shape)
# print(U.shape, S.shape, V.shape)
# print(V)
# print(S)
print(up_mean)
#plt last right singular vector
cam_center = np.mean(rot_poses, axis=0)
ax.quiver(cam_center[0], cam_center[1], cam_center[2], up_mean[0], up_mean[1], up_mean[2], color='r', length=0.7, normalize=True)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Camera Poses')

plt.show()