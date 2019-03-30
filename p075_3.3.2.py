### -*-coding:utf-8-*-
## 3.3.2 Various kernels
import matplotlib.backends
import matplotlib 
matplotlib.use('Agg')

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

# NOTE: For 1D input 

## define kernel function 
# linear
def linear(x, x_dash):
    return np.array([1,x]).dot(np.array([1,x_dash]))
# gaussian
def rbf(x, x_dash, theta=1.0):
    return np.exp(-1* ((x-x_dash)**2)/theta)
# exponential kernel
def exponential(x, x_dash, theta=1.0):
    return np.exp(-1* np.abs(x-x_dash)/theta)
# periodic kernel
def periodic(x, x_dash, theta1=1.0, theta2=np.pi/6.):
    return np.exp(theta1 * np.cos(np.abs(x - x_dash)/theta2))

## define callback function
def handler(func, *args):
    return func(*args)

# set sampling point
N = 100 # for obtaining smooth line
callbacks = {'linear':linear, 'rbf':rbf, 'exponential':exponential,'periodic':periodic}

# set drawing cambus
fig, ax = plt.subplots(2, 2, figsize=((8,6)))

x = np.linspace(-4,4,N)

for ind,key in enumerate(callbacks):

    # covariance matrix K
    K = [[handler(callbacks[key],i,j) for i in x for j in x]]
    K = np.array(K).reshape(N,N)

    # sampling from n-D Gaussian distribution
    y = np.random.multivariate_normal(mean=np.zeros(len(x)), cov=K, size=4) # 4 samples

    ## draw
    for yy in y: # 4 iterations
        ax[ind//2][ind%2].plot(x,yy)
    ax[ind//2][ind%2].set_title(key)
    ax[ind//2][ind%2].set_xlim(-4,4)
    ax[ind//2][ind%2].set_ylim(-4,4)


    
fig.savefig("p075_3.3.2_sample.png")
plt.close()

fig, ax = plt.subplots(2,2, figsize=((10,8)))
for ind,key in enumerate(callbacks):
    # covariance matrix K
    K = [[handler(callbacks[key],i,j) for i in x for j in x]]
    K = np.array(K).reshape(N,N)
    ax[ind//2][ind%2] = sns.heatmap(K, ax=ax[ind//2][ind%2])
    ax[ind//2][ind%2].set_title(key)
fig.savefig("p075_3.3.2_kernel.png")
plt.close()

### supplimentsl
# create mixed kernels
## rbf and periodic kernel
c1= 'rbf'
c2 = 'linear'
fig, ax = plt.subplots(1,1, figsize=((8,8)))
K_c1 = [[handler(callbacks[key],i,j) for i in x for j in x]]
K_c1 = np.array(K)
K_c2 = [[handler(callbacks[key],i,j) for i in x for j in x]]
K_c2 = np.array(K)
K = K_c1 + K_c2
ax = sns.heatmap(K, ax=ax)
ax.set_title("%s and %s kernels" % (str(c1), str(c2)))


fig.savefig("p075_3.3.2_%s_%s_kernel.png" % (str(c1), str(c2)))
plt.close()

 
