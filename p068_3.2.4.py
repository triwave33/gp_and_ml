### -*-coding:utf-8-*-
## 3.2.4 Sampling from Gaussian process

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as st

# define kernel function (RBF)
def rbf(x, x_dash, theta1=1.0, theta2=1.0):
    return theta1 * np.exp(-1* np.sqrt((x-x_dash)**2)/theta2)

# set sampling point
N = [5,20,50] # as lists

# set drawing cambus
plt, ax = plt.subplots(len(N), 2, figsize=((8,6)))

# iteration for each sample number
for ind, n in enumerate(N):
    x = np.linspace(1,4,n)

    # covariance matrix K
    K = [[rbf(i,j) for i in x for j in x]]
    K = np.array(K).reshape(n,n)

    # sampling from n-D Gaussian distribution
    y = np.random.multivariate_normal(mean=np.zeros(len(x)), cov=K)

    ## draw
    ax[ind][0].scatter(x,y, marker='x')
    ax[ind][1] = sns.heatmap(K, ax=ax[ind][1])
    
    if ind==0:
        ax[ind][0].set_title("sampling, n={}".format(n))
        ax[ind][1].set_title("covariance matrix")
    else:
        ax[ind][0].set_title("n={}".format(n))
    if ind==len(N)-1:
        ax[ind][0].set_xlabel("x")
    ax[ind][0].set_ylabel("y")


plt.savefig("p068_3.2.4.png")

