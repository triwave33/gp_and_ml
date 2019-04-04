import numpy as np
#import matplotlib 
#matplotlib.use('tkagg')
import scipy.linalg as LA
import matplotlib.pyplot as plt

# parameter of kernel function
theta1 = 1.0
theta2 = 0.4
theta3 = 0.1

## define kernel function 
def rbf(x, x_dash,  theta1=theta1, theta2=theta2):
    return theta1 * np.exp(-1* ((x-x_dash)**2)/theta2) 

# train data 
x_train = np.array([-5, -3.,-1.8,0.8, 2.3])
y_train = np.array([0.8, 2.1, 1.78,2.4, 1.2])
assert len(x_train) == len(y_train)
N = len(y_train)

# add Gaussian noise
K = [[rbf(ix,jx) + theta3 if i==j else rbf(ix,jx) \
        for i,ix in enumerate(x_train)] for j,jx in enumerate(x_train)]
K = np.array(K).reshape((N,N))

K_inv = LA.inv(K)
yy = K_inv.dot(y_train)


# test data
M =100
x_test = np.linspace (-5,3,M)
k = [[rbf(ix,jx) + theta3 if \
        for i,ix in enumerate(x_test)] for j,jx in enumerate(x_train)]
k = np.array(k).reshape((N,M))

s = [[rbf(ix,jx) + theta3 if i==j else rbf(ix,jx) \
        for i,ix in enumerate(x_test)] for j,jx in enumerate(x_test)]
s = np.array(s).reshape((M,M))

# results
mu = (k.T).dot(yy) # mean value
var = s - (k.T).dot(K_inv).dot(k)
var_diag = np.diag(var) # 

# plot
plt.scatter(x_train, y_train, marker='x') # plot of train data
plt.plot(x_test, mu, c='Orange') # predictive line using GP regression
# error region (using 2 sigma)
plt.fill_between(x_test, mu-2 *var_diag, mu+2*var_diag, color='grey', alpha=.4)
plt.savefig("p086_3.4.2_GP_regression.png")

