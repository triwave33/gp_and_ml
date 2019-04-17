import numpy as np
#import matplotlib 
#matplotlib.use('agg')
import scipy.linalg as LA
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import glob


# parameter of kernel function
theta1 = 10.0
theta2 = 1.4
theta3 = 0.1

## define kernel function 
def rbf(x, x_dash,  theta1=theta1, theta2=theta2):
    return theta1 * np.exp(-1* ((x-x_dash)**2)/theta2) 

# entire Gaussian processs
def gp_reg(x_train, y_train, x_test, t1, t2, t3):
    N = len(x_train)
    M = len(x_test)

    y_mean = np.mean(y_train)
    y_train_minus_mean = y_train - y_mean
    # add Gaussian noise
    K = [[rbf(ix,jx,t1,t2) + t3 if i==j else rbf(ix,jx, t1,t2) \
            for i,ix in enumerate(x_train)] for j,jx in enumerate(x_train)]
    K = np.array(K).reshape((N,N))
    print("K.shape {}".format(K.shape))

    K_inv = LA.inv(K)
    yy = K_inv.dot(y_train_minus_mean)


    k = [[rbf(ix,jx,t1,t2)  \
            for i,ix in enumerate(x_test)] for j,jx in enumerate(x_train)]
    k = np.array(k).reshape((N,M))

    s = [[rbf(ix,jx,t1,t2) + t3 if i==j else rbf(ix,jx,t1,t2) \
            for i,ix in enumerate(x_test)] for j,jx in enumerate(x_test)]
    s = np.array(s).reshape((M,M))

    # results
    mu = (k.T).dot(yy) 
    var = s - (k.T).dot(K_inv).dot(k)
    var_diag = np.diag(var) # 
    return mu, var, var_diag, y_mean

#############################
# Sample generation
'''
x: generated from uniform distribution
y = coef * x + intercept + noise
noise: generated from normal distribution 
'''

N = 10
sigma = 0.3
coef = 0.5
intercept= 5
x_sampler = np.random.uniform(-4*np.pi,0,size=N) # [-4pi, 0]
x_sampler = np.sort(x_sampler)
y_sampler = coef * x_sampler + np.sin(x_sampler) + intercept + np.random.normal(size=N) * sigma

# grid purpose
x_test = np.linspace(-4*np.pi,np.pi,200)

#########################################
# run GP
'''
Gaussian process is done from time to time
{t_0 , ... , t_{i}}
latest sample point is plotted as red marker
'''
for i in range(N):
    fig = plt.figure(figsize=(4,4))
    print("i= {}, length= {}".format(i, len(x_sampler[:i+1])))
    # gp NOTE: regression value is calculated AFTER mean subtraction
    mu, var, var_diag, y_mean = gp_reg(x_sampler[:i+1], y_sampler[:i+1], x_test, theta1, theta2, theta3)

    # plot 
    plt.scatter(x_sampler[:i], y_sampler[:i], marker='x') # train data
    plt.scatter(x_sampler[i], y_sampler[i], marker='x', c='red') # latest train data
    plt.plot(x_test, mu+y_mean, c='Orange', label='gp_reg') # predictive line using GP regression
    plt.plot(x_test, np.ones(200) * y_mean, '--',  c='red', linewidth=0.5,\
            label='mean') # mean val
    plt.plot(x_test, coef * x_test + intercept + np.sin(x_test),'--', c='k', label='answer') # ground truth
    plt.fill_between(x_test, mu+y_mean -2 *var_diag, mu+y_mean + 2*var_diag, color='grey', alpha=.4) # error region
    plt.title("N: {}".format(i))
    plt.xlim(-4*np.pi,np.pi)
    plt.ylim(-2,8)
    plt.legend(loc=2)

    plt.savefig("images/gp_regression_minus_mean_%03d.png" % i)
    plt.close()

images = []
files = glob.glob("images/gp_regression_minus_mean*.png")
files.sort()
#
for f in files:
    im = Image.open(f)
    images.append(im)

images[0].save('gp_process_minus_mean.gif',\
               save_all=True, append_images=images[1:],\
			   optimize=False, duration=1000, loop=0)




