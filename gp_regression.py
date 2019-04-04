import numpy as np
import matplotlib 
matplotlib.use('tkagg')
import scipy.linalg as LA
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import glob

### functionize gp regression
#input:
#    x_train: 
#    y_train:
#    x_test: usually as a grid sample 
##paramter:
#    theta1
#    theta2
#    theta3
#
#output
#    mu: predictive mean
#    var: uncertainly of prediction (sigma**2)
#   var_diag: diagnostic value of var matrix


# parameter of kernel function
theta1 = 1.0
theta2 = 0.4
theta3 = 0.2

## define kernel function 
def rbf(x, x_dash,  theta1=theta1, theta2=theta2):
    return theta1 * np.exp(-1* ((x-x_dash)**2)/theta2) 

# train data 
x_train = np.array([-5, -3.,-1.8,0.8, 2.3])
y_train = np.array([0.8, 2.1, 1.78,2.4, 1.2])
assert len(x_train) == len(y_train)
N = len(y_train)
# test data
M =100
x_test = np.linspace (-5,3,M)


def gp_reg(x_train, y_train, x_test, t1, t2, t3):
    N = len(x_train)
    M = len(x_test)
    # add Gaussian noise
    K = [[rbf(ix,jx,t1,t2) + t3 if i==j else rbf(ix,jx, t1,t2) \
            for i,ix in enumerate(x_train)] for j,jx in enumerate(x_train)]
    K = np.array(K).reshape((N,N))
    print("K.shape {}".format(K.shape))
    
    K_inv = LA.inv(K)
    yy = K_inv.dot(y_train)
    
    
    k = [[rbf(ix,jx,t1,t2)  \
            for i,ix in enumerate(x_test)] for j,jx in enumerate(x_train)]
    k = np.array(k).reshape((N,M))
    
    s = [[rbf(ix,jx,t1,t2) + t3 if i==j else rbf(ix,jx,t1,t2) \
            for i,ix in enumerate(x_test)] for j,jx in enumerate(x_test)]
    s = np.array(s).reshape((M,M))
    
    # results
    mu = (k.T).dot(yy) # mean value
    var = s - (k.T).dot(K_inv).dot(k)
    var_diag = np.diag(var) # 
    return mu, var, var_diag

## plot
#plt.scatter(x_train, y_train, marker='x') # plot of train data
#plt.plot(x_test, mu, c='Orange') # predictive line using GP regression
## error region (using 2 sigma)
#plt.fill_between(x_test, mu-2 *var_diag, mu+2*var_diag, color='grey', alpha=.4)
#plt.savefig("p086_3.4.2_GP_regression.png")
#

theta2_list = [0.5, 1., 3]
theta3_list = [0.0, 0.1, 0.5]

N = 20
sigma = 0.3
coef = 0.5
x_test = np.linspace(0,7,200)
x_sampler = np.random.rand(N) * 5 # [0,5]
#x_sampler = np.sort(x_sampler)
y_sampler = coef * x_sampler + np.sin(x_sampler) + np.random.normal(size=N) * sigma
#fig, ax = plt.subplots(3,3, figsize=(12,12))

#for i, t2 in enumerate(theta2_list):
#    for j, t3 in enumerate(theta3_list):
#        print("{}, {}".format(i,j))
#        mu, var, var_diag = gp_reg(x_train, y_train, x_test, theta1, t2, t3)
#
#        ax[i][j].scatter(x_train, y_train, marker='x') # plot of train data
#        ax[i][j].plot(x_test, mu, c='Orange') # predictive line using GP regression
#        ax[i][j].fill_between(x_test, mu-2 *var_diag, mu+2*var_diag, color='grey', alpha=.4)
#        ax[i][j].set_title("theta2: {}, theta3: {}".format(t2, t3))

for i in range(N):
    fig = plt.figure(figsize=(4,4))
    print("i= {}, length= {}".format(i, len(x_sampler[:i+1])))
    mu, var, var_diag = gp_reg(x_sampler[:i+1], y_sampler[:i+1], x_test, theta1, theta2, theta3)

    plt.scatter(x_sampler[:i], y_sampler[:i], marker='x') # plot of train data
    plt.scatter(x_sampler[i], y_sampler[i], marker='x', c='red') # plot of latest train data
    plt.plot(x_test, mu, c='Orange') # predictive line using GP regression
    plt.plot(x_test, coef * x_test + np.sin(x_test),'--', c='k') # ground truth
    plt.fill_between(x_test, mu-2 *var_diag, mu+2*var_diag, color='grey', alpha=.4)
    plt.title("N: {}".format(i))
    plt.xlim(0,7)
    plt.ylim(-1.5,3.5)

    plt.savefig("images/gp_process_%03d.png" % i)
    plt.close()

images = []
files = glob.glob("images/gp_process*.png")
files.sort()
#
for f in files:
    im = Image.open(f)
    images.append(im)

images[0].save('gp_process.gif',\
               save_all=True, append_images=images[1:],\
			   optimize=False, duration=1000, loop=0)




