import numpy as np
#import matplotlib 
#matplotlib.use('agg')
import scipy.linalg as LA
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import glob
from tqdm import tqdm
import sys
args = sys.argv
import numba


# kernel calc mode 
mode = args[1] # 2D, list, for


## define kernel function 
@numba.jit
def rbf(x, x_dash,  theta1, theta2):
    return theta1 * np.exp(-1* ((x-x_dash)**2)/theta2) 

############ return kernel output as 2D matrix (3 ways)#########
def rbf_for(x, x_dash, theta1, theta2, theta3, add_theta3):
    K = np.zeros((len(x),len(x_dash)))
    if add_theta3:
        for i,ix in enumerate(x):
            for j,jx in enumerate(x_dash):
                if i==j:
                    K[i,j] =rbf(ix,jx,theta1,theta2) + theta3
                else:
                    K[i,j] =rbf(ix,jx,theta1,theta2)
    else:
        for i,ix in enumerate(x):
            for j,jx in enumerate(x_dash):
                if i==j:
                    K[i,j] =rbf(ix,jx,theta1,theta2) + theta3
    return K

def rbf_list(x, x_dash, theta1, theta2, theta3, add_theta3):
    if add_theta3:
        K = [[rbf(ix,jx,theta1,theta2) + theta3 if i==j else rbf(ix,jx, theta1,theta2) \
           for i,ix in enumerate(x_dash)] for j,jx in enumerate(x)]
        K = np.array(K).reshape((len(x), len(x_dash)))
    else:
        K = [[rbf(ix,jx,theta1,theta2) \
           for i,ix in enumerate(x_dash)] for j,jx in enumerate(x)]
        K = np.array(K).reshape((len(x), len(x_dash)))
    return K

def rbf_2Darray(x_2Darray, x_dash_2Darray, theta1, theta2, theta3, add_theta3):
    if add_theta3:
        assert len(x_2Darray) == len(x_dash_2Darray)
        return theta1 * np.exp(-1 * np.square((x_2Darray - x_dash_2Darray.T))/theta2) + np.identity(len(x_2Darray)) + theta3
    else:
        return theta1 * np.exp(-1 * np.square((x_2Darray - x_dash_2Darray.T))/theta2)

#########################################################################


# return 2D kernel matrix according to mode ("for" or "list" or "2D") 
@numba.jit
def kernel_matrix(x, x_dash, theta1, theta2, theta3, mode, add_theta3):
    if mode == '2D':
        return rbf_2Darray(x, x_dash, theta1, theta2, theta3, add_theta3)
    elif mode == 'list':
        return rbf_list(x, x_dash, theta1, theta2, theta3, add_theta3)
    elif mode == 'for':
        return rbf_for(x, x_dash, theta1, theta2, theta3, add_theta3)

# whole Gaussian processs
@numba.jit
def gp_reg(x_train, y_train, x_test, tau, sigma, eta):
    theta1 = np.exp(tau)
    theta2 = np.exp(sigma)
    theta3 = np.exp(eta)

    N = len(x_train)
    M = len(x_test)

    K = kernel_matrix(x_train, x_train, theta1, theta2, theta3, mode, True)

    K_inv = LA.inv(K)
    yy = K_inv.dot(y_train)

    detK = LA.det(K)

    likelifood = -1. * y_train.T.dot(yy) - np.log(detK)

    k = kernel_matrix(x_train, x_test, theta1, theta2, theta3, mode, False)
    s = kernel_matrix(x_test, x_test, theta1, theta2, theta3, mode, False)

    # results
    if mode == '2D':
        mu = (k.T).dot(yy)[:,0] # mean value, converted to 1Darray
    else:
        mu = (k.T).dot(yy) # mean value
    var = s - (k.T).dot(K_inv).dot(k)
    var_diag = np.diag(var) # 

    return likelifood, mu, var, var_diag, K

# core logic of single mcmc process
@numba.jit
def next(theta2, theta3, x_train, y_train, x_test, mcmc_s1, mcmc_s2,mode):
    while True:
        params_new = [0,0]
        theta2_new = theta2 + np.random.normal(0, mcmc_s1)
        theta3_new = theta3 + np.random.normal(0, mcmc_s2)
        lf_log,mu,var,var_diag,K = gp_reg(x_train, y_train, x_test, 0, theta2, theta3)
        lf_new_log,mu_new,var_new,var_diag_new,K_new = gp_reg(x_train, y_train, x_test, 0, theta2_new, theta3_new)
        lf =np.exp(lf_log)
        lf_new =np.exp(lf_new_log)
        #print("  lf: %.3e, lf_new: %.3e, p: %.2f" % (lf, lf_new, lf_new/lf))
        
        if np.random.uniform() <= min(lf_new/lf, 1):
        #if (lf_new > lf ):
            #print('    mcmc setp proceed') 
            return lf_new, theta2_new, theta3_new,  mu_new, var_diag_new
        #else:
            #print('    discarded')


# iterative mcmc process
@numba.jit
def mcmc(iter_num, theta2, theta3, x_sample, y_sample, x_test, mcmc_s1, mcmc_s2, mode, verbose):
    # initialize
    lf = 0
    mu = np.zeros(len(x_test))
    var_diag = np.zeros(len(x_test))
    theta2_array = np.zeros(iter_num)
    theta3_array = np.zeros(iter_num)
    lf_array = np.zeros(iter_num)

    if verbose:
        iter = tqdm(range(0,iter_num), ncols=70)
    else:
        iter = range(0,iter_num)

    for i in iter:
        lf, theta2, theta3, mu, var_diag = next(theta2, theta3, x_sample, y_sample, x_test, mcmc_s1, mcmc_s2, mode)
        theta2_array[i] = theta2
        theta3_array[i] = theta3
        likelifood_array[i] = lf
    return theta2_array, theta3_array, likelifood_array, lf, mu, var_diag


## make ground truth data
N = 20
high_end = 5
eps = 0.3
coef = 0.5
line_space = 100
x_test = np.linspace(0,high_end,line_space)
x_sample = np.random.rand(N) * high_end # [0,30]
y_sample = coef * x_sample + np.sin(x_sample) + np.random.normal(size=N) * eps

x_test_1D = x_test
x_sample_1D = x_sample
y_sample_1D = y_sample

# reshape as 2D matrix
if mode =='2D':
    x_test = x_test.reshape(-1,1)
    x_sample = x_sample.reshape(-1,1)
    y_sample = y_sample.reshape(-1,1)


## initial parameter value
# parameter of kernel function
theta1 = 1, #0.5
theta2 = np.random.uniform(-2,4)
theta3 = np.random.uniform(-2,4)
mcmc_s1 = 0.05
mcmc_s2 = 0.10


BURNIN = 0
SAMPLE = 25000

save_interval =SAMPLE-1

# burn in
burn_x = np.zeros(BURNIN)
burn_y = np.zeros(BURNIN)
likelifood_array = np.zeros(BURNIN+SAMPLE)


# exec MCMC
theta2_array, theta3_array, lf_array, lf, mu, var_diag = mcmc(SAMPLE, theta2, theta3, x_sample, y_sample, x_test, mcmc_s1, mcmc_s2, mode, True)




def result_plot(x_s,y_s,x_t,t2,t3,mu,coef,var,mode):
    fig = plt.figure(figsize=(6,9))

    plt.subplot(411)

    plt.plot(t2, t3, marker = 'o', markersize=1)
    plt.xlabel('sigma')
    plt.ylabel('eta')
    
    plt.subplot(412)
    plt.scatter(x_s, y_s, marker='x') # plot of train data
    plt.plot(x_t, mu, c='Orange') # predictive line using GP regression
    plt.plot(x_t, coef * x_t + np.sin(x_t),'--', c='k') # ground truth
    plt.fill_between(x_t, mu-2 *var, mu+2*var, color='grey', alpha=.4)
    #plt.title("N: {}".format(e))
    plt.xlim(0,high_end)
    plt.xlim(0,high_end)
    #
    plt.ylim(-1.5,4)
    
    plt.subplot(413)
    plt.plot(lf_array)
    plt.xlabel("step")
    plt.xlabel("likelifood")
    
    plt.subplot(414)
    plt.plot(t2, c='Orange', label='theta2') # predictive line using GP regression
    plt.plot(t3, c='b', label='theta3') # predictive line using GP regression
    plt.xlabel("step")
    plt.ylabel("param")
    plt.savefig("images/gp_mcmc_%s.png" % mode)
    plt.close()
 
result_plot(x_sample_1D,y_sample_1D,x_test_1D,theta2,theta3,mu,coef,var_diag,mode)

#
##
##
##plt.savefig("images/gp_regression_grad_%04d.png" % i)
##plt.close()
#
#
#images = []
#files = glob.glob("images/gp_mcmc*.png")
#files.sort()
##
#for f in files:
#    im = Image.open(f)
#    images.append(im)
#
#images[0].save('gp_mcmc.gif',\
#               save_all=True, append_images=images[1:],\
#			   optimize=False, duration=500, loop=0)
#



