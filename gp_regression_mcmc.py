import numpy as np
#import matplotlib 
#matplotlib.use('agg')
import scipy.linalg as LA
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import glob


## define kernel function 
def rbf(x, x_dash,  theta1, theta2):
    return theta1 * np.exp(-1* ((x-x_dash)**2)/theta2) 


# whole Gaussian processs
def gp_reg(x_train, y_train, x_test, tau, sigma, eta):
    theta1 = np.exp(tau)
    theta2 = np.exp(sigma)
    theta3 = np.exp(eta)

    N = len(x_train)
    M = len(x_test)
    # add Gaussian noise
    #K = [[rbf(ix,jx,tau,sigma) + eta if i==j else rbf(ix,jx, tau,sigma) \
    K = [[rbf(ix,jx,theta1,theta2) + theta3 if i==j else rbf(ix,jx, theta1,theta2) \
            for i,ix in enumerate(x_train)] for j,jx in enumerate(x_train)]
    K = np.array(K).reshape((N,N))
    #print("K.shape {}".format(K.shape))

    K_inv = LA.inv(K)
    yy = K_inv.dot(y_train)

    detK = LA.det(K)

    #print("detK: %.4e" % detK)

    likelifood = -1. * y_train.dot(yy) - np.log(detK)

    k = [[rbf(ix,jx,theta1,theta2)  \
            for i,ix in enumerate(x_test)] for j,jx in enumerate(x_train)]
    k = np.array(k).reshape((N,M))

    s = [[rbf(ix,jx,theta1,theta2) + theta3 if i==j else rbf(ix,jx,theta1,theta2) \
            for i,ix in enumerate(x_test)] for j,jx in enumerate(x_test)]
    s = np.array(s).reshape((M,M))

    # results
    mu = (k.T).dot(yy) # mean value
    var = s - (k.T).dot(K_inv).dot(k)
    var_diag = np.diag(var) # 

    return likelifood, mu, var, var_diag, K

def next(x, x_train, y_train, x_test,sigma0, sigma1):
    while True:
        x_new = [0,0]
        x_new[0] = x[0] + np.random.normal(0, sigma0)
        x_new[1] = x[1] + np.random.normal(0, sigma1)
        lf,mu,var,var_diag,K = gp_reg(x_train, y_train, x_test, 1, x[0], x[1])
        lf_new,mu_new,var_new,var_diag_new,K_new = gp_reg(x_train, y_train, x_test, 1, x_new[0], x_new[1])
        if np.random.uniform() <= min(lf_new/lf, 1):
            return lf_new,x_new, mu_new, var_diag_new

## make ground truth data
N = 10
high_end = 5
eps = 0.3
coef = 0.5
x_test = np.linspace(0,high_end,200)
x_sampler = np.random.rand(N) * high_end # [0,30]
y_sampler = coef * x_sampler + np.sin(x_sampler) + np.random.normal(size=N) * eps

## initial parameter value
# parameter of kernel function
theta1 = 1, #0.5
x = [0.5, 0.5] #sigma, eta
sigma0 = 0.03
sigma1 = 0.003


num_minibatch = 10 # N for fullbatch, <N for minibatch
save_interval = 5

SAMPLE = 3000
BURNIN = 100

# burn in
burn_x = np.zeros(BURNIN)
burn_y = np.zeros(BURNIN)
likelifood_array = np.zeros(BURNIN+SAMPLE)
for i in range(BURNIN):
    print("burnin i: %d, x: %.3f, y: %.3f" % (i,x[0], x[1]))
    lf,x, mu, var_diag = next(x, x_sampler, y_sampler, x_test, sigma0, sigma1)
    burn_x[i] = x[0]
    burn_y[i] = x[1]
    likelifood_array[i] = lf

sample_x = np.zeros(SAMPLE)
sample_y = np.zeros(SAMPLE)

for i in range(0, SAMPLE):
    print("mcmc i: %d, x: %.3f, y: %.3f" % (i,x[0], x[1]))
    lf,x, mu, var_diag = next(x,x_sampler, y_sampler, x_test,sigma0,sigma1)
    sample_x[i] = x[0]
    sample_y[i] = x[1]
    likelifood_array[i+BURNIN] = lf


fig = plt.figure(figsize=(6,9))

plt.subplot(311)

plt.plot(burn_x, burn_y, marker = 'x', markersize=1)
plt.plot(sample_x, sample_y, marker = 'o', markersize=1)
plt.xlabel('sigma')
plt.ylabel('eta')

plt.subplot(312)
plt.scatter(x_sampler, y_sampler, marker='x') # plot of train data
plt.plot(x_test, mu, c='Orange') # predictive line using GP regression
plt.plot(x_test, coef * x_test + np.sin(x_test),'--', c='k') # ground truth
plt.fill_between(x_test, mu-2 *var_diag, mu+2*var_diag, color='grey', alpha=.4)
#plt.title("N: {}".format(e))
plt.xlim(0,high_end)
plt.xlim(0,high_end)
#
plt.ylim(-1.5,4)

plt.subplot(313)
plt.plot(likelifood_array)
plt.xlabel("step")
plt.xlabel("likelifood")
plt.savefig("gp_mcmc.png")

#
#
#plt.savefig("images/gp_regression_grad_%03d.png" % e)
#plt.close()

#images = []
#files = glob.glob("images/gp_regression_grad*.png")
#files.sort()
##
#for f in files:
#    im = Image.open(f)
#    images.append(im)
#
#images[0].save('gp_regression_grad.gif',\
#               save_all=True, append_images=images[1::save_interval],\
#			   optimize=False, duration=1000, loop=0)




