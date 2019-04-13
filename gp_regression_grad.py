import numpy as np
import matplotlib 
matplotlib.use('agg')
import scipy.linalg as LA
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import glob


## define kernel function 
def rbf(x, x_dash,  theta1, theta2):
    return theta1 * np.exp(-1* ((x-x_dash)**2)/theta2) 

## define gradient function
def dk_dTAU(x, x_dash, n, n_dash,  tau, sigma, eta):
    if n == n_dash:
        return rbf(x, x_dash, theta1=np.exp(tau), theta2=np.exp(sigma)) - np.exp(eta)
    else:
        return rbf(x, x_dash, theta1=np.exp(tau), theta2=np.exp(sigma))

def dk_dSIGMA(x,x_dash,n,n_dash,tau,sigma,eta):
    k = rbf(x,x_dash,theta1=np.exp(tau), theta2=np.exp(sigma))


    if n ==n_dash:
        k -=  np.exp(eta)
    return k * np.exp(-1*sigma)*((x - x_dash)**2)

def dk_dETA(x,x_dash,n,n_dash,tau,sigma,eta):
    if n==n_dash:
        return np.exp(eta)
    else:
        return 0

def dL_dTHETA(K_THETA, dk_dTHETA, y):
    K_THETA_INV = LA.inv(K_THETA)
    return -1 * np.trace((K_THETA_INV).dot(dk_dTHETA)) + \
            ((K_THETA_INV.dot(y)).T).dot(dk_dTHETA).dot(K_THETA_INV.dot(y))



# whole Gaussian processs
def gp_reg(x_train, y_train, x_test, tau, sigma, eta):


    N = len(x_train)
    M = len(x_test)
    # add Gaussian noise
    K = [[rbf(ix,jx,tau,sigma) + eta if i==j else rbf(ix,jx, tau,sigma) \
            for i,ix in enumerate(x_train)] for j,jx in enumerate(x_train)]
    K = np.array(K).reshape((N,N))
    #print("K.shape {}".format(K.shape))
    
    K_inv = LA.inv(K)
    yy = K_inv.dot(y_train)

    detK = LA.det(K)

    print("detK: %.4e" % detK)

    likelifood = -1. * y_train.dot(yy) - np.log(detK)


    
    
    k = [[rbf(ix,jx,tau,sigma)  \
            for i,ix in enumerate(x_test)] for j,jx in enumerate(x_train)]
    k = np.array(k).reshape((N,M))
    
    s = [[rbf(ix,jx,tau,sigma) + eta if i==j else rbf(ix,jx,tau,sigma) \
            for i,ix in enumerate(x_test)] for j,jx in enumerate(x_test)]
    s = np.array(s).reshape((M,M))
    
    # results
    mu = (k.T).dot(yy) # mean value
    var = s - (k.T).dot(K_inv).dot(k)
    var_diag = np.diag(var) # 

    # gradients
    dKdT = [[dk_dTAU(ix,jx,i,j,tau,sigma,eta)\
            for i,ix in enumerate(x_train)] for j,jx in enumerate(x_train)]
    dKdT = np.array(dKdT).reshape(N,N)

    dKdS = [[dk_dSIGMA(ix,jx,i,j,tau,sigma,eta)\
            for i,ix in enumerate(x_train)] for j,jx in enumerate(x_train)]
    dKdS = np.array(dKdS).reshape(N,N)

    dKdE = [[dk_dETA(ix,jx,i,j,tau,sigma,eta)\
            for i,ix in enumerate(x_train)] for j,jx in enumerate(x_train)]
    dKdE = np.array(dKdE).reshape(N,N)

    dLdT = dL_dTHETA(K, dKdT, y_train)
    dLdS = dL_dTHETA(K, dKdS, y_train)
    dLdE = dL_dTHETA(K, dKdE, y_train)



    return likelifood, mu, var, var_diag, K, dKdT, dKdS, dKdE, dLdT, dLdS, dLdE


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
theta1 = 20000, #0.5
theta2 = 20000, #0.5
theta3 = 10., #0.1
tau = np.log(theta1)
sigma = np.log(theta2)
eta = np.log(theta3)

lr = 1e-3 # learning rate

num_minibatch = 10
epoch =100


for e in range(epoch):
    minibatch_index = np.random.permutation(N)[:num_minibatch]
    x_batch = x_sampler[minibatch_index]
    y_batch = y_sampler[minibatch_index]


    fig = plt.figure(figsize=(4,4))
    #print("i= {}, length= {}".format(i, len(x_sampler[:i+1])))
    likelifood, mu, var, var_diag, K, dKdT, dKdS, dKdE, dLdT, dLdS, dLdE = \
            gp_reg(x_batch, y_batch, x_test, tau, sigma, eta)

    tau += lr * (dLdT)
    sigma += lr * (dLdS) 
    eta += lr * (dLdE) 

    print("e:%d, lf: %.4f, tau:%.4f, sigma:%.4f, eta:%.4f" % \
            (e, likelifood, tau, sigma, eta))


    plt.scatter(x_sampler, y_sampler, marker='x') # plot of train data
    plt.plot(x_test, mu, c='Orange') # predictive line using GP regression
    plt.plot(x_test, coef * x_test + np.sin(x_test),'--', c='k') # ground truth
    plt.fill_between(x_test, mu-2 *var_diag, mu+2*var_diag, color='grey', alpha=.4)
    plt.title("N: {}".format(e))
    plt.xlim(0,high_end)
    plt.xlim(0,high_end)

    plt.ylim(-1.5,4)


    plt.savefig("images/gp_process_%03d.png" % e)
    plt.close()

images = []
files = glob.glob("images/gp_process*.png")
files.sort()
#
for f in files:
    im = Image.open(f)
    images.append(im)

images[0].save('gp_process_grad.gif',\
               save_all=True, append_images=images[1:],\
			   optimize=False, duration=1000, loop=0)




