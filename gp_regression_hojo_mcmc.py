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


mode = "induce" # pure or induce
# whole Gaussian processs
def gp_reg(x_train, y_train, x_induce, x_test, tau, sigma, eta):
    theta1 = np.exp(tau)
    theta2 = np.exp(sigma)
    theta3 = np.exp(eta)

    N = len(x_train)
    M = len(x_induce)
    S = len(x_test)

    if(mode == "induce"):
        # add Gaussian noise
        #K = [[rbf(ix,jx,tau,sigma) + eta if i==j else rbf(ix,jx, tau,sigma) \
        K_NN = [[rbf(ix,jx,theta1,theta2) + theta3 if i==j else 0  # NOTE diag mat for induce variable 
                for i,ix in enumerate(x_train)] for j,jx in enumerate(x_train)]
        K_NN = np.array(K_NN).reshape((N,N))
        #print("K.shape {}".format(K.shape))

        K_NN_inv = LA.inv(K_NN)

        detK_NN = LA.det(K_NN)

        yy_NN = K_NN_inv.dot(y_train)
        likelifood = -1. * y_train.dot(yy_NN) - np.log(detK_NN)
        #print("detK: %.4e" % detK)


        K_NM = [[rbf(ix,jx,theta1,theta2)  \
                for i,ix in enumerate(x_induce)] for j,jx in enumerate(x_train)]
        K_NM = np.array(K_NM).reshape((N,M))

        K_MM = [[rbf(ix,jx,theta1,theta2)  \
                for i,ix in enumerate(x_induce)] for j,jx in enumerate(x_induce)]
        K_MM = np.array(K_MM).reshape((M,M))
        K_MM_inv = LA.inv(K_MM)

        #lam = K_NN - K_NM.dot(K_MM_inv).dot(K_NM.T)
        lam = [np.diag(K_NN)[i]  - (K_NM[i,:]).dot(K_MM_inv).dot(K_NM[i,:]) for i in range(N)]
        #print("lam shape ", len(lam))
        #print(lam)
        lam_diag = np.diag(lam)

        tmp = (K_NM.T).dot(LA.inv(lam_diag + (theta3**2)*np.identity(N)))
        Q_MM = K_MM + tmp.dot(K_NM)
        Q_MM_inv = LA.inv(Q_MM)
        u = K_MM.dot(Q_MM_inv).dot(tmp).dot(y_train)
        SIGMA_u = K_MM.dot(Q_MM_inv).dot(K_MM)
        #print(u.shape)
        #print(SIGMA_u.shape)
        #print(Q_MM.shape)
        #print("len(u): ", len(u))

        K_MS = [[rbf(ix,jx,theta1,theta2)  \
                for i,ix in enumerate(x_test)] for j,jx in enumerate(x_induce)]
        K_MS = np.array(K_MS).reshape((M,S))

        K_SS = [[rbf(ix,jx,theta1,theta2)  \
                for i,ix in enumerate(x_test)] for j,jx in enumerate(x_test)]
        K_SS = np.array(K_SS).reshape((S,S))


        f = K_MS.T.dot(K_MM_inv).dot(u)
        SIGMA_f = K_SS - K_MS.T.dot(LA.inv(SIGMA_u)).dot(K_MS)
        SIGMA_f_diag = np.diag(SIGMA_f)
        #print("len(f): ", len(f))

        mu = f
        var = SIGMA_f
        var_diag = np.diag(SIGMA_f)

    
    else: # pure
        K = [[rbf(ix,jx,theta1,theta2) + theta3 if i==j else rbf(ix,jx,theta1,theta2)  \
            for i,ix in enumerate(x_train)] for j,jx in enumerate(x_train)]
        K = np.array(K).reshape((N,N))

        detK = LA.det(K)

        K_inv = LA.inv(K) # NN
        yy = K_inv.dot(y_train) # NN N1 -> N1
        likelifood = -1. * y_train.dot(yy) - np.log(detK)

        k = [[rbf(ix,jx,theta1,theta2)  \
                for i,ix in enumerate(x_test)] for j,jx in enumerate(x_train)]
        k = np.array(k).reshape((N,S))

        s = [[rbf(ix,jx,theta1,theta2)  \
                for i,ix in enumerate(x_test)] for j,jx in enumerate(x_test)]
        s = np.array(s).reshape((S,S))


        # results
        mu = (k.T).dot(yy) # mean value
        var = s - (k.T).dot(K_inv).dot(k)
        var_diag = np.diag(var) # 

        

        #print("len(mu): ", len(mu))
    
    ii = np.random.randint(N)
    #print("i: %d, mu[i]: %.3f, f[i]: %.3f, var[i,i]: %.3f, sigma_f[i,i]: %.3f" % (ii, mu[ii], f[ii], var[ii,ii], SIGMA_f[ii,ii]))
    #print("i: %d, mu[i]: %.3f, var[i,i]: %.3ff" % (ii, mu[ii], var[ii,ii]))


    return likelifood, mu, var, var_diag

def next(x, x_train, y_train, x_induce, x_test,sigma0, sigma1):
    while True:
        x_new = [0,0]
        x_new[0] = x[0] + np.random.normal(0, sigma0)
        x_new[1] = x[1] + np.random.normal(0, sigma1)
        lf_log,mu,var,var_diag = gp_reg(x_train, y_train, x_induce, x_test, 0, x[0], x[1])
        lf_new_log,mu_new,var_new,var_diag_new = gp_reg(x_train, y_train, x_induce, x_test, 0, x_new[0], x_new[1])
        lf =np.exp(lf_log)
        lf_new =np.exp(lf_new_log)
        print("  mode: %s, lf: %.3e, lf_new: %.3e, p: %.2f" % (mode, lf, lf_new, lf_new/lf))
        
        if np.random.uniform() <= min(lf_new/lf, 1):
        #if (lf_new > lf ):
            #print('    mcmc setp proceed') 
            return lf_new, x_new, mu_new, var_diag_new
        #else:
            #print('    discarded')

## make ground truth data
N = 10
high_end = 5
eps = 0.3
coef = 0.5

linspace_num = 50
x_test = np.linspace(0,high_end, linspace_num)
x_sampler = np.random.rand(N) * high_end # [0,30]
y_sampler = coef * x_sampler + np.sin(x_sampler) + np.random.normal(size=N) * eps

## inducing point
M = 10
lowside = np.min(x_sampler)
highside = np.max(x_sampler)
x_induce = np.linspace(lowside, highside, M)

## initial parameter value
# parameter of kernel function
theta1 = 1, #0.5
#x = [0.5, 0.5] #sigma, eta
x = np.random.uniform(-0,2, size=2)
sigma0 = 0.03 
sigma1 = 0.08


num_minibatch = N # N for fullbatch, <N for minibatch NOTE not used
save_interval = 20

SAMPLE = 1000
BURNIN = 0

# burn in
burn_x = np.zeros(BURNIN)
burn_y = np.zeros(BURNIN)
likelifood_array = np.zeros(BURNIN+SAMPLE)
for i in range(BURNIN):
    print("burnin i: %d, x: %.3f, y: %.3f" % (i,x[0], x[1]))
    lf,x, mu, var_diag = next(x, x_sampler, y_sampler, x_induce, x_test, sigma0, sigma1)
    burn_x[i] = x[0]
    burn_y[i] = x[1]
    likelifood_array[i] = lf

sample_x = np.zeros(SAMPLE)
sample_y = np.zeros(SAMPLE)

for i in range(0, SAMPLE):
    print("mcmc i: %d, x: %.3f, y: %.3f" % (i,x[0], x[1]))
    lf,x, mu, var_diag = next(x,x_sampler, y_sampler, x_induce, x_test,sigma0,sigma1)
    sample_x[i] = x[0]
    sample_y[i] = x[1]
    likelifood_array[i+BURNIN] = lf
    if (i % save_interval==0):

        fig = plt.figure(figsize=(6,9))
        
        plt.subplot(411)
        
        plt.plot(sample_x, sample_y, marker = 'o', markersize=1)
        plt.xlabel('sigma')
        plt.ylabel('eta')
        
        plt.subplot(412)
        plt.scatter(x_sampler, y_sampler, marker='x') # plot of train data
        plt.plot(x_test, mu, c='Orange') # predictive line using GP regression
        plt.plot(x_test, coef * x_test + np.sin(x_test),'--', c='k') # ground truth
        plt.fill_between(x_test, mu-2 *var_diag, mu+2*var_diag, color='grey', alpha=.4)
        #plt.title("N: {}".format(e))
        plt.xlim(0,high_end)
        plt.xlim(0,high_end)
        #
        plt.ylim(-1.5,4)
        
        plt.subplot(413)
        plt.plot(likelifood_array)
        plt.xlabel("step")
        plt.xlabel("likelifood")
        
        plt.subplot(414)
        plt.plot(sample_x, c='Orange', label='theta2') # predictive line using GP regression
        plt.plot(sample_y, c='b', label='theta3') # predictive line using GP regression
        plt.xlabel("step")
        plt.ylabel("param")
        plt.savefig("images/gp_mcmc_%04d.png" % i)
        plt.close()
    

#
#
#plt.savefig("images/gp_regression_grad_%03d.png" % e)
#plt.close()

images = []
files = glob.glob("images/gp_mcmc*.png")
files.sort()
#
for f in files:
    im = Image.open(f)
    images.append(im)

images[0].save('gp_hojo_mcmc.gif',\
               save_all=True, append_images=images[1:],\
			   optimize=False, duration=500, loop=0)




