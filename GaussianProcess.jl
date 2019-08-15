
import Pkg
Pkg.add("LinearAlgebra")
Pkg.add("Random")
Pkg.add("Plots")

using Plots
gr()
using Random
rng = MersenneTwister(1234)
using LinearAlgebra

## define kernel function
function rbf(x, x_dash, theta1=theta1, theta2=theta2)
    theta1 * exp(-1* ((x-x_dash)^2)/theta2)
end
    

# whole Gaussian processes
function gp_reg(x_train, y_train, x_test, t1, t2, t3)
    N = length(x_train)
    M = length(x_test)
    
    # Kernel matrix between sample and sample
    K = zeros((N,N))
    for i in 1:N
        for j in 1:N
            # for diagonal components, noise is add
            if i==j
                K[i,j]= rbf(x_train[i], x_train[j],t1,t2) +t3
            else
                K[i,j]= rbf(x_train[i], x_train[j],t1,t2)
            end
        end
    end
    
    K_inv= inv(K)
    yy = K_inv * (y_train)
    
    # Kernel Matrix between sample and test
    k = zeros((N,M))
    for i in 1:N
        for j in 1:M
            k[i,j] = rbf(x_train[i], x_test[j],t1,t2)
        end
    end
        
    # Kernel Matrix between sample and test
    s = zeros((M,M))
    for i in 1:M
        for j in 1:M
                s[i,j] = rbf(x_test[i], x_test[j], t1, t2)    
        end
    end
    
    # results
    mu = transpose(k) * yy
    var = s - transpose(k) * K_inv * k
    var_diag = diag(var)
    mu, var, var_diag
end


# sampling parameter
N = 20    # num of samples
sigma = 0.2    # 
coef = 0.5
x_test = LinRange(0,7,200)
x_sampler =  rand!(rng, zeros(N)) * 5
y_sampler = coef * x_sampler + sin.(x_sampler) + randn!(rng, zeros(N)) * sigma

# parameter of kernel function
theta1 = 1
theta2 = 0.4
theta3 = 0.3

# execute gaussian process over samples
mu, var, var_diag = gp_reg(x_sampler, y_sampler, x_test, theta1, theta2, theta3)
scatter(x_sampler, y_sampler, label="sample", ylims=(0,5), markersize=2.5)

plot!(x_test, coef .* x_test + sin.(x_test), label="GroundTruth")
plot!(x_test, mu, ribbon=(var_diag, var_diag), label="predict", fillalpha=0.2)


