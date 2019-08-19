
let
    @eval using Pkg
    pkgs = ["Plots"]
    for pkg in pkgs
        if Base.find_package(pkg) === nothing
            Pkg.add(pkg)
        end
    end
end


using Plots
gr()
using Random
rng = MersenneTwister(1234)
using LinearAlgebra:diag, I
using Statistics
using Distributions

## define kernel function
function rbf(x, x_dash, theta1=theta1, theta2=theta2)
    theta1 * exp(-1* ((x-x_dash)^2)/theta2)
end


# whole Gaussian processes
function gp_reg(x_train, y_train, x_test, tau, sigma, eta)
    theta1 = exp(tau)
    theta2 = exp(sigma)
    theta3 = exp(eta)
    N = length(x_train)
    M = length(x_test)

    # Kernel matrix between sample and sample
    K = rbf.(x_train, x_train') + theta3 * I

    K_inv= inv(K)
    yy = K_inv * (y_train)

    detK = det(K)

    likelifood = -1 .* y_train * yy .- log(detK)

    # Kernel Matrix between sample and test
    k = rbf.(x_train, x_test')

    # Kernel Matrix between sample and test
    s = rbf.(x_test, x_test') 

    # results
    mu = transpose(k) * yy
    var = s - transpose(k) * K_inv * k
    var_diag = diag(var)
    return likelifood, mu, var, var_diag
end

function next(x, x_train, y_train, x_test, sigma1, sigma2)
    while True
        x_new = [0,0]
        x_new[1] = x[1] + randn!(rng, zeros(1)) * sigma1
        x_new[2] = x[2] + randn!(rng, zeros(1)) * sigma2
        lf_log, mu, var, var_diag, K = gp_reg(x_train, y_train, x_test, 0, x[1], x[2])
        lf_new_log, mu_new, var_new, var_diag_new, K_new = gp_reg(x_train, y_train, x_test, 0, x[1], x[2])
        lf = exq(lf_log)
        lf_new = exp(lf_new_log)
        printf("    lf: %.3e, lf_new: %3e, p: %.2f", lf,lf_new,lf_new/lf)
        if rand!(rng, zeros(1)) <= min(lf_new/lf, 1)
            return lf_new, x_new, mu_new, var_diag_new
        end
    end
end





# parameter
N = 100    # num of samples
high_end = 5
eps = 0.3
coef = 0.5
x_test = LinRange(0,7,200)
x_sampler =  rand!(rng, zeros(N)) * high_end
y_sampler = coef * x_sampler + sin.(x_sampler) + randn!(rng, zeros(N)) * eps

## initial parameter value
# parameter of kernel function
theta1 = 1
x = rand(Uniform(-2,4), 2)
println(x[1])
println(x[2])
sigma1 = 0.01
sigma2 = 0.02

num_minibatch = N # N for fullbatch, <N for minibatch
save_interval = 100

SAMPLE = 3000
BURNIN = 10

# burn in
burn_x = zeros(BURNIN)
burn_y = zeros(BURNIN)
likelifood_array = zeros(BURNIN + SAMPLE)
for i in 1:BURNIN
    printf("burnin i: %d, x: %.3f, y: %.3f", i, x[1], x[2])
    lf,x,mu,var_dig = next(x, x_sampler, y_sampler, x_test, sigma1, sigma2)
    burn_x[i] = x[1]
    burn_y[i] = x[2]
    likelifood_array[i] = of
end

sample_x = zeros(SAMPLE)
sample_y = zeros(SAMPLE)

for i in 1:SAMPLE
    printf("mcmc i: %d, x: %.3f, y: %.3f", i, x[1], x[2])
    lf,x,mu,var_diag = next(x,xsampler, y_sampler,x_test,sigma1,sigma2)
    sample_x[i] = x[1]
    sample_y[i] = x[2]
    likelifood_array[i+BURNIN] = lf
end



## normalization
#y_std =  std(y_sampler)
#y_mean = mean(y_sampler)
#y_sampler_norm =( y_sampler .- y_mean) ./ y_std
#
#
## parameter of kernel function
#theta1 = 1
#theta2 = 0.4
#theta3 = 0.1
#
#mu, var, var_diag = gp_reg(x_sampler, y_sampler, x_test, theta1, theta2, theta3)
#scatter(x_sampler, y_sampler, label="sample", ylims=(0,5), markersize=2.5)
#
#plot!(x_test, coef .* x_test + sin.(x_test), label="GroundTruth")
#plot!(x_test, mu, ribbon=(var_diag, var_diag), label="predict", fillalpha=0.2)
#
#mu_norm, var_norm, var_diag_norm = gp_reg(x_sampler, y_sampler_norm, x_test, theta1, theta2, theta3)
#
#mu = mu_norm .* y_std .+ y_mean
#var_diag = var_diag_norm
#scatter(x_sampler, y_sampler, label="sample", ylims=(0,5), markersize=2.5)
#
#plot!(x_test, coef .* x_test + sin.(x_test), label="GroundTruth")
#plot!(x_test, mu, ribbon=(var_diag, var_diag), label="predict", fillalpha=0.2)
#
#
