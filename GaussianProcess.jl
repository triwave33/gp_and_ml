
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

## define kernel function
function rbf(x, x_dash, theta1=theta1, theta2=theta2)
    theta1 * exp(-1* ((x-x_dash)^2)/theta2)
end
    

# whole Gaussian processes
function gp_reg(x_train, y_train, x_test, t1, t2, t3)
    N = length(x_train)
    M = length(x_test)
    
    # Kernel matrix between sample and sample
    K = rbf.(x_train, x_train') + t3 * I
    
    K_inv= inv(K)
    yy = K_inv * (y_train)
    
    # Kernel Matrix between sample and test
    k = rbf.(x_train, x_test') 
        
    # Kernel Matrix between sample and test
    s = rbf.(x_test, x_test') + t3 * I
    
    # results
    mu = transpose(k) * yy
    var = s - transpose(k) * K_inv * k
    var_diag = diag(var)
    mu, var, var_diag
end

# parameter
N = 10    # num of samples
sigma = 0.2    # 
coef = 0.5
x_test = LinRange(0,7,200)
x_sampler =  rand!(rng, zeros(N)) * 5
y_sampler = coef * x_sampler + sin.(x_sampler) + randn!(rng, zeros(N)) * sigma

# normalization
y_std =  std(y_sampler)
y_mean = mean(y_sampler)
y_sampler_norm =( y_sampler .- y_mean) ./ y_std


# parameter of kernel function
theta1 = 1
theta2 = 0.4
theta3 = 0.1

mu, var, var_diag = gp_reg(x_sampler, y_sampler, x_test, theta1, theta2, theta3)
scatter(x_sampler, y_sampler, label="sample", ylims=(0,5), markersize=2.5)

plot!(x_test, coef .* x_test + sin.(x_test), label="GroundTruth")
plot!(x_test, mu, ribbon=(var_diag, var_diag), label="predict", fillalpha=0.2)

mu_norm, var_norm, var_diag_norm = gp_reg(x_sampler, y_sampler_norm, x_test, theta1, theta2, theta3)

mu = mu_norm .* y_std .+ y_mean
var_diag = var_diag_norm 
scatter(x_sampler, y_sampler, label="sample", ylims=(0,5), markersize=2.5)

plot!(x_test, coef .* x_test + sin.(x_test), label="GroundTruth")
plot!(x_test, mu, ribbon=(var_diag, var_diag), label="predict", fillalpha=0.2)


