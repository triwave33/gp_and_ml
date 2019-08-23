
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
using LinearAlgebra:diag, I, det, dot
using Statistics
using Distributions
using Printf
using ProgressBars
using ProgressMeter

# kernel calc mode
#m = ARGS[1]
m = "2D"
println(m)

## define kernel function
function rbf(x, x_dash, theta1=theta1, theta2=theta2)
    theta1 * exp(-1* ((x-x_dash)^2)/theta2)
end

function kernel_matrix(x, x_dash, theta1, theta2, theta3, m, add_theta3)
  if (add_theta3)
    if m == "2D"
      return rbf.(x, x_dash',theta1, theta2) + theta3 * I
    elseif m == "list"
      K = [[(if i==j rbf(ix,jx,theta1,theta2) + theta3 else rbf(ix,jx, theta1,theta2) end) for (i,ix) in enumerate(x_dash)] for (j,jx) in enumerate(x)]
      K = hcat(K...)
      return K'   # return transposed form
    elseif m == "for"
      K = zeros(length(x),length(x_dash))
      for (i,ix) in enumerate(x)
        for (j,jx) in enumerate(x_dash)
          if i==j
            K[i,j] =rbf(ix,jx,theta1,theta2) + theta3
          else
              K[i,j] =rbf(ix,jx,theta1,theta2)
          end
        end
      end
      return K
    end
  else
    if m == "2D"
      return rbf.(x, x_dash',theta1, theta2) 
    elseif m == "list"
      K = [[rbf(ix,jx,theta1,theta2) for (i,ix) in enumerate(x_dash)] for (j,jx) in enumerate(x)]
      K = hcat(K...)
      return K'   # return transpose form
    elseif m == "for"
      K = zeros(length(x),length(x_dash))
      for (i,ix) in enumerate(x)
        for (j,jx) in enumerate(x_dash)
              K[i,j] =rbf(ix,jx,theta1,theta2)
          end
        end
      end
      return K
    end
  end


# whole Gaussian processes
function gp_reg(x_train, y_train, x_test, tau, sigma, eta, m)
    theta1 = exp(tau)
    theta2 = exp(sigma)
    theta3 = exp(eta)

    N = length(x_train)
    M = length(x_test)

   
    # Kernel Matrix between sample and sample
    K = kernel_matrix(x_train, x_train, theta1, theta2, theta3, m, true) 
    K_inv= inv(K)
    yy = K_inv *  y_train

    detK = det(K)

    likelifood = -1 .* dot(y_train, yy) - log(detK)

    # Kernel Matrix between sample and test
    k = kernel_matrix(x_train, x_test, theta1, theta2, theta3, m, false) 

    # Kernel Matrix between test and test
    s = kernel_matrix(x_test, x_test, theta1, theta2, theta3, m, false) 

    # results
    mu = k' * yy
    var = s - transpose(k) * K_inv * k
    var_diag = diag(var)
    return likelifood, mu, var, var_diag, K
end

function next(theta2,theta3, x_train, y_train, x_test, mcmc_s1, mcmc_s2, m)
    while true
        params_new = [0.0,0.0]

        theta2_new = theta2 + randn!(rng, zeros(1))[1] * mcmc_s1
        theta3_new = theta3 + randn!(rng, zeros(1))[1] * mcmc_s2
        lf_log, mu, var, var_diag, K = gp_reg(x_train, y_train, x_test, 0, theta2, theta3, m)
        lf_new_log, mu_new, var_new, var_diag_new, K_new = gp_reg(x_train, y_train, x_test, 0, theta2_new, theta3_new, m)
        lf = exp(lf_log)
        lf_new = exp(lf_new_log)
        #@printf("    lf: %.3e, lf_new: %3e, p: %.2e", lf,lf_new,lf_new/lf)
        if rand!(rng, zeros(1))[1] <= min(lf_new/lf, 1)
            return lf_new, theta2_new, theta3_new, mu_new, var_diag_new
        end
    end
end

function mcmc(iter_num, theta2, theta3, x_sample, y_sample, x_test, mcmc_s1, mcmc_s2, m, verbose)
  #initialize 
  lf = 0
  mu = zeros(length(x_test))
  var_diag = zeros(length(x_test))
  theta2_array = zeros(iter_num)
  theta3_array = zeros(iter_num)
  lf_array = zeros(iter_num)
  
  if verbose
    iter = tqdm(1:iter_num)
  else
    iter = 1:iter_num
  end

  iter = Progress(iter_num, dt=0.1, barlen=50) # ProgressMeter
  for i in 1:iter_num # ProgressMeter
  #for i in iter # ProgressBar
      #@printf("burnin i: %d, x: %.3f, y: %.3f", i, x[1], x[2])
      lf,theta2,theta3,mu,var_diag = next(theta2,theta3, x_sample, y_sample, x_test, mcmc_s1, mcmc_s2,m)
      theta2_array[i] = theta2
      theta3_array[i] = theta3
      likelifood_array[i] = lf
      next!(iter) # ProgressMeter
  end
  return theta2_array, theta3_array, likelifood_array, lf, mu, var_diag
end




# parameter
N = 20    # num of samples
high_end = 5
eps = 0.3
coef = 0.5
line_space = 100
x_test = LinRange(0,high_end,line_space)
x_sample =  rand!(rng, zeros(N)) * high_end
y_sample = coef * x_sample + sin.(x_sample) + randn!(rng, zeros(N)) * eps

## initial parameter value
# parameter of kernel function
theta1 = 1
theta2  = rand(Uniform(-2,4))
theta3  = rand(Uniform(-2,4))
mcmc_s1 = 0.05
mcmc_s2 = 0.10


BURNIN = 0
SAMPLE = 1000

save_interval = 100

# burn in
burn_x = zeros(BURNIN)
burn_y = zeros(BURNIN)
likelifood_array = zeros(BURNIN + SAMPLE)

theta2_array, theta3_array, likelifood_array, lf, mu, var_diag = mcmc(SAMPLE, theta2, theta3, x_sample, y_sample, x_test, mcmc_s1, mcmc_s2,m, true)


# plot
scatter(x_sample, y_sample, label="sample", ylims=(0,5), markersize=2.5)

plot!(x_test, coef .* x_test + sin.(x_test), label="GroundTruth")
plot!(x_test, mu, ribbon=(var_diag, var_diag), label="predict", fillalpha=0.2)


