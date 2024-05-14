include("../Utils.jl")

using DelimitedFiles
using Distributions
import Base.string

num_params = 3 :: Int64

"""
An object of this class represents a point in parameter space.
There are functions defined to evaluate the log likelihood and
move around.
"""
mutable struct Particle
    us     :: Vector{Float64}
    params :: Vector{Float64}
end

"""
Load the data
"""
data = readdlm("models/road.txt")
N = size(data)[1]

"""
A constructor. Makes params have length num_params
"""
function Particle()
    return Particle(Vector{Float64}(undef, num_params),
                    Vector{Float64}(undef, num_params))
end

"""
Generate params from the prior
"""
function from_prior!(particle::Particle)
    particle.us = rand(num_params)
    particle.params = us_to_params(particle.us)
end

"""
Do a metropolis proposal.
"""
function perturb!(particle::Particle) :: Float64

    # Perturb more than one parameter at a time
    reps = length(particle.us)^rand()
    reps = Int64(floor(reps))

    for rep in 1:reps
        i = rand(1:num_params)
        particle.us[i] += randh()
        particle.us[i] = mod(particle.us[i], 1.0)
    end
    particle.params = us_to_params(particle.us)

    return 0.0
end

"""
Convert to string, for output to sample.txt
"""
function string(particle::Particle)
    return join([string(x, " ") for x in particle.params])
end


# A standard normal distribution
normal = Normal()

function us_to_params(us::Vector{Float64}) :: Vector{Float64}
    params = Vector{Float64}(undef, num_params)
    params[1] = 1000*quantile(normal, us[1]) # Like 1000*qnorm
    params[2] = 1000*quantile(normal, us[2])
    params[3] = exp(-10 + 20*us[3])
    return params
end



"""
Evaluate the log likelihood
"""
function log_likelihood(particle::Particle) :: Float64
    beta0 = particle.params[1]
    beta1 = particle.params[2]
    sigma = particle.params[3]
    mu = beta0 .+ beta1 .* data[:,1]
    return -0.5*N*log(2*pi*sigma^2) -  0.5*sum((data[:,2] .- mu).^2)/sigma^2
end




