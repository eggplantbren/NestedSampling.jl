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
data = readdlm("models/pareto_data.txt")

# Convert from 2D to 1D
data = data[:,1]
N = length(data)
data_min = minimum(data)

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
    i = rand(1:num_params)
    particle.us[i] += randh()
    particle.us[i] = mod(particle.us[i], 1.0)
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

    # Lognormal with median 0 (for the log) and width 3 (for the log10)
    # This is for the xmin parameter of the Pareto
    params[1] = 10.0^(3.0*quantile(normal, us[1]))

    # Slope parameter
    # Uniform(0, 5) prior
    params[2] = 5.0*us[2]

    return params
end



"""
Evaluate the log likelihood
"""
function log_likelihood(particle::Particle) :: Float64
    x_min = particle.params[1]
    alpha = particle.params[2]

    # If x_min is above any data, the likelihood is zero
    if(x_min > data_min)
        return -1E300
    end

    return N*log(alpha) + alpha*N*log(x_min) - (alpha+1.0)*sum(log.(data))
end




