include("../Utils.jl")

"""
An object of this class represents a point in parameter space.
There are functions defined to evaluate the log likelihood and
move around.
"""
mutable struct Particle
	params::Vector{Float64}
end

"""
A constructor. Makes params have length 20
"""
function Particle()
	return Particle(Vector{Float64}(undef, 20))
end

"""
Generate params from the prior
"""
function from_prior!(particle::Particle)
	particle.params = -0.5 .+ rand(length(particle.params))
end

"""
Do a metropolis proposal. Return log(hastings factor for prior sampling)
"""
function perturb!(particle::Particle) :: Float64
	i = rand(1:length(particle.params))
	particle.params[i] += randh()
	particle.params[i] = mod(particle.params[i] + 0.5, 1.0) - 0.5
	return 0.0
end

"""
Evaluate the log likelihood
"""
function log_likelihood(particle::Particle) :: Float64
	logL1 = -length(particle.params)*0.5*log(2*pi*0.1^2)
	logL2 = -length(particle.params)*0.5*log(2*pi*0.01^2)
	for i in 1:length(particle.params)
		logL1 += -0.5*(particle.params[i]/0.1)^2
		logL2 += -0.5*((particle.params[i] - 0.031)/0.01)^2
	end
	return logsumexp([logL1, logL2 + log(100.0)])
end

import Base.string

"""
Convert to string, for output to sample.txt
"""
function string(particle::Particle)
	return join([string(x, " ") for x in particle.params])
end

