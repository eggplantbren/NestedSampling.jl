include("../Utils.jl")

@doc """
An object of this class represents a point in parameter space.
There are functions defined to evaluate the log likelihood and
move around.
""" ->
type Particle
	params::Vector{Float64}
end

@doc """
A constructor. Makes params have length 20
""" ->
function Particle()
	return Particle(Array(Float64, (20, )))
end

@doc """
Generate params from the prior
""" ->
function from_prior!(particle::Particle)
	particle.params = -0.5 + rand(length(particle.params))
	return nothing
end

@doc """
Do a metropolis proposal. Return log(hastings factor for prior sampling)
""" ->
function perturb!(particle::Particle)
	i = rand(1:length(particle.params))
	particle.params[i] += randh()
	particle.params[i] = mod(particle.params[i] + 0.5, 1.0) - 0.5
	return 0.0
end

@doc """
Evaluate the log likelihood
""" ->
function log_likelihood(particle::Particle)
	logL1 = -length(particle.params)*0.5*log(2*pi*0.1^2)
	logL2 = -length(particle.params)*0.5*log(2*pi*0.01^2)
	for i in 1:length(particle.params)
		logL1 += -0.5*(particle.params[i]/0.1)^2
		logL2 += -0.5*((particle.params[i] - 0.031)/0.01)^2
	end
	return logsumexp([logL1, logL2 + log(100.0)])
end

@doc """
Convert to string, for output to sample.txt
"""
import Base.string
function string(particle::Particle)
	return join([string(signif(x, 6), " ") for x in particle.params])
end

