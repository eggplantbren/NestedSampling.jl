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
	i = randint(len(particle.params))
	particle.params[i] += randh()
	particle.params[i] = mod(particle.params[i] + 0.5, 1.0) - 0.5
	return 0.0
end

@doc """
Evaluate the log likelihood
""" ->
function log_likelihood(particle::Particle)
	logL = 0.0
	for(i in 1:length(particle.params))
		logL += -0.5*particle.params[i]^2
	end
	return logL
end

