@doc """
Sampler class
""" ->
type Sampler
	num_particles::Int64
	mcmc_steps::Int64
	particles::Vector{Particle}

	# Current iteration
	iteration::Int64
end

@doc """
Constructor that only takes num_particles and mcmc_steps
as input
""" ->
function Sampler(num_particles::Int64, mcmc_steps::Int64)
	@assert (num_particles >= 1) & (mcmc_steps >= 1)
	return Sampler(num_particles, mcmc_steps,
								fill(Particle(), num_particles), 0)
end

@doc """
Generate all particles from the prior
""" ->
function initialise!(sampler::Sampler)
	for(i in 1:sampler.num_particles)
		sampler.particles[i] = Particle()
		from_prior!(sampler.particles[i])
	end
	sampler.iteration = 0
	return nothing
end

