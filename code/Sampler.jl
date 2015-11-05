@doc """
Sampler class
""" ->
type Sampler
	num_particles::Int64
	mcmc_steps::Int64
	particles::Vector{Particle}
	logl::Vector{Float64}

	# Current iteration
	iteration::Int64

	# Current log likelihood threshold
	logl_threshold::Float64
end

@doc """
Constructor that only takes num_particles and mcmc_steps
as input
""" ->
function Sampler(num_particles::Int64, mcmc_steps::Int64)
	@assert (num_particles >= 1) & (mcmc_steps >= 1)
	return Sampler(num_particles, mcmc_steps,
								fill(Particle(), num_particles),
								Array(Float64, (num_particles, )), 0, -Inf)
end

@doc """
Generate all particles from the prior
""" ->
function initialise!(sampler::Sampler)
	for(i in 1:sampler.num_particles)
		from_prior!(sampler.particles[i])
		sampler.logl[i] = log_likelihood(sampler.particles[i])
	end
	sampler.iteration = 0
	sampler.logl_threshold = -Inf
	return nothing
end

@doc """
Find and save worst particle,
then generate replacement.
""" ->
function do_iteration!(sampler::Sampler, equilibrate=true)
	sampler.iteration += 1
	return nothing
end

