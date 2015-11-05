@doc """
Sampler class
""" ->
type Sampler
	num_particles::Int64
	mcmc_steps::Int64

	particles::Vector{Particle}
end

@doc """
Constructor that only takes num_particles and mcmc_steps
as input
""" ->
function Sampler(num_particles::Int64, mcmc_steps::Int64)
	@assert (num_particles >= 1) & (mcmc_steps >= 1)
	return Sampler(num_particles, mcmc_steps,
								Array(Particle, (num_particles, )))
end


