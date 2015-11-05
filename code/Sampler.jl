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
function do_iteration!(sampler::Sampler, equilibrate::Bool=true)
	# Find index of worst particle
	worst = find_worst_particle(sampler::Sampler)

	# Write its information to the output file
	if(sampler.iteration == 0)
		f = open("sample_info.txt", "w")
	else
		f = open("sample_info.txt", "a")
	end
	f = open("sample_info.txt", "w")
	write(f, string(sampler.iteration+1), " ", sampler.logl[worst])
	close(f)

	# Set likelihood threshold
	sampler.logl_threshold = sampler.logl[worst]
	if(!equilibrate)
		return sampler.logl_threshold
	end

	# Clone a survivor
	if(sampler.num_particles != 1)
		which = rand(1:sampler.num_particles)
		while(which == worst)
			which = rand(1:sampler.num_particles)
		end
		sampler.particles[worst] = sampler.particles[which]
		sampler.logl[worst] = sampler.logl[which]
	end
	which = worst

	# Evolve
	for(i in 1:sampler.mcmc_steps)
		proposal = deepcopy(sampler.particles[which])
		logH = perturb!(proposal)
		if(logH > 0.0)
			logH = 0.0
		end
		logl_proposal = log_likelihood(proposal)

		if((rand() <= exp(logH)) && (logl_proposal > sampler.logl_threshold))
			sampler.particles[which] = proposal
			sampler.logl[which] = logl_proposal
		end
	end

	sampler.iteration += 1
	return sampler.logl_threshold
end

@doc """
Find the index of the worst particle.
"""
function find_worst_particle(sampler::Sampler)
	# Find worst particle
	worst = 1
	for(i in 2:sampler.num_particles)
		if(sampler.logl[i] < sampler.logl[worst])
			worst = i
		end
	end
	return worst
end

