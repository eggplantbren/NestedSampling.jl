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
	logx_threshold::Float64
	logl_threshold::Float64
end

@doc """
Constructor that only takes num_particles and mcmc_steps
as input
""" ->
function Sampler(num_particles::Int64, mcmc_steps::Int64)
	@assert (num_particles >= 1) & (mcmc_steps >= 1)
	return Sampler(num_particles, mcmc_steps,
								Array(Particle, (num_particles, )),
								zeros(num_particles), 0, 0.0, -Inf)
end

@doc """
Generate all particles from the prior
""" ->
function initialise!(sampler::Sampler)
	for(i in 1:sampler.num_particles)
		sampler.particles[i] = Particle()
		from_prior!(sampler.particles[i])
		sampler.logl[i] = log_likelihood(sampler.particles[i])
	end
	return nothing
end

@doc """
Find and save worst particle,
then generate replacement.
""" ->
function do_iteration!(sampler::Sampler)
	sampler.iteration += 1
	sampler.logx_threshold = -sampler.iteration/sampler.num_particles

	# Find index of worst particle
	worst = find_worst_particle(sampler::Sampler)

	# Write its information to the output file
	if(sampler.iteration == 1)
		f = open("sample_info.txt", "w")
		write(f, "# num_particles, iteration, log(X), log(L)\n")
	else
		f = open("sample_info.txt", "a")
	end
	write(f, string(sampler.num_particles), " ", string(sampler.iteration), " ",
			string(sampler.logx_threshold, " ", string(sampler.logl[worst]), "\n"))
	close(f)

	# Set likelihood threshold
	sampler.logl_threshold = sampler.logl[worst]
	println("Iteration ", sampler.iteration, ", log(X) = ",
				sampler.logx_threshold,	", log(L) = ", sampler.logl_threshold)

	# Clone a survivor
	if(sampler.num_particles != 1)
		which = rand(1:sampler.num_particles)
		while(which == worst)
			which = rand(1:sampler.num_particles)
		end
		sampler.particles[worst] = deepcopy(sampler.particles[which])
		sampler.logl[worst] = deepcopy(sampler.logl[which])
	end
	if(sampler.logl[worst] != log_likelihood(sampler.particles[worst]))
		println("Eh 1 ", sampler.iteration)
		exit()
	end

	# Evolve
	accepted = 0::Int64
	for(i in 1:sampler.mcmc_steps)
		proposal = deepcopy(sampler.particles[worst])
		logH = perturb!(proposal)
		if(logH > 0.0)
			logH = 0.0
		end
		logl_proposal = log_likelihood(proposal)

		if((rand() <= exp(logH)) && (logl_proposal > sampler.logl_threshold))
			sampler.particles[worst] = proposal
			sampler.logl[worst] = logl_proposal
			accepted += 1
		end
	end
	println("Accepted ", accepted, "/", sampler.mcmc_steps, " MCMC steps.\n")

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

