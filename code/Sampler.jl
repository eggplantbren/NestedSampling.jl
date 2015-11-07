using PyCall
@pyimport matplotlib.pyplot as plt

@doc """
Sampler class
""" ->
type Sampler
	num_particles::Int64
	mcmc_steps::Int64
	particles::Vector{Particle}
	logl::Vector{Float64}
	tiebreakers::Vector{Float64}

	# Current iteration
	iteration::Int64

	# Current log likelihood threshold
	logx_threshold::Float64
	logl_threshold::Float64
	tb_threshold::Float64
end

@doc """
Constructor that only takes num_particles and mcmc_steps
as input
""" ->
function Sampler(num_particles::Int64, mcmc_steps::Int64)
	@assert (num_particles >= 1) & (mcmc_steps >= 1)
	return Sampler(num_particles, mcmc_steps,
								Array(Particle, (num_particles, )),
								zeros(num_particles), zeros(num_particles),
								0, 0.0, -Inf, 0.0)
end

@doc """
Generate all particles from the prior
""" ->
function initialise!(sampler::Sampler)
	for(i in 1:sampler.num_particles)
		sampler.particles[i] = Particle()
		from_prior!(sampler.particles[i])
		sampler.logl[i] = log_likelihood(sampler.particles[i])
		sampler.tiebreakers[i] = rand()
	end
	return nothing
end

@doc """
Find and save worst particle,
then generate replacement.
""" ->
function do_iteration!(sampler::Sampler, verbose::Bool)
	sampler.iteration += 1
	sampler.logx_threshold = -sampler.iteration/sampler.num_particles

	# Find index of worst particle
	worst = find_worst_particle(sampler::Sampler)

	# Write its information to the output files
	if(sampler.iteration == 1)
		f = open("sample_info.txt", "w")
		write(f, "# num_particles, iteration, log(X), log(L)\n")
		f2 = open("sample.txt", "w")
		write(f2, "# The samples themselves. Use log(X) from sample_info.txt as un-normalised prior weights.\n")
	else
		f = open("sample_info.txt", "a")
		f2 = open("sample.txt", "a")
	end
	write(f, string(sampler.num_particles), " ", string(sampler.iteration), " ",
			string(sampler.logx_threshold, " ", string(sampler.logl[worst]), "\n"))
	close(f)
	write(f2, string(string(sampler.particles[worst]), "\n"))
	close(f2)

	# Set likelihood threshold
	sampler.logl_threshold = sampler.logl[worst]
	if(verbose)
		println("Iteration ", sampler.iteration, ", log(X) = ",
				sampler.logx_threshold,	", log(L) = ", sampler.logl_threshold)
	end

	# Clone a survivor
	if(sampler.num_particles != 1)
		which = rand(1:sampler.num_particles)
		while(which == worst)
			which = rand(1:sampler.num_particles)
		end
		sampler.particles[worst] = deepcopy(sampler.particles[which])
		sampler.logl[worst] = deepcopy(sampler.logl[which])
		sampler.tiebreakers[worst] = deepcopy(sampler.tiebreakers[which])
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
		tb_proposal = sampler.tiebreakers[worst] + randh()
		tb_proposal = mod(tb_proposal, 1.0)

		if((rand() <= exp(logH)) && is_less_than(
						(sampler.logl_threshold, sampler.tb_threshold),
						(logl_proposal, tb_proposal)))
			sampler.particles[worst] = proposal
			sampler.logl[worst] = logl_proposal
			sampler.tiebreakers[worst] = tb_proposal
			accepted += 1
		end
	end
	if(verbose)
		println("Accepted ", accepted, "/", sampler.mcmc_steps, " MCMC steps.\n")
	end

	return (sampler.logx_threshold, sampler.logl_threshold)
end

@doc """
Compare based on likelihoods first. Use tiebreakers to break a tie
""" ->
function is_less_than(x::Tuple{Float64, Float64}, y::Tuple{Float64, Float64})
	if(x[1] < y[1])
		return true
	end
	if((x[1] == y[1]) && (x[2] < y[2]))
		return true
	end
	return false
end


@doc """
Find the index of the worst particle.
"""
function find_worst_particle(sampler::Sampler)
	# Find worst particle
	worst = 1
	for(i in 2:sampler.num_particles)
		if(is_less_than((sampler.logl[i], sampler.tiebreakers[i]),
						(sampler.logl[worst], sampler.tiebreakers[worst])))
			worst = i
		end
	end
	return worst
end


@doc """
Do a Nested Sampling run.
"""
function do_nested_sampling(num_particles::Int64, mcmc_steps::Int64,
												depth::Float64; plot=true,
												verbose=true)

	# Number of NS iterations
	steps = Int64(max_depth*mcmc_steps)

	# Create the sampler
	sampler = Sampler(num_particles, mcmc_steps)
	initialise!(sampler)

	# Do 'steps' iterations of NS
	# Storage for results
	steps = Int64(max_depth)*num_particles
	plot_skip = num_particles

	# Store logX, logL
	keep = Array(Float64, (steps, 2))

	if(plot)
		plt.ion()
	end
	for(i in 1:steps)
		(keep[i, 1], keep[i, 2]) = do_iteration!(sampler, verbose)

		if(plot && (rem(i, plot_skip) == 0))
			# Prior weights
			log_prior = keep[1:i, 1] - logsumexp(keep[1:i, 1])
			# Unnormalised posterior weights
			log_post = log_prior + keep[1:i, 2]
			# log evidence and information
			logZ = logsumexp(log_post)
			post = exp(log_post - logZ)
			H = sum(post.*(log_post - logZ - log_prior))
			uncertainty = sqrt(H/num_particles)

			plt.subplot(2, 1, 1)
			plt.hold(false)
			plt.plot(keep[1:i, 1], keep[1:i, 2], "bo-", markersize=1)
			plt.ylabel("\$\\ln(L)\$")
			plt.title(string("\$\\ln(Z) =\$ ", signif(logZ, 6),
						" +- ", signif(uncertainty, 3),
						", \$H = \$", signif(H, 6), " nats"))

			# Adaptive ylim (exclude bottom 5%)
			logl_sorted = sort(keep[1:i, 2])
			lower = logl_sorted[1 + Int64(floor(0.05*i))]
			plt.ylim([lower, logl_sorted[end] + 0.05*(logl_sorted[end] - lower)])

			plt.subplot(2, 1, 2)
			plt.plot(keep[1:i], exp(log_post - maximum(log_post)), "bo-", markersize=1)
			plt.xlabel("\$\\ln(X)\$")
			plt.ylabel("Relative posterior weights")

			plt.draw()
		end
	end

	if(verbose)
		println("Done!")
	end
	if(plot)
		plt.ioff()
		plt.show()
	end

	# Prior weights
	log_prior = keep[:, 1] - logsumexp(keep[:, 1])
	# Unnormalised posterior weights
	log_post = log_prior + keep[:, 2]
	# log evidence and information
	logZ = logsumexp(log_post)
	post = exp(log_post - logZ)
	H = sum(post.*(log_post - logZ - log_prior))

	return (logZ, H)
end

