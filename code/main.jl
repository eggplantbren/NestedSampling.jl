# Use matplotlib for plotting
using PyCall
@pyimport matplotlib.pyplot as plt

# Run Nested Sampling on the model imported from here
include("models/SpikeSlab.jl")

# Import sampler type
include("Sampler.jl")

# Tuning parameters
num_particles = 100
mcmc_steps = 1000

# Depth in nats
max_depth = 100.0

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

plt.ion()
for(i in 1:steps)
	(keep[i, 1], keep[i, 2]) = do_iteration!(sampler)

	if(rem(i, plot_skip) == 0)
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
		lower = logl_sorted[1 + Int64(i/20)]
		plt.ylim([lower, logl_sorted[end] + 0.05*(logl_sorted[end] - lower)])

		plt.subplot(2, 1, 2)
		plt.plot(keep[1:i], exp(log_post - maximum(log_post)), "bo-", markersize=1)
		plt.xlabel("\$\\ln(X)\$")
		plt.ylabel("Relative posterior weights")

		plt.draw()
	end
end

println("Done!")
plt.ioff()
plt.show()

