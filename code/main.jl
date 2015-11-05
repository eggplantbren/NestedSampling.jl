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
max_depth = 1000.0

# Number of NS iterations
steps = Int64(max_depth*mcmc_steps)

# Create the sampler
sampler = Sampler(num_particles, mcmc_steps)
initialise!(sampler)

# Do 'steps' iterations of NS
# Storage for results
steps = Int64(max_depth)*num_particles
plot_skip = num_particles
keep = Array(Float64, (steps, ))

plt.ion()
for(i in 1:steps)
	keep[i] = do_iteration!(sampler)

	if(rem(i, plot_skip) == 0)
		plt.hold(false)
		plt.plot(-(1:i)/num_particles, keep[1:i], "bo-", markersize=3)
		plt.xlabel("\$\\ln(X)\$")
		plt.ylabel("\$\\ln(L)\$")
		plt.draw()
	end
end

plt.ioff()
plt.show()

