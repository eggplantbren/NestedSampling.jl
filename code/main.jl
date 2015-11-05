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
for(i in 1:steps)
	println(do_iteration!(sampler, i!=steps))


end

