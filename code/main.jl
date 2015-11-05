include("Sampler.jl")

# Run Nested Sampling on the model imported from here
include("models/SpikeSlab.jl")

# Tuning parameters
num_particles = 100
mcmc_steps = 1000

# Create the sampler
sampler = Sampler(num_particles, mcmc_steps)


