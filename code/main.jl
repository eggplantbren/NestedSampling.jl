# Run Nested Sampling on the model imported from here
include("models/SpikeSlab.jl")

# Import sampler type
include("Sampler.jl")

# Tuning parameters
num_particles = 100
mcmc_steps = 1000

# Create the sampler
sampler = Sampler(num_particles, mcmc_steps)

