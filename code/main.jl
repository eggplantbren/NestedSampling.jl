# Run Nested Sampling on the model imported from here
include("models/SpikeSlab.jl")

# Tuning parameters
num_particles = 100
mcmc_steps = 1000

# Depth in nats
max_depth = 100.0

# Do an NS run
include("Sampler.jl")
do_nested_sampling(num_particles, mcmc_steps, max_depth)

