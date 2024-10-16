# Run Nested Sampling on the model imported from here
include("Utils.jl")

include("models/StraightLine.jl")

# Tuning parameters
num_particles = 100
mcmc_steps = 1000

# Depth in nats
depth = 30.0

# Do an NS run
include("Sampler.jl")
do_nested_sampling(num_particles, mcmc_steps, depth)

