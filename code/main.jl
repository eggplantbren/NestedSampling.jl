# Run Nested Sampling on the model imported from here
include("Utils.jl")
include("models/StraightLine.jl")
include("Sampler.jl")

# Tuning parameters
num_particles = 100
mcmc_steps = 1000

# Depth in nats
max_depth = 100.0
early_termination = true

# Do an NS run
do_nested_sampling(num_particles, mcmc_steps, max_depth, early_termination)

