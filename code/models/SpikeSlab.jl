@doc """
An object of this class represents a point in parameter space.
There are functions defined to evaluate the log likelihood and
move around.
""" ->
type Particle
	params::Vector{Float64}
end

@doc """
A constructor. Makes params have length 20
"""
function Particle()
	return Particle(Array(Float64, (20, )))
end

