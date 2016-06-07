@doc """
Heavy tailed distribution used for proposals
""" ->
function randh()
	a = randn()
	b = randn()
	c = randn()
	t = c/sqrt((a^2 + b^2)/2)
	return 10.0^(1.5 - 3*t)*randn()
end

@doc """
log(sum(exp(x)))
""" ->
function logsumexp(x::Array{Float64, 1})
	top = maximum(x)
	y = exp(x - top)
	return log(sum(y)) + top
end

@doc """
log(exp(a) - exp(b))
""" ->
function logdiffexp(a::Float64, b::Float64)
	@assert a > b
	return b + log(exp(a - b) - 1.0)
end

