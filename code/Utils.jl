@doc """
Heavy tailed distribution used for proposals
""" ->
function randh()
	t = randn()/sqrt(-log(rand()))
	return 10.0^(1.5 - 3*abs(t))*randn()
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

