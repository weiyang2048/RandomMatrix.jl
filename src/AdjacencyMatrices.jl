export randErdosRenyi

"""
    randErdosRenyi(n::Int, p = 0.5)

Generate the adjacency matrix of an Erdős–Rényi random graph on `n`
vertices: an `n × n` symmetric matrix whose entries are i.i.d.
`Bernoulli(p)` (the diagonal included, via [`randHermitian`](@ref)), so
each possible edge is present independently with probability `p`.

# Examples
```julia
randErdosRenyi(6, 0.5)
```
"""
function randErdosRenyi(n::Int, p = 0.5::Float64 )
    randHermitian(Bernoulli(p),n)
end

