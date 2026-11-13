export randErdosRenyi, randRegularGraph

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

"""
    randRegularGraph([rng,] n::Int, d::Int)

Generate the adjacency matrix of a uniformly random simple
`d`-regular graph on `n` vertices, as a symmetric `n × n` `Int`
matrix with zero diagonal and every row summing to `d`. Requires
`0 <= d <= n - 1` and `n * d` even (the handshaking lemma); both are
checked and throw `ArgumentError` otherwise.

The sampler is the configuration (pairing) model conditioned on
simplicity: the `d` stubs of every vertex are paired uniformly at
random, and the pairing is restarted whenever it produces a loop or
a multiple edge. Conditioned on simplicity every simple `d`-regular
graph arises from the same number of pairings, so the output is
exactly uniform — no MCMC burn-in or bias. The price is the restart
rate: for fixed `d` the probability that a random pairing is simple
tends to `exp(-(d^2 - 1) / 4)` as `n` grows, so the expected number
of attempts is about `exp((d^2 - 1) / 4)` — a handful for `d = 3`,
dozens for `d = 4`, hundreds for `d = 5`. The sampler is practical
in that range; it is not the tool for large `d`.

The spectrum of the output has the exact trivial eigenvalue `d`
(one per connected component, almost surely one), and the remaining
eigenvalues — the bulk — converge to the [`KestenMcKay`](@ref) law
as `n` grows.

# Examples
```julia
A = randRegularGraph(60, 3)
sum(A, dims = 2)   # every entry is 3
```
"""
function randRegularGraph(rng::AbstractRNG, n::Int, d::Int)
    (n >= 1 && 0 <= d <= n - 1) || throw(ArgumentError(
        "randRegularGraph requires 0 <= d <= n - 1, got n = $n, d = $d"))
    iseven(n * d) || throw(ArgumentError(
        "randRegularGraph requires n * d even (handshaking lemma), got n = $n, d = $d"))
    stubs = repeat(1:n, inner = d)
    A = zeros(Int, n, n)
    while true
        shuffle!(rng, stubs)
        fill!(A, 0)
        ok = true
        for k in 1:2:length(stubs)
            i, j = stubs[k], stubs[k + 1]
            if i == j || A[i, j] == 1
                ok = false
                break
            end
            A[i, j] = 1
            A[j, i] = 1
        end
        ok && return A
    end
end

randRegularGraph(n::Int, d::Int) = randRegularGraph(Random.default_rng(), n, d)

