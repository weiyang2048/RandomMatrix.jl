# Numerical free probability: free additive and multiplicative convolution
# of probability measures, with Cauchy / R / S transforms.
#
# A `FreeMeasure` is stored as a discrete measure (nodes + weights); named
# laws are converted by trapezoid weights on a grid. All transforms are then
# exact rational functions of the discrete measure:
#     G(z)   = sum_i w_i / (z - a_i)                     (Cauchy transform)
#     psi(z) = sum_i w_i * z a_i / (1 - z a_i)           (moment transform)
#     R(w)   = K(w) - 1/w,  K = G^{<-1>}                 (R-transform)
#     S(w)   = (1 + w)/w * chi(w),  chi = psi^{<-1>}      (S-transform)
# Functional inverses are computed by Newton iteration.
#
# freeAdd uses the Bercovici-Voiculescu subordination fixed point
#     G(z) = G_mu(w1) = G_nu(w2),   w1 = z - (w2 - 1/G_nu(w2)),
#                                   w2 = z - (w1 - 1/G_mu(w1)),
# contractive for Im z > 0; the density is -(1/pi) Im G(x + i eps).
# freeMul uses the S-transform product chi_{mu boxtimes nu}(w) =
# (1 + w)/w * chi_mu(w) chi_nu(w), solved by nested Newton iteration.
# Atoms are computed exactly (Bercovici-Voiculescu): for freeAdd an atom
# at a + b of mass mu({a}) + nu({b}) - 1 whenever that is positive; for
# freeMul an atom at a*b under the same rule, and at 0 of mass
# max(mu({0}), nu({0})).

export FreeMeasure, semicircleLaw, freePoissonLaw, diracLaw, discreteLaw,
    empiricalLaw, cauchyTransform, rTransform, sTransform, freeAdd, freeMul,
    freeDensity, freeAtoms, freeMean, freeVar

"""
    FreeMeasure(nodes, weights, atoms = Pair{Float64,Float64}[])

A probability measure represented by discrete quadrature `nodes` and
`weights` (summing to the mass of the absolutely continuous part) plus a
list of `atoms` (location => mass). This is the working representation
for the free-probability transforms and convolutions: on it, the Cauchy
transform is an exact rational function.
"""
struct FreeMeasure
    nodes::Vector{Float64}
    weights::Vector{Float64}
    atoms::Vector{Pair{Float64,Float64}}
    function FreeMeasure(nodes, weights, atoms=Pair{Float64,Float64}[])
        length(nodes) == length(weights) ||
            throw(ArgumentError("nodes and weights must have equal length"))
        new(float.(nodes), float.(weights), atoms)
    end
end

# Append the atoms (as extra nodes carrying their exact mass) and
# normalize the weights to a probability measure.
function _normalize_measure(nodes, w, atoms)
    for (a, m) in atoms
        push!(nodes, a)
        push!(w, m)
    end
    tot = sum(w)
    w ./= tot
    return FreeMeasure(nodes, w, atoms)
end

# Convert a density on a grid to a discrete measure (trapezoid weights),
# optionally with distinguished atoms (also added as nodes).
function _density_measure(xs, dens, atoms)
    n = length(xs)
    w = zeros(n)
    for i in 1:n
        h = (i > 1 ? xs[i] - xs[i-1] : 0.0) + (i < n ? xs[i+1] - xs[i] : 0.0)
        w[i] = dens[i] * h / 2
    end
    return _normalize_measure(collect(xs), w, atoms)
end

"""
    semicircleLaw(sigma=1.0; center=0.0, npoints=801)

The semicircle law with variance `sigma^2`, as a [`FreeMeasure`](@ref):
density `sqrt(4 sigma^2 - (x-center)^2) / (2 pi sigma^2)` on
`[center - 2 sigma, center + 2 sigma]`. Its R-transform is
`R(w) = sigma^2 w`.
"""
function semicircleLaw(σ::Real=1.0; center::Real=0.0, npoints::Integer=801)
    xs = collect(range(center - 2σ, center + 2σ; length=npoints))
    dens = [sqrt(max(4σ^2 - (x - center)^2, 0.0)) / (2π * σ^2) for x in xs]
    return _density_measure(xs, dens, Pair{Float64,Float64}[])
end

"""
    freePoissonLaw(gamma; scale=1.0, npoints=1201)

The free Poisson law with parameter `gamma` - the Marcenko-Pastur law
with ratio `gamma`, with the variable multiplied by `scale`. Support
`scale * (1 +- sqrt(gamma))^2`, mean `scale`; for `gamma > 1` there is an
atom at 0 of mass `1 - 1/gamma`. Its transforms (for `scale = 1`) are
`R(w) = 1 / (1 - gamma * w)` and `S(w) = 1 / (1 + gamma * w)`. The
discretization uses a Chebyshev grid in the support variable, so the
density spike at the lower edge for `gamma` near 1 (and the singular
density at `gamma = 1`) is resolved accurately.
"""
function freePoissonLaw(γ::Real; scale::Real=1.0, npoints::Integer=1201)
    # Build the law on the Chebyshev grid y = mid - half*cos(theta) in the
    # unit-scale variable: the transformed density
    #     rho(y(theta)) * y'(theta) = half^2 sin^2(theta) / (2 pi gamma y)
    # is smooth for every gamma, whereas on a uniform grid the density
    # spike at the lower edge is unresolved for gamma near 1 (the mean
    # came out 2.3% high at gamma = 0.99) and at gamma = 1 the density
    # itself is NaN at 0, which poisoned every weight with NaN.
    lo, hi = (1 - sqrt(γ))^2, (1 + sqrt(γ))^2
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    θs = range(0, π; length=npoints)
    ys = mid .- half .* cos.(θs)
    g = [y > 0 ? half^2 * sin(θ)^2 / (2π * γ * y) : half / (π * γ)
         for (θ, y) in zip(θs, ys)]
    w = zeros(npoints)
    for i in 1:npoints
        h = (i > 1 ? θs[i] - θs[i-1] : 0.0) + (i < npoints ? θs[i+1] - θs[i] : 0.0)
        w[i] = g[i] * h / 2
    end
    atoms = γ > 1 ? [0.0 => 1 - 1 / γ] : Pair{Float64,Float64}[]
    return _normalize_measure(scale .* ys, w, atoms)
end

"""
    diracLaw(a)

The point mass at `a`, as a [`FreeMeasure`](@ref).
"""
diracLaw(a::Real) = FreeMeasure([float(a)], [1.0], [float(a) => 1.0])

"""
    discreteLaw(points, masses)

A discrete probability measure with the given atoms.
"""
function discreteLaw(points::AbstractVector{<:Real}, masses::AbstractVector{<:Real})
    s = sum(masses)
    atoms = [float(p) => float(m) / s for (p, m) in zip(points, masses)]
    return FreeMeasure(collect(float.(points)), collect(float.(masses)) ./ s, atoms)
end

"""
    empiricalLaw(samples)

The empirical measure of a sample (equal weights), as a
[`FreeMeasure`](@ref). Useful for free convolution of data-driven
distributions.
"""
function empiricalLaw(samples::AbstractVector{<:Real})
    xs = collect(float.(samples))
    return FreeMeasure(xs, fill(1 / length(xs), length(xs)), Pair{Float64,Float64}[])
end

"""
    freeAtoms(mu)

The atoms of `mu` as `location => mass` pairs (only the atoms recorded at
construction or produced by an exact atom rule in a convolution).
"""
freeAtoms(μ::FreeMeasure) = μ.atoms

"""
    freeMean(mu)

Mean of a [`FreeMeasure`](@ref) from its discrete representation.
"""
freeMean(μ::FreeMeasure) = sum(μ.weights .* μ.nodes)

"""
    freeVar(mu)

Variance of a [`FreeMeasure`](@ref) from its discrete representation.
"""
function freeVar(μ::FreeMeasure)
    m = freeMean(μ)
    return sum(μ.weights .* (μ.nodes .- m) .^ 2)
end

"""
    cauchyTransform(mu, z)

The Cauchy (Stieltjes) transform `G_mu(z) = int d mu(t) / (z - t)`,
evaluated exactly for the discrete representation of `mu`.
"""
cauchyTransform(μ::FreeMeasure, z::Number) =
    sum(w / (z - a) for (a, w) in zip(μ.nodes, μ.weights))

_psi(μ::FreeMeasure, z::Number) =
    sum(w * z * a / (1 - z * a) for (a, w) in zip(μ.nodes, μ.weights))

_dpsi(μ::FreeMeasure, z::Number) =
    sum(w * a / (1 - z * a)^2 for (a, w) in zip(μ.nodes, μ.weights))

# Newton inversion of a monotone-on-branch scalar function with derivative.
function _newton_inverse(f, df, target, z0; tol=1e-12, maxiter=60)
    z = z0
    for _ in 1:maxiter
        dz = (f(z) - target) / df(z)
        z -= dz
        abs(dz) <= tol * (1 + abs(z)) && break
    end
    return z
end

"""
    rTransform(mu, w)

The R-transform of `mu` at `w`: `R(w) = K(w) - 1/w` where `K` is the
functional inverse of the Cauchy transform. `R` linearizes free additive
convolution: `R_{mu boxplus nu} = R_mu + R_nu`. Computed by Newton
inversion of `G`; accurate for `w` in a neighbourhood of 0 whose size is
set by the support of `mu`.
"""
function rTransform(μ::FreeMeasure, w::Number)
    w == 0 && return complex(freeMean(μ))
    z0 = 1 / w + freeMean(μ)
    K = _newton_inverse(z -> cauchyTransform(μ, z),
                        z -> -sum(wt / (z - a)^2 for (a, wt) in zip(μ.nodes, μ.weights)),
                        w, z0)
    return K - 1 / w
end

"""
    sTransform(mu, w)

The S-transform of `mu` at `w`: `S(w) = (1 + w)/w * chi(w)` where `chi`
is the functional inverse of `psi(z) = int z t / (1 - z t) d mu(t)`.
`S` linearizes free multiplicative convolution:
`S_{mu boxtimes nu} = S_mu * S_nu`. Computed by Newton inversion of
`psi`; requires `freeMean(mu) != 0`.
"""
function sTransform(μ::FreeMeasure, w::Number)
    m = freeMean(μ)
    m == 0 && throw(ArgumentError("S-transform requires nonzero mean"))
    z0 = w / m
    χ = _newton_inverse(z -> _psi(μ, z), z -> _dpsi(μ, z), w, z0)
    return (1 + w) / w * χ
end

"""
    freeDensity(mu, xs; eps=...)

The (absolutely continuous) density of `mu` at the points `xs`, from
Stieltjes inversion: `rho(x) = -(1/pi) Im G_mu(x + i eps)`. Atoms appear
as Lorentzian bumps of width `eps`; use [`freeAtoms`](@ref) for their
exact locations and masses.
"""
function freeDensity(μ::FreeMeasure, xs::AbstractVector{<:Real};
                     eps::Real=0.01)
    return [max(-imag(cauchyTransform(μ, x + im * eps)) / π, 0.0) for x in xs]
end

function _freeadd_atoms(μ::FreeMeasure, ν::FreeMeasure)
    out = Pair{Float64,Float64}[]
    for (a, ma) in μ.atoms, (b, mb) in ν.atoms
        ma + mb > 1 && push!(out, (a + b) => (ma + mb - 1))
    end
    return out
end

function _freemul_atoms(μ::FreeMeasure, ν::FreeMeasure)
    out = Pair{Float64,Float64}[]
    m0μ = sum(m for (a, m) in μ.atoms if a == 0.0; init=0.0)
    m0ν = sum(m for (a, m) in ν.atoms if a == 0.0; init=0.0)
    m0 = max(m0μ, m0ν)
    m0 > 0 && push!(out, 0.0 => m0)
    for (a, ma) in μ.atoms, (b, mb) in ν.atoms
        (a == 0.0 || b == 0.0) && continue
        ma + mb > 1 && push!(out, (a * b) => (ma + mb - 1))
    end
    return out
end

# Density of mu boxplus nu (or boxtimes) on a grid, given the Cauchy
# transform of the result evaluated pointwise; strips the Lorentzian
# bumps of the exactly-known atoms and renormalizes the a.c. mass.
function _assemble_result(xs, dens_raw, atoms, eps)
    dens = copy(dens_raw)
    for (c, m) in atoms
        dens .-= (m / π) .* eps ./ ((xs .- c) .^ 2 .+ eps^2)
    end
    dens = max.(dens, 0.0)
    mass_ac = sum((dens[i] + dens[i+1]) * (xs[i+1] - xs[i]) / 2
                  for i in 1:length(xs)-1)
    target = 1 - sum(m for (_, m) in atoms; init=0.0)
    if mass_ac > 0 && target > 0
        dens .*= target / mass_ac
    end
    return _density_measure(xs, dens, atoms)
end

"""
    freeAdd(mu, nu; npoints=1201, eps=nothing)

The free additive convolution `mu boxplus nu`. The density is computed by
the subordination fixed point on a grid covering the Minkowski sum of the
supports; atoms follow the exact Bercovici-Voiculescu rule. `eps` is the
Stieltjes regularization (default: the grid spacing).
"""
function freeAdd(μ::FreeMeasure, ν::FreeMeasure; npoints::Integer=1201, eps=nothing)
    lo = minimum(μ.nodes) + minimum(ν.nodes)
    hi = maximum(μ.nodes) + maximum(ν.nodes)
    pad = 0.05 * (hi - lo) + 0.01
    xs = collect(range(lo - pad, hi + pad; length=npoints))
    ε = eps === nothing ? (xs[2] - xs[1]) : eps
    dens = zeros(length(xs))
    ω1 = ω2 = complex(0.0, ε)
    for i in length(xs):-1:1
        z = xs[i] + im * ε
        if i == length(xs)
            ω1 = ω2 = z
        end
        G, ω1, ω2 = _subordination_G(μ, ν, z, ω1, ω2)
        dens[i] = max(-imag(G) / π, 0.0)
    end
    atoms = _freeadd_atoms(μ, ν)
    return _assemble_result(xs, dens, atoms, ε)
end

function _subordination_G(μ, ν, z, ω1, ω2)
    G1 = G2 = zero(ComplexF64)
    for _ in 1:2000
        G1 = cauchyTransform(μ, ω1)
        G2 = cauchyTransform(ν, ω2)
        nω1 = z - (ω2 - 1 / G2)
        nω2 = z - (ω1 - 1 / G1)
        δ = max(abs(nω1 - ω1), abs(nω2 - ω2), abs(G1 - G2))
        ω1, ω2 = nω1, nω2
        δ < 1e-11 && break
    end
    return (G1 + G2) / 2, ω1, ω2
end

"""
    freeMul(mu, nu; npoints=1001, eps=nothing)

The free multiplicative convolution `mu boxtimes nu` for measures on
`[0, infinity)`, via the S-transform product rule with nested Newton
inversion. Atoms: at 0 the mass is `max(mu({0}), nu({0}))`; elsewhere
the sum-minus-one rule at products of atoms. The output grid is a
Chebyshev grid, clustered at the support ends, and `eps` (the Stieltjes
regularization) defaults to half the mean grid spacing.

Accuracy, measured against the exact moments of
`MP_gamma boxtimes MP_delta` (from the S-transform) at the default
resolution: when both factors are bounded away from 0 the first three
moments err by at most about 2.5%. When a factor's support touches 0
the product density diverges at 0, the `eps`-smoothing leaks mass
below 0 at a rate `~ eps^(1/3)`, and the moment errors grow to about
5% (`MP_1 boxtimes MP_0.5`) and 11% (`MP_1 boxtimes MP_1`). Passing a
smaller `eps` (down to about `1e-4` times the support scale) improves
the cases bounded away from 0 to ~0.05% and `MP_1 boxtimes MP_0.5` to
~0.6%, but for measures with an atom at 0, very small `eps` makes the
Newton continuation fail at many points near the lower edge (and the
run slows down tenfold) — keep the default there.
"""
function freeMul(μ::FreeMeasure, ν::FreeMeasure; npoints::Integer=1001, eps=nothing)
    (minimum(μ.nodes) >= 0 && minimum(ν.nodes) >= 0) ||
        throw(ArgumentError("freeMul requires measures supported on [0, Inf)"))
    if any(a == 0.0 && m == 1.0 for (a, m) in μ.atoms) ||
       any(a == 0.0 && m == 1.0 for (a, m) in ν.atoms)
        return diracLaw(0.0)
    end
    # A Dirac factor dilates: delta_a boxtimes nu is nu with nodes scaled by a.
    if length(μ.nodes) == 1 && μ.weights[1] == 1.0
        return _dilate(ν, μ.nodes[1])
    end
    if length(ν.nodes) == 1 && ν.weights[1] == 1.0
        return _dilate(μ, ν.nodes[1])
    end
    m1 = freeMean(μ) * freeMean(ν)
    hi = maximum(μ.nodes) * maximum(ν.nodes)
    loμ = minimum(μ.nodes); loν = minimum(ν.nodes)
    lo = (loμ > 0 && loν > 0) ? loμ * loν : 0.0
    pad = 0.05 * hi + 0.01
    # Chebyshev grid on [lo, hi + pad], clustered at the support ends,
    # and eps defaulting to half the mean grid spacing. (The accuracy
    # limits of this scheme are documented in the docstring: the
    # dominant error is the eps-smoothing of the density, which for a
    # factor touching 0 leaks mass below 0 at a rate ~ eps^(1/3).)
    mid, half = (lo + hi + pad) / 2, (hi + pad - lo) / 2
    xs = mid .- half .* cos.(range(0, π; length=npoints))
    ε = eps === nothing ? (xs[end] - xs[1]) / (2 * (npoints - 1)) : eps
    dens = zeros(length(xs))
    # Sweep from the right edge with continuation: the state holds the
    # outer variable w = psi_{mu boxtimes nu}(1/zeta) and the s = 1/chi
    # values of the two inner inverses.
    state = _MulState(0.0 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im)
    for i in length(xs):-1:1
        ζ = xs[i] + im * ε
        target = 1 / ζ
        if i == length(xs)
            disc = sqrt(complex(1 + 4 * m1 * target))
            state.w = (-1 + disc) / 2
            state.sμ = freeMean(μ) / state.w
            state.sν = freeMean(ν) / state.w
        end
        G, ok = _mul_G(μ, ν, target, state)
        d = -imag(G) / π
        if !ok || !isfinite(d)
            # Continuation failed at this point (non-convergence is
            # common near the lower edge when eps is very small): reset
            # the state from the closed-form guess and retry once.
            disc = sqrt(complex(1 + 4 * m1 * target))
            state.w = (-1 + disc) / 2
            state.sμ = freeMean(μ) / state.w
            state.sν = freeMean(ν) / state.w
            G, ok = _mul_G(μ, ν, target, state)
            d = -imag(G) / π
        end
        if !isfinite(d)
            # Still failed: fall back to the neighbouring value and
            # reset the continuation state.
            d = i < length(xs) ? dens[i+1] : 0.0
            disc = sqrt(complex(1 + 4 * m1 * target))
            state.w = (-1 + disc) / 2
            state.sμ = freeMean(μ) / state.w
            state.sν = freeMean(ν) / state.w
        end
        dens[i] = max(d, 0.0)
    end
    atoms = _freemul_atoms(μ, ν)
    return _assemble_result(xs, dens, atoms, ε)
end

mutable struct _MulState
    w::ComplexF64
    sμ::ComplexF64
    sν::ComplexF64
end

# chi_mu(w) = psi_mu^{<-1>}(w), computed through s = 1/chi: since
# psi_mu(z) = s G_mu(s) - 1 for s = 1/z, chi is obtained by solving
# s G_mu(s) = 1 + w in the upper half plane, where G is well conditioned
# (no real-axis poles). Also returns d chi/dw.
function _chi_s(μ::FreeMeasure, w::Number, s0::Number)
    s = s0
    G = zero(ComplexF64); dG = zero(ComplexF64)
    for _ in 1:60
        G = cauchyTransform(μ, s)
        dG = -sum(wt / (s - a)^2 for (a, wt) in zip(μ.nodes, μ.weights))
        h = s * G - (1 + w)
        abs(h) < 1e-13 * (1 + abs(w)) && break
        Δ = h / (G + s * dG)
        s -= Δ
        abs(Δ) < 1e-14 * (1 + abs(s)) && break
    end
    G = cauchyTransform(μ, s)
    dG = -sum(wt / (s - a)^2 for (a, wt) in zip(μ.nodes, μ.weights))
    χ = 1 / s
    dχ = -χ^2 / (G + s * dG)
    return χ, s, dχ
end

function _dilate(μ::FreeMeasure, a::Real)
    nodes = μ.nodes .* a
    atoms = [x * a => m for (x, m) in μ.atoms]
    return FreeMeasure(nodes, copy(μ.weights), atoms)
end

# Returns (G, converged): converged is false when the Newton iteration
# hit its iteration cap without meeting either tolerance, in which case
# G is only the last iterate and the caller should retry or fall back.
function _mul_G(μ, ν, target, st::_MulState)
    w = st.w
    converged = false
    for _ in 1:40
        A, sμ, dA = _chi_s(μ, w, st.sμ)
        B, sν, dB = _chi_s(ν, w, st.sν)
        st.sμ, st.sν = sμ, sν
        χprod = (1 + w) / w * A * B
        H = χprod - target
        abs(H) < 1e-11 * (1 + abs(target)) && (converged = true; break)
        dχprod = χprod * (1 / (1 + w) - 1 / w + dA / A + dB / B)
        Δ = H / dχprod
        cap = 0.25 * (1 + abs(w))
        abs(Δ) > cap && (Δ *= cap / abs(Δ))
        w -= Δ
        abs(Δ) < 1e-13 * (1 + abs(w)) && (converged = true; break)
    end
    st.w = w
    return (1 + w) * target, converged
end
