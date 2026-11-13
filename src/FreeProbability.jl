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
    freeCompress, silversteinLaw, FreeBessel, freeBesselMoments,
    freeBesselSupport, freeDensity, freeAtoms, freeMean, freeVar

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

function _freecompress_atoms(μ::FreeMeasure, τ::Real)
    out = Pair{Float64,Float64}[]
    for (a, m) in μ.atoms
        m > 1 - τ && push!(out, a => (m - (1 - τ)) / τ)
    end
    return out
end

# Cauchy transform of the compression mu_tau at z, via the implicit
# equation derived in freeCompress's docstring: G_{mu_tau}(z) =
# G_mu(s)/tau with s - (1 - tau)/G_mu(s) = z, solved by Newton
# iteration on the exact rational G_mu. `s0` is the continuation state
# carried along the grid sweep. Returns (G, s); on non-convergence the
# state is reset once to the closed-form guess s = z/tau and retried.
function _compress_G(μ, z, τ, s0)
    s = s0
    for attempt in 1:2
        ok = false
        for _ in 1:60
            G = cauchyTransform(μ, s)
            dG = -sum(wt / (s - a)^2 for (a, wt) in zip(μ.nodes, μ.weights))
            H = s - (1 - τ) / G - z
            if abs(H) < 1e-12 * (1 + abs(z))
                ok = true
                break
            end
            Δ = H / (1 + (1 - τ) * dG / G^2)
            s -= Δ
            if abs(Δ) < 1e-13 * (1 + abs(s))
                # Confirm the residual, not just the step size.
                G = cauchyTransform(μ, s)
                ok = abs(s - (1 - τ) / G - z) < 1e-10 * (1 + abs(z))
                break
            end
        end
        ok && return cauchyTransform(μ, s) / τ, s
        attempt == 1 && (s = z / τ)
    end
    return cauchyTransform(μ, s) / τ, s
end

"""
    freeCompress(mu, tau; npoints=1201, eps=nothing)

The free compression of `mu` by a factor `tau` in `(0, 1]`: the law of
`p a p` in the compressed space `(p A p, phi/tau)`, where `a` has law
`mu` and `p` is a projection of trace `tau` free from `a`. This is the
compression of Nica & Speicher, *Lectures on the Combinatorics of Free
Probability*, Cambridge University Press, 2006. Note the normalization:
the trace on the compressed space is renormalized by `1/tau`, so the
mean is preserved. (The law of `p a p` under the *original* trace is
the different measure `(1 - tau) delta_0 + tau * mu_tau`, and some
authors parametrize the same family by `1/tau >= 1`, as a
decompression.)

Characterizing identities: the R-transform scales as
`R_{mu_tau}(w) = R_mu(tau * w)`, i.e. the free cumulants scale as
`kappa_n -> tau^(n-1) kappa_n`; an atom of `mu` at `a` of mass `m`
survives exactly when `m > 1 - tau`, with compressed mass
`(m - (1 - tau)) / tau`. Consequences: a semicircle of variance
`sigma^2` compresses to a semicircle of variance `tau * sigma^2`, and
the free Poisson (Marcenko-Pastur) law `MP_gamma` compresses to
`MP_{tau * gamma}` — its atom at 0 (mass `1 - 1/gamma` for
`gamma > 1`) disappears exactly when `tau * gamma <= 1`. For a sample
covariance matrix this is the limiting spectral law of a principal
submatrix retaining a `tau` fraction of the coordinates.

Computation: the Cauchy transform of the result is evaluated through
the implicit equation `G_{mu_tau}(z) = G_mu(s) / tau`, where `s` solves
`s - (1 - tau) / G_mu(s) = z` (Newton iteration on the exact rational
`G_mu`, with continuation along the grid), the density follows by
Stieltjes inversion, and the atoms are inserted exactly. `eps` is the
Stieltjes regularization (default: the grid spacing). `tau == 1`
returns `mu` unchanged.
"""
function freeCompress(μ::FreeMeasure, τ::Real; npoints::Integer=1201, eps=nothing)
    0 < τ <= 1 ||
        throw(ArgumentError("freeCompress requires 0 < tau <= 1, got tau = $τ"))
    τ == 1 && return FreeMeasure(copy(μ.nodes), copy(μ.weights), copy(μ.atoms))
    # A scalar compresses to itself: p (a I) p = a p has law delta_a in
    # the compressed space.
    if length(μ.atoms) == 1 && μ.atoms[1][2] == 1.0
        return diracLaw(μ.atoms[1][1])
    end
    lo = minimum(μ.nodes)
    hi = maximum(μ.nodes)
    pad = 0.05 * (hi - lo) + 0.01
    xs = collect(range(lo - pad, hi + pad; length=npoints))
    ε = eps === nothing ? (xs[2] - xs[1]) : eps
    dens = zeros(length(xs))
    s = complex(0.0, ε)
    for i in length(xs):-1:1
        z = xs[i] + im * ε
        if i == length(xs)
            s = z / τ
        end
        G, s = _compress_G(μ, z, τ, s)
        dens[i] = max(-imag(G) / π, 0.0)
    end
    atoms = _freecompress_atoms(μ, τ)
    return _assemble_result(xs, dens, atoms, ε)
end

# --- Silverstein equation: general-population limiting spectral law ---
#
# For the sample covariance (1/n) T^{1/2} X X' T^{1/2} with population
# spectral distribution H and aspect gamma = p/n, Silverstein's
# equation (Silverstein & Choi 1995, Eq. (1.3); Bai & Silverstein 2010,
# Ch. 4) characterizes the limiting law F through its companion
# Stieltjes transform munderline(z) (the transform of the n x n Gram
# matrix's limiting law): munderline is the unique solution with
# Im munderline(z) > 0 for Im z > 0 of
#     munderline = -(z - gamma * int t / (1 + t munderline) dH(t))^{-1}.
# The transform of F itself is recovered from the exact relation
#     munderline(z) = -(1 - gamma)/z + gamma * m_F(z),
# and the density is rho(x) = Im m_F(x + i0) / pi. (Calibration note:
# at H = delta_1 the unconverted Im munderline / pi is exactly gamma
# times the Marcenko-Pastur density -- the companion check that pins
# this convention.) Support edges are the critical values of the
# inverse map z(m) = gamma * int t/(1 + t m) dH(t) - 1/m, at the real
# m where dz/dm = 1/m^2 - gamma * int t^2/(1 + t m)^2 dH(t) vanishes.

_sil_I(m, t, w) = sum(wi * ti / (1 + ti * m) for (ti, wi) in zip(t, w))
_sil_dI(m, t, w) = -sum(wi * ti^2 / (1 + ti * m)^2 for (ti, wi) in zip(t, w))

# Newton solve of the Silverstein equation for munderline at z (upper
# half plane). Returns (munderline, converged); on non-convergence the
# caller retries cold or falls back, as in _mul_G.
function _silverstein_m(t, w, γ, z, m0)
    m = m0
    for _ in 1:100
        u = z - γ * _sil_I(m, t, w)
        F = m + 1 / u
        abs(F) < 1e-13 * (1 + abs(m)) && return m, true
        Δ = F / (1 + γ * _sil_dI(m, t, w) / u^2)
        m -= Δ
        if abs(Δ) < 1e-14 * (1 + abs(m))
            return m, abs(m + 1 / (z - γ * _sil_I(m, t, w))) < 1e-10 * (1 + abs(m))
        end
    end
    return m, false
end

# Critical values of the inverse map z(m): scan every interval between
# consecutive poles m = -1/t_i (and (0, Inf)) for sign changes of
# dz/dm, refined by bisection. Only nonnegative critical values can be
# support edges of F.
function _silverstein_edges(t, w, γ)
    poles = sort(unique([-1 / ti for ti in t if ti > 0]))
    bounds = vcat(-Inf, poles, 0.0, Inf)
    crit = Float64[]
    N = 250
    for k in 1:length(bounds)-1
        lo_b, hi_b = bounds[k], bounds[k + 1]
        ms = Vector{Float64}(undef, N + 1)
        for j in 0:N
            u = j / N
            ms[j + 1] = if lo_b == -Inf && hi_b == Inf
                tan(π * (u - 0.5))
            elseif lo_b == -Inf
                hi_b - tan(π / 2 * (1 - u))
            elseif hi_b == Inf
                lo_b + tan(π / 2 * u)
            else
                lo_b + (hi_b - lo_b) * u
            end
        end
        prev_m = ms[1]
        prev_d = 1 / prev_m^2 + γ * _sil_dI(prev_m, t, w)
        for j in 2:N + 1
            m = ms[j]
            d = 1 / m^2 + γ * _sil_dI(m, t, w)
            if isfinite(d) && isfinite(prev_d) && prev_d != 0 && sign(d) != sign(prev_d)
                a, b, da = prev_m, m, prev_d
                for _ in 1:80
                    mid = (a + b) / 2
                    sign(1 / mid^2 + γ * _sil_dI(mid, t, w)) == sign(da) ? (a = mid) : (b = mid)
                end
                mc = (a + b) / 2
                push!(crit, γ * _sil_I(mc, t, w) - 1 / mc)
            end
            prev_m, prev_d = m, d
        end
    end
    return sort(filter(z -> z >= -1e-8, crit))
end

function _silverstein_check(γ)
    (isfinite(γ) && γ > 0) ||
        throw(ArgumentError("silversteinLaw requires a finite gamma > 0, got gamma = $γ"))
    return nothing
end

function _silverstein_discrete(t, w, γ, npoints, eps)
    _silverstein_check(γ)
    isempty(t) && throw(ArgumentError("silversteinLaw: empty population spectrum"))
    all(isfinite, t) || throw(ArgumentError("silversteinLaw: population eigenvalues must be finite"))
    all(x -> x >= 0, t) ||
        throw(ArgumentError("silversteinLaw: population eigenvalues must be >= 0 (T is positive semidefinite)"))
    all(isfinite, w) || throw(ArgumentError("silversteinLaw: masses must be finite"))
    all(x -> x >= 0, w) || throw(ArgumentError("silversteinLaw: masses must be >= 0"))
    s = sum(w)
    s > 0 || throw(ArgumentError("silversteinLaw: masses must have positive total"))
    w = collect(float.(w)) ./ s
    t = collect(float.(t))
    # T = 0: the sample covariance is 0 and F = delta_0.
    all(x -> x == 0, t) && return diracLaw(0.0)
    atom0 = max(sum(wi for (ti, wi) in zip(t, w) if ti == 0; init=0.0), 1 - 1 / γ)
    atoms = atom0 > 0 ? [0.0 => atom0] : Pair{Float64,Float64}[]
    ed = _silverstein_edges(t, w, γ)
    hi_bound = (1 + sqrt(γ))^2 * maximum(t)
    hi = max(isempty(ed) ? 0.0 : ed[end], hi_bound)
    pad = 0.02 * hi + 0.01
    # Chebyshev grid on [0, hi + pad], clustered at both ends: when a
    # support edge sits at 0 (e.g. H = delta_1 at gamma = 1) the
    # density diverges there like x^(-1/2), and a uniform grid leaves
    # a ~2% bias in the mean (the same fix freePoissonLaw and freeMul
    # already use for the same singularity).
    mid, half = (hi + pad) / 2, (hi + pad) / 2
    xs = mid .- half .* cos.(range(0, π; length=npoints))
    # Default eps: a fifth of the mean grid spacing. The eps smoothing
    # of an edge singularity is the dominant moment error (at
    # H = delta_1, gamma = 1 it takes the mean bias from 1.5% at half
    # the mean spacing to 0.9%); the Newton continuation stays
    # converged at this level for every calibrated case, atoms
    # included.
    ε = eps === nothing ? (xs[end] - xs[1]) / (5 * (npoints - 1)) : eps
    dens = zeros(length(xs))
    mstate = complex(0.0, 0.0)
    for i in length(xs):-1:1
        z = xs[i] + im * ε
        if i == length(xs)
            mstate = -1 / z
        end
        mu, ok = _silverstein_m(t, w, γ, z, mstate)
        if !ok
            mu, ok = _silverstein_m(t, w, γ, z, -1 / z)
        end
        mstate = mu
        mF = (mu + (1 - γ) / z) / γ
        d = imag(mF) / π
        dens[i] = isfinite(d) ? max(d, 0.0) : (i < length(xs) ? dens[i + 1] : 0.0)
    end
    return _assemble_result(xs, dens, atoms, ε)
end

"""
    silversteinLaw(points, masses, gamma; npoints=2001, eps=nothing)
    silversteinLaw(mu::FreeMeasure, gamma; npoints=2001, eps=nothing)
    silversteinLaw(d::UnivariateDistribution, gamma; hpoints=401, npoints=2001, eps=nothing)

The limiting spectral distribution `F` of the sample covariance
matrix ``(1/n) T^{1/2} X X' T^{1/2}``, where `X` is `p x n` with
i.i.d. entries of mean 0 and variance 1, `T` is a deterministic
positive semidefinite population covariance whose spectral
distribution is `H`, and `gamma = p/n`. Where
[`MarchenkoPastur`](@ref) covers the identity population (`H` a point
mass at 1), this is the law for a general population.

`F` is defined only implicitly, through the Silverstein equation
(Silverstein & Choi, "Analysis of the limiting spectral distribution
of large dimensional random matrices", J. Multivariate Anal. 54(2),
1995, Eq. (1.3); Bai & Silverstein, *Spectral Analysis of Large
Dimensional Random Matrices*, Springer, 2010, Ch. 4): the companion
Stieltjes transform ``\\underline m(z) = \\int (t - z)^{-1}
d\\underline F(t)`` of the Gram matrix's limiting law is the unique
solution in the upper half plane of

```math
\\underline m = -\\Big(z - \\gamma \\int \\frac{t}{1 + t \\underline m}
\\, dH(t)\\Big)^{-1},
\\qquad
\\underline m(z) = -\\frac{1 - \\gamma}{z} + \\gamma \\, m_F(z),
```

and the density of `F` is ``\\rho(x) = \\operatorname{Im} m_F(x +
i0)/\\pi``. The solver evaluates ``\\underline m`` by Newton
iteration with continuation along the grid, recovers `m_F` from the
displayed relation, and reads the density off by Stieltjes
inversion; the grid runs from 0 to the largest support edge, the
edges being the critical values of the inverse map ``z(m) = \\gamma
\\int t/(1 + tm) dH(t) - 1/m``. The grid is a Chebyshev grid,
clustered at the ends (a support edge at 0 carries an
``x^{-1/2}``-diverging density, which a uniform grid misweights, as
in [`freePoissonLaw`](@ref)). `eps` is the Stieltjes regularization
(default: a fifth of the mean grid spacing).

The atom at 0 is exact: its mass is ``\\max(H(\\{0\\}), 1 -
1/\\gamma)`` -- the rank deficiency of the sample covariance,
whichever of `T` and the sampling rectangle causes more of it. The
moments are exact in the limit: ``\\int x dF = \\int t dH`` and
``\\int x^2 dF = \\int t^2 dH + \\gamma (\\int t dH)^2``.

API shape: the result is a [`FreeMeasure`](@ref), not a
`Distributions` law -- like the free convolutions, this law has no
closed-form density, cdf, or sampler, and the grid measure is the
package's working representation for such laws (`freeMean`,
`freeVar`, `freeDensity`, `freeAtoms` and `cauchyTransform` all
apply). `H` is taken primarily as discrete `points`/`masses`, where
the Silverstein integral is an exact finite sum; a `FreeMeasure`
`H` contributes its nodes and weights; a `Distributions` law `H`
is first discretized on an equal-weight quantile grid of `hpoints`
points, an approximation that converges as `hpoints` grows (the law
must be supported on `[0, Inf)`, as `T` is positive semidefinite).

Accuracy, calibrated against the exact anchors before the test bands
were set: at `H = delta_1` the support edges agree with
[`MarchenkoPastur`](@ref) exactly and the density reproduces it to
about `1e-3` in the interior (`7e-3` within a smoothing width of an
edge); at the default resolution the first two moments of the
returned measure match the exact formulas above to within 0.3%
across single-mass, two-mass, and three-mass populations, with and
without an atom at 0 -- the worst case is `H = delta_1` at
`gamma = 1`, whose density diverges at the lower edge 0 (mean
within 0.9%). A `Distributions` population adds its
quantile-discretization error (0.2% on the mean for `Gamma(2, 1.5)`
at the default `hpoints`).

# Examples

```julia
# Population eigenvalues 1 (half) and 4 (half), p/n = 0.1: the
# limiting support splits into two clusters, [0.555, 1.408] and
# [2.485, 6.052]; at p/n = 0.5 it is a single interval.
F = silversteinLaw([1.0, 4.0], [0.5, 0.5], 0.1)
freeMean(F)   # 2.5 = mean of H
freeAtoms(F)  # no atom: gamma < 1 and H has no atom at 0
```
"""
silversteinLaw(points::AbstractVector{<:Real}, masses::AbstractVector{<:Real}, γ::Real;
               npoints::Integer=2001, eps=nothing) =
    _silverstein_discrete(points, masses, γ, npoints, eps)

silversteinLaw(μ::FreeMeasure, γ::Real; npoints::Integer=2001, eps=nothing) =
    _silverstein_discrete(μ.nodes, μ.weights, γ, npoints, eps)

function silversteinLaw(d::UnivariateDistribution, γ::Real;
                        hpoints::Integer=401, npoints::Integer=2001, eps=nothing)
    qs = [(i - 0.5) / hpoints for i in 1:hpoints]
    pts = quantile.(Ref(d), qs)
    return _silverstein_discrete(pts, fill(1.0, hpoints), γ, npoints, eps)
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

# --- Free Bessel laws -------------------------------------------------
#
# The free Bessel law pi_{s,t} of Banica, Belinschi, Capitaine & Collins,
# "Free Bessel laws", Canad. J. Math. 63(1), 2011: for s >= 1 it is
# pi^{boxtimes (s-1)} boxtimes pi^{boxplus t} (free multiplicative and
# additive powers of the free Poisson law pi = MP_1), and for t <= 1 it
# is ((1-t) delta_0 + t delta_1) boxtimes pi^{boxtimes s}; the two
# definitions agree on the overlap (their Theorem 1.3). Its S-transform
# is S(z) = (1+z)^{-(s-1)} / (t+z), and the moment series
# f(z) = sum_k m_k z^k satisfies the algebraic equation (their
# Proposition 4.1)
#     f = 1 + z f^s (f + t - 1),
# from which everything below is computed: the moments by the
# Fuss-Narayana formula (their Theorem 5.2), the support and the atom
# by their Theorems 2.1, 3.1 and 3.2, and the density by solving the
# equation along the real axis (no smoothing parameter).

function _freebessel_check(s::Real, t::Real)
    (isfinite(s) && s > 0) ||
        throw(ArgumentError("FreeBessel requires a finite s > 0, got s = $s"))
    (isfinite(t) && t > 0) ||
        throw(ArgumentError("FreeBessel requires a finite t > 0, got t = $t"))
    (s < 1 && t > 1) &&
        throw(ArgumentError("FreeBessel is not defined on the critical rectangle " *
                            "0 < s < 1, t > 1 (Banica-Belinschi-Capitaine-Collins, " *
                            "Definition 1.4); got s = $s, t = $t"))
    return nothing
end

"""
    freeBesselSupport(s, t)

The support of the free Bessel law `pi_{s,t}` as a named tuple
`(lo, hi, atom)`: the absolutely continuous part lives on `[lo, hi]`
and `atom` is the mass of the atom at 0. For `t < 1` the atom is
`1 - t` and the interval is bounded away from 0 (Banica-Belinschi-
Capitaine-Collins, Theorem 3.1); for `t = 1` the support is
`[0, (s+1)^(s+1)/s^s]` (Theorem 2.1); for `t > 1` there is no atom and
the support is `[0, K_+]` with `K_+` from Theorem 3.2 — except at
`s = 1`, where the law is the free Poisson law of parameter `t` and
the lower edge is `(1 - sqrt(t))^2`.
"""
function freeBesselSupport(s::Real, t::Real)
    _freebessel_check(s, t)
    s, t = float(s), float(t)
    if t == 1.0
        return (lo=0.0, hi=(s + 1)^(s + 1) / s^s, atom=0.0)
    elseif t < 1.0
        disc = sqrt(t^2 * (s - 1)^2 + 4 * s * t)
        wp = (t * s - t + 2 + disc) / (2 * (1 - t))
        wm = (t * s - t + 2 - disc) / (2 * (1 - t))
        Φ(w) = t * w * ((1 - w) / (1 - (1 - t) * w))^s
        return (lo=t / Φ(wp), hi=t / Φ(wm), atom=1.0 - t)
    else
        disc = sqrt(t^2 * (s - 1)^2 + 4 * s * t)
        w1 = (t * (s + 1) - disc) / (2 * s * (t - 1))
        Φst = w1 * (1 - w1)^s / (t + (1 - t) * w1)
        lo = s == 1.0 ? (1 - sqrt(t))^2 : 0.0
        return (lo=lo, hi=1.0 / Φst, atom=0.0)
    end
end

"""
    freeBesselMoments(s, t, kmax)

The first `kmax` moments `m_1, ..., m_kmax` of the free Bessel law
`pi_{s,t}`, exactly, by the Fuss-Narayana formula (Banica-Belinschi-
Capitaine-Collins, Theorem 5.2)

```math
m_k = \\sum_{b=1}^{k} \\frac{1}{b} \\binom{k-1}{b-1} \\binom{s k}{b-1} t^b
```

with the generalized binomial coefficient for real `s`. At `t = 1`
these are the Fuss-Catalan numbers `1/(s k + 1) * binomial(s k + k, k)`
— at `s = 2`: 1, 3, 12, 55, 273; at `s = 3`: 1, 4, 22, 140, 969 — and
`m_1 = t` always.
"""
function freeBesselMoments(s::Real, t::Real, kmax::Integer)
    _freebessel_check(s, t)
    kmax >= 0 || throw(ArgumentError("freeBesselMoments requires kmax >= 0, got $kmax"))
    s, t = float(s), float(t)
    out = Vector{Float64}(undef, kmax)
    for k in 1:kmax
        tot = 0.0
        for b in 1:k
            cb = 1.0
            for j in 0:b-2
                cb *= (s * k - j)
            end
            cb /= factorial(b - 1)
            tot += (1.0 / b) * binomial(k - 1, b - 1) * cb * t^b
        end
        out[k] = tot
    end
    return out
end

# Newton solve of z f^{s+1} + (t-1) z f^s - f + 1 = 0 for f (complex).
function _fb_newton_f(z, s, t, f0; iters=60)
    f = f0
    for _ in 1:iters
        fs = f^s
        P = z * fs * (f + t - 1.0) - f + 1.0
        dP = z * fs * ((s + 1.0) + (t - 1.0) * s / f) - 1.0
        step = P / dP
        f -= step
        abs(step) < 1e-13 * max(1.0, abs(f)) && break
    end
    return f
end

# Density at a few points by the eta-ladder: descend from Im xi = 0.3
# to eta_final, carrying the solution along. Used as the fallback for
# grid points where the real-axis continuation fails its residual.
function _fb_density_ladder(s, t, xs; eta_final=5e-5)
    etas = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, eta_final]
    rho = zeros(length(xs))
    f_prev = nothing
    for (i, x) in enumerate(xs)
        f = 1.0 + 0.0im
        for eta in etas
            z = 1.0 / (x + im * eta)
            seed = eta == etas[1] ? (f_prev === nothing ? 1.0 + t * z : f_prev) : f
            f = _fb_newton_f(z, s, t, seed)
        end
        G = f / (x + im * eta_final)
        rho[i] = max(-imag(G) / π, 0.0)
        f_prev = f
    end
    return rho
end

# The density of pi_{s,t} on the grid xs, from the cubic on the real
# axis: seed one interior point by the eta-ladder, then sweep in x in
# both directions with Newton continuation; a point whose converged
# residual is not < 1e-7 falls back to the ladder value. Isolated
# small-x calls can seed the wrong (real) branch — the sweep from an
# interior point is what keeps the branch, which is why the density is
# only ever evaluated as a full grid sweep.
function _fb_density(s, t, xs)
    rho = zeros(length(xs))
    nz = findall(>(0), xs)
    isempty(nz) && return rho
    i0 = nz[div(length(nz), 3) + 1]
    f = nothing
    for eta in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
        z = 1.0 / (xs[i0] + im * eta)
        f = _fb_newton_f(z, s, t, f === nothing ? 1.0 + t * z : f)
    end
    resid(z, f) = abs(z * f^s * (f + t - 1.0) - f + 1.0)
    function sweep(idxs, f)
        for i in idxs
            if xs[i] <= 0
                rho[i] = NaN
                continue
            end
            z = 1.0 / xs[i]
            f2 = _fb_newton_f(z, s, t, f)
            if resid(z, f2) < 1e-7
                f = f2
                rho[i] = max(-imag(f * z) / π, 0.0)
            else
                rho[i] = NaN
            end
        end
        return f
    end
    sweep(i0:length(xs), f)
    sweep((i0 - 1):-1:1, f)
    bad = .!isfinite.(rho)
    if any(bad)
        rho[bad] = _fb_density_ladder(s, t, xs[bad])
    end
    rho[.!isfinite.(rho)] .= 0.0
    return rho
end

"""
    FreeBessel(s, t; npoints=2001)

The free Bessel law `pi_{s,t}` of Banica, Belinschi, Capitaine &
Collins, "Free Bessel laws", Canad. J. Math. 63(1), 2011, pp. 3-37
(arXiv:0710.5931), as a [`FreeMeasure`](@ref) grid law: the density is
computed from the algebraic equation for the moment series
`f = 1 + z f^s (f + t - 1)` (their Proposition 4.1) on a grid over the
support ([`freeBesselSupport`](@ref)), and the atom `(1 - t)_+` at 0
is carried exactly in the atoms list. The grid is clustered at the
lower edge (`u^3` when the support touches 0, where the density
diverges — like `x^{-s/(s+1)}` at `t = 1` and like `x^{-1/s}` for
`t > 1` — and `u^2` when it is bounded away from 0, where the density
vanishes at both ends).

The parametrization is `(s, t)` as in the paper: `s` is the free
multiplicative power (at integer `s` and `t = 1`, `pi_{s,1}` is the
limiting spectral law of a product of `s` independent square Ginibre
matrices, i.e. a product of `s` square Wishart matrices), and `t` is
the mean — the compression trace in the Bernoulli-factor definition.
It is deliberately *not* parametrized by a product depth and a
Marcenko-Pastur ratio `(k, gamma)`: a product of `k >= 2` laws
`MP_gamma` with `gamma != 1` is `pi_{s,t}` for no `(s, t)` (the
S-transform identity would force `(t + z)(1 + gamma z)^k =
(1 + z)^{s-1}`, impossible for `gamma != 1`; the paper's Section 5).
The boundary dictionaries are: `pi_{1,t}` is the dilation by `t` of
`MP_{1/t}` (see [`freePoissonLaw`](@ref)), and `pi_{s,1}` is the
`s`-fold free multiplicative convolution of `MP_1`.

Domain: `(s, t)` in `(0, Inf)^2` minus the critical rectangle
`(0, 1) x (1, Inf)` of the paper's Definition 1.4; the constructor
throws `ArgumentError` there rather than extrapolating the equation
into a region where the law may not exist.

API shape: the result is a `FreeMeasure`, not a `Distributions` law —
the density has no elementary closed form, only the algebraic
equation above, and the grid measure is the package's working
representation for such laws (`freeMean`, `freeVar`, `freeDensity`,
`freeAtoms`, `cauchyTransform` and the free convolutions all apply).
Exact moments are available separately from
[`freeBesselMoments`](@ref), support and atom from
[`freeBesselSupport`](@ref).

Accuracy, calibrated against the paper's anchors before the test
bands were set: `freeBesselMoments` agrees with the independent
recursion from `f = 1 + z f^s (f + t - 1)` to 0 at integer cases and
`< 2e-12` at non-integer ones (`k <= 10`); the density reproduces the
exact moments to a worst relative error of 0.0069% in quadrature
before normalization; and the returned measure's first four moments
match the exact ones to at most about 0.13% at the default resolution
across the calibrated cases `(2, 1)`, `(3, 1)`, `(2, 1/2)`, `(2, 2)`,
`(2, 3)`, `(3/2, 2)` — the worst case is `(3, 1)`, whose
`x^{-3/4}` edge singularity is the hardest to weight. The singular-edge
rates check against the paper's sharp constants: at `s = 2`,
`t > 1`, `pi rho(x) sqrt(x) -> 1/sqrt(t - 1)` (0.707 at `t = 3`,
read off the grid sweep to three digits).

# Examples

```julia
fb = FreeBessel(2, 1)          # product of two square Wisharts' limit
freeMean(fb)                   # 1 = t
freeAtoms(FreeBessel(2, 0.5))  # [0.0 => 0.5]
freeBesselMoments(2, 1, 4)     # [1, 3, 12, 55]  (Fuss-Catalan)
```
"""
function FreeBessel(s::Real, t::Real; npoints::Integer=2001)
    _freebessel_check(s, t)
    sup = freeBesselSupport(s, t)
    lo, hi = sup.lo, sup.hi
    pw = lo == 0.0 ? 3 : 2
    u = range(0, 1; length=npoints)
    xs = lo .+ (hi - lo) .* u .^ pw
    rho = _fb_density(float(s), float(t), xs)
    atoms = sup.atom > 0 ? [0.0 => sup.atom] : Pair{Float64,Float64}[]
    return _density_measure(xs, rho, atoms)
end
