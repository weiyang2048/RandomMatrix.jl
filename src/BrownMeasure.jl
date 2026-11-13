# The Brown measure of an R-diagonal element: the Haagerup-Larsen
# radial law (single-ring law).
#
# For an R-diagonal element a (equivalently a = u p with u a Haar
# unitary free from a positive p), the Brown measure mu_a is the
# two-dimensional limiting eigenvalue distribution. It is rotation
# invariant, hence determined by its radial CDF, and Haagerup &
# Larsen (J. Funct. Anal. 176 (2000), 331-367, Thm. 4.4 and
# Cor. 4.5) compute it from nu, the law of a* a, through its
# S-transform:
#     mu_a(B(0, S_nu(t - 1)^(-1/2))) = t,     t in (nu({0}), 1],
# with S_nu strictly decreasing, S_nu(0) = 1 / int t dnu. The
# support is the annulus ||a^{-1}||_2^{-1} <= |z| <= ||a||_2 (a disk
# when the inner radius vanishes), i.e.
#     outer^2 = int t dnu(t),   inner^2 = 1 / int t^{-1} dnu(t)
# (inner = 0 when nu has an atom at 0 or int t^{-1} dnu = Inf), and
# mu_a has an atom at 0 of mass exactly nu({0}). The planar density
# on the annulus is rho(z) = Phi'(|z|) / (2 pi |z|), Phi the radial
# CDF. Guionnet, Krishnapur & Zeitouni (Ann. of Math. 174(2)
# (2011), 1189-1217) prove that the eigenvalues of the matrix model
# A = U T V (U, V Haar unitary, T deterministic diagonal with
# T^2's spectral law -> nu, ||T|| bounded, no singular-value mass
# escaping to 0) converge to this measure: the single ring theorem.
#
# Computation: parametrize by s in (0, Inf) through psi, the moment
# transform of nu. With N(s) = sum_j w_j s t_j / (1 + s t_j)
# (= -psi(-s), an exact sum for a discrete nu) and u = 1 - N(s),
# the identity chi(u - 1) = -s in the S-transform gives the
# Haagerup-Larsen pair
#     radial CDF Phi(r) = u,   r^2 = N(s) / (s u) = 1 / S_nu(u - 1),
# and differentiating the parametrization gives the planar density
# in closed form,
#     rho = D s^2 (1 - N)^2 / (pi (N (1 - N) - s D)),
#     D(s) = sum_j w_j t_j / (1 + s t_j)^2 = N'(s).
# The sweep solves N(s) = 1 - u by Newton iteration with
# continuation on a cosine-clustered u-grid. Radial moments obey
# the exact identity (change of variables in the parametrization;
# the Brown measure does NOT reproduce the *-moments of a)
#     int |z|^{2k} d mu_a = int_0^1 S_nu(u - 1)^{-k} du.

export BrownMeasure, brownMeasure, radialCDF, brownDensity

"""
    BrownMeasure(inner, outer, atom0, radius, cdf, density)

The Brown measure of an R-diagonal element, as returned by
[`brownMeasure`](@ref): a rotation-invariant probability measure on
the complex plane, stored through its radial law.

- `inner`, `outer`: the support is the annulus
  `inner <= |z| <= outer` (a disk when `inner == 0`); the radii are
  exact, `outer = sqrt(int t dnu)` and
  `inner = sqrt(1 / int t^{-1} dnu)` (0 when `nu` has an atom at 0).
- `atom0`: the atom at `z = 0`, of mass exactly `nu({0})`.
- `radius`, `cdf`: the radial CDF `Phi(r) = mu({|z| <= r})` on an
  increasing grid from `inner` to `outer`, endpoints included;
  `Phi` already accounts for the atom (`Phi(inner) = atom0` up to
  the continuous part starting there).
- `density`: the planar density `rho(z) = Phi'(|z|) / (2 pi |z|)`
  at the grid radii (interior values are the closed-form
  Haagerup–Larsen derivative; the two endpoint values repeat the
  nearest interior value). Empty when the measure is singular (a
  single circle: `nu = delta_c`), where no planar density exists.
"""
struct BrownMeasure
    inner::Float64
    outer::Float64
    atom0::Float64
    radius::Vector{Float64}
    cdf::Vector{Float64}
    density::Vector{Float64}
end

# Solve N(s) = target for s > 0, where
#     N(s) = sum_j w_j s t_j / (1 + s t_j),  N'(s) = D(s) > 0,
# by Newton iteration from s0 (continuation); on failure fall back
# to bisection on a doubling bracket. N increases from 0 to
# 1 - nu({0}) as s goes 0 -> Inf, so a solution exists exactly for
# target in (0, 1 - nu({0})).
function _brown_solve_s(t, w, target, s0)
    s = s0
    for _ in 1:100
        N = sum(wj * s * tj / (1 + s * tj) for (tj, wj) in zip(t, w))
        D = sum(wj * tj / (1 + s * tj)^2 for (tj, wj) in zip(t, w))
        step = (N - target) / D
        s -= step
        if !(s > 0) || !isfinite(s)
            s = (s + step) / 2
        end
        abs(step) <= 1e-14 * (1 + s) && return s
    end
    lo, hi = 0.0, max(s0, 1.0)
    Nof = x -> sum(wj * x * tj / (1 + x * tj) for (tj, wj) in zip(t, w))
    while Nof(hi) < target
        hi *= 2
        hi > 1e300 && break
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        Nof(mid) < target ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

function _brown_discrete(t, w, npoints)
    isempty(t) && throw(ArgumentError("brownMeasure: empty input law"))
    all(isfinite, t) || throw(ArgumentError("brownMeasure: points must be finite"))
    all(x -> x >= 0, t) ||
        throw(ArgumentError("brownMeasure: points must be >= 0 (nu is a law on [0, Inf))"))
    all(isfinite, w) || throw(ArgumentError("brownMeasure: masses must be finite"))
    all(x -> x >= 0, w) || throw(ArgumentError("brownMeasure: masses must be >= 0"))
    sw = sum(w)
    sw > 0 || throw(ArgumentError("brownMeasure: masses must have positive total"))
    t = collect(float.(t))
    w = collect(float.(w)) ./ sw
    p0 = sum(wj for (tj, wj) in zip(t, w) if tj == 0.0; init=0.0)
    # nu = delta_0: a = 0, the Brown measure is delta_0.
    p0 == 1.0 && return BrownMeasure(0.0, 0.0, 1.0, [0.0], [1.0], Float64[])
    m1 = sum(w .* t)
    outer = sqrt(m1)
    pos = unique([tj for tj in t if tj > 0.0])
    # nu = delta_c (c > 0): a = sqrt(c) * (Haar unitary), the Brown
    # measure is the uniform measure on the circle of radius
    # sqrt(c) (Haagerup-Larsen, Prop. 4.6(i)) -- singular, no
    # density, radial CDF a step.
    if p0 == 0.0 && length(pos) == 1
        r = sqrt(pos[1])
        return BrownMeasure(r, r, 0.0, [r], [1.0], Float64[])
    end
    inner = p0 > 0 ? 0.0 :
        sqrt(1 / sum(wj / tj for (tj, wj) in zip(t, w) if tj > 0.0))
    # Cosine-clustered mass grid on the OPEN interval (p0, 1): the
    # endpoints are the exact rows (inner, p0) and (outer, 1)
    # appended below.
    th = range(0, pi; length=npoints + 2)[2:end-1]
    us = p0 .+ (1 - p0) .* (1 .- cos.(th)) ./ 2
    rs = Float64[]; uu = Float64[]; rho = Float64[]
    sizehint!(rs, npoints); sizehint!(uu, npoints); sizehint!(rho, npoints)
    s = (1 - us[end]) / m1  # small-s guess: N(s) ~ m1 s
    for u in reverse(us)
        target = 1 - u
        s = _brown_solve_s(t, w, target, s)
        N = sum(wj * s * tj / (1 + s * tj) for (tj, wj) in zip(t, w))
        D = sum(wj * tj / (1 + s * tj)^2 for (tj, wj) in zip(t, w))
        r = sqrt(target / (s * u))
        den = N * (1 - N) - s * D
        push!(rs, r); push!(uu, u)
        push!(rho, den > 0 ? D * s^2 * (1 - N)^2 / (pi * den) : NaN)
    end
    reverse!(rs); reverse!(uu); reverse!(rho)
    # Guard: replace any non-finite / negative density (possible at
    # the extreme grid points through cancellation in den) by
    # interpolation over the radius grid.
    for i in eachindex(rho)
        if !isfinite(rho[i]) || rho[i] < 0
            j = findnext(x -> isfinite(x) && x >= 0, rho, i + 1)
            k = findprev(x -> isfinite(x) && x >= 0, rho, i - 1)
            rho[i] = if j !== nothing && k !== nothing
                rho[k] + (rho[j] - rho[k]) * (rs[i] - rs[k]) / (rs[j] - rs[k])
            elseif j !== nothing
                rho[j]
            elseif k !== nothing
                rho[k]
            else
                0.0
            end
        end
    end
    radius = vcat(inner, rs, outer)
    cdf = vcat(p0, uu, 1.0)
    density = vcat(rho[1], rho, rho[end])
    return BrownMeasure(inner, outer, p0, radius, cdf, density)
end

"""
    brownMeasure(points, masses; npoints=2001)
    brownMeasure(mu::FreeMeasure; npoints=2001)
    brownMeasure(d::UnivariateDistribution; hpoints=401, npoints=2001)

The Brown measure of an R-diagonal element `a` whose squared
modulus `a* a` has law `nu` — the two-dimensional limiting
eigenvalue distribution of the matrix model `A = U P` with `U`
Haar unitary and `P` positive with spectral law `nu` (more
generally `A = U T V`, the single ring theorem of Guionnet,
Krishnapur & Zeitouni, Ann. of Math. 174(2) (2011), 1189–1217).
This is the package's first non-Hermitian limiting law: the measure
lives on the complex plane, is rotation invariant, and is returned
as a [`BrownMeasure`](@ref) (radii, radial CDF and planar density
on a grid), not as a `FreeMeasure` — a real-line grid law cannot
represent it.

By Haagerup & Larsen (J. Funct. Anal. 176 (2000), 331–367,
Thm. 4.4, Cor. 4.5), with `S_nu` the S-transform of `nu`, the
radial CDF `Phi(r) = mu_a({|z| <= r})` is determined by

```math
\\mu_a\\big(B(0, S_\\nu(t-1)^{-1/2})\\big) = t,
\\qquad t \\in (\\nu(\\{0\\}), 1],
```

the support is the annulus (single ring) with exact radii
`outer = sqrt(int t dnu)` and `inner = sqrt(1 / int t^{-1} dnu)`,
and `mu_a` has an atom at 0 of mass exactly `nu({0})`. Boundary
behaviour, all exact: an atom `nu({0}) = p_0 > 0` gives the atom
at 0 and forces `inner = 0` (the continuous part fills the disk);
`int t^{-1} dnu = Inf` (e.g. `nu` with a density at 0) also gives
`inner = 0`; `nu = delta_c` gives the uniform measure on the
circle of radius `sqrt(c)` (singular: the returned `density` grid
is empty); `nu = delta_0` gives `delta_0`. When `inner = 0` and
`nu` has a density at 0 the planar density stays finite at the
centre (for the circular element it is the constant `1/pi`); for
an atom-carrying `nu` it can diverge at the *outer* edge instead
(Haagerup–Larsen's Bernoulli example below).

The solver parametrizes the Haagerup–Larsen relation by
`s in (0, Inf)` through the moment transform of the discrete
`nu`: with `N(s) = sum_j w_j s t_j / (1 + s t_j)` and
`u = 1 - N(s)`, the pair `(r, Phi(r)) = (sqrt((1-u)/(s u)), u)`
traces the radial CDF exactly, and the planar density follows in
closed form by differentiating the parametrization. The grid is
cosine-clustered in the mass coordinate `u` (`npoints` interior
points); `nu` is taken primarily as discrete `points`/`masses`,
where `N(s)` is an exact finite sum; a `FreeMeasure` contributes
its nodes and weights; a `Distributions` law is first discretized
on an equal-weight quantile grid of `hpoints` points (an
approximation that converges as `hpoints` grows; the law must be
supported on `[0, Inf)`).

Accuracy, calibrated against the primary sources' own worked
examples before the test bands were set (calibration scripts
`hidden_files/brown_proto.jl`, `brown_calib.jl`): the circular
element (`nu` the free Poisson law `MP_1`, Haagerup–Larsen
Ex. 5.1) returns `Phi(r) = r^2` to about `1e-7` and density `1/pi`
to `1e-14` on the bulk of the disk; within `r <~ 0.05` of the
centre its density instead reflects the input grid law's own
discreteness near `t = 0` (the `freePoissonLaw` grid carries a
node at 0 of weight `8e-4`, and the solver is faithful to the
law it is given); the free Poisson family `nu_c` (Ex. 5.2)
returns density exactly `1/pi` on its annulus `[sqrt(c-1), sqrt(c)]` (atom
`1 - c` at 0 and disk when `c < 1`); the Bernoulli law
`nu = (1-a) delta_0 + a delta_1` (Ex. 5.3) returns the density
`(1-a) / (pi (1-r^2)^2)` on the disk of radius `sqrt(a)` to
`1e-4` (worst at the edge, `1e-8` inside); a two-point `nu`
reproduces the exact radii and, through
`int |z|^{2k} dmu_a = int_0^1 S_nu(u-1)^{-k} du`, the S-transform
moment identity to about `1e-4`. Formed-matrix simulation
(`A = U P`, Haar `U` from this package) at `n = 500` matches the
radial CDF with pooled KS distance about `0.03` — the deviation
sits in the finite-`n` edge layers (eigenvalue extremes overshoot
the outer radius by about 2% at that size), not in the bulk.

# Examples

```julia
# Two-point nu: radii sqrt(1.6), sqrt(2.5); the median radius is
# exactly sqrt(2).
bm = brownMeasure([1.0, 4.0], [0.5, 0.5])
(bm.inner, bm.outer)   # (1.2649..., 1.5811...)
radialCDF(bm, sqrt(2.0))   # 0.5

# The circular element: nu = MP_1, the uniform unit disk.
bm = brownMeasure(freePoissonLaw(1.0))
radialCDF(bm, 0.5)     # 0.25 = 0.5^2
brownDensity(bm, 0.5)  # 0.3183... = 1/pi
```
"""
brownMeasure(points::AbstractVector{<:Real}, masses::AbstractVector{<:Real};
             npoints::Integer=2001) =
    _brown_discrete(points, masses, npoints)

brownMeasure(μ::FreeMeasure; npoints::Integer=2001) =
    _brown_discrete(μ.nodes, μ.weights, npoints)

function brownMeasure(d::UnivariateDistribution;
                      hpoints::Integer=401, npoints::Integer=2001)
    qs = [(i - 0.5) / hpoints for i in 1:hpoints]
    pts = quantile.(Ref(d), qs)
    return _brown_discrete(pts, fill(1.0, hpoints), npoints)
end

"""
    radialCDF(bm::BrownMeasure, r)

The radial CDF of the Brown measure `bm` at radius `r`:
`Phi(r) = mu({z : |z| <= r})`, by linear interpolation on the
stored grid. `Phi(r) = atom0` inside the hole (`r < bm.inner`),
`0` for `r < 0`, and `1` for `r >= bm.outer`. For the singular
circle law (`nu = delta_c`) it is the step at `sqrt(c)`.
"""
function radialCDF(bm::BrownMeasure, r::Real)
    r < 0 && return 0.0
    bm.inner == bm.outer && return r < bm.inner ? bm.atom0 : 1.0
    r <= bm.radius[1] && return bm.cdf[1]
    r >= bm.radius[end] && return 1.0
    i = searchsortedlast(bm.radius, r)
    r1, r2 = bm.radius[i], bm.radius[i + 1]
    c1, c2 = bm.cdf[i], bm.cdf[i + 1]
    return c1 + (c2 - c1) * (r - r1) / (r2 - r1)
end

"""
    brownDensity(bm::BrownMeasure, r)
    brownDensity(bm::BrownMeasure, z::Number)

The planar density of the Brown measure `bm` at radius `r` (or at
the complex point `z`, through `|z|`):
`rho = Phi'(r) / (2 pi r)`, by linear interpolation on the stored
grid; `0` outside the support annulus. Returns `NaN` for the
singular laws (a single circle, or `delta_0`), which have no
planar density.
"""
function brownDensity(bm::BrownMeasure, r::Real)
    isempty(bm.density) && return NaN
    (r < bm.inner || r > bm.outer) && return 0.0
    r <= bm.radius[1] && return bm.density[1]
    r >= bm.radius[end] && return bm.density[end]
    i = searchsortedlast(bm.radius, r)
    r1, r2 = bm.radius[i], bm.radius[i + 1]
    d1, d2 = bm.density[i], bm.density[i + 1]
    return d1 + (d2 - d1) * (r - r1) / (r2 - r1)
end

brownDensity(bm::BrownMeasure, z::Number) = brownDensity(bm, abs(z))
