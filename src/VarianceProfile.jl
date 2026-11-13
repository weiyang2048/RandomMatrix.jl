# Limiting spectral law of Wigner matrices with a variance profile,
# via the quadratic vector equation (QVE)
#     -1/m_i(z) = z + sum_j a_ij m_j(z),   Im m_i(z) > 0,
# where a_ij = s_ij / n are the entry variances of the matrix (package
# normalization E|X_ij|^2 = s_ij / n). The limiting density is
# rho(x) = (1/(pi n)) sum_i Im m_i(x + i0) (Ajanki, Erdos & Kruger,
# "Quadratic vector equations on complex upper half-plane",
# Probab. Theory Related Fields 170 (2018); arXiv:1506.05095).
#
# Identical rows of the profile share an m value (the QVE solution is
# unique, and the system is equivariant under permuting coordinates
# with equal rows), so the solve runs on row groups with the reduced
# matrix Ared[a, b] = S[rep_a, rep_b] * count_b / n: block-constant
# profiles solve on the block system exactly, and any profile costs
# per group, not per coordinate. A coordinate whose profile row is
# identically zero decouples exactly (m_i = -1/z) and contributes an
# exact atom at 0 of mass (zero rows)/n.
#
# Solvers: Newton on F(m) = 1./m + z + A m for small group counts;
# Anderson-accelerated fixed-point iteration (depth 5, restart on a
# bad step) for larger ones, with an eta-laddered damped fixed point
# as fallback (the plain iteration's contraction degrades like
# 1 - O(sqrt(eps)) near support edges, which is what makes the
# accelerated path necessary at the default eps).

export profileWignerLaw, randProfileWigner

# Validate a variance profile and return it as a symmetrized
# Matrix{Float64}. `who` prefixes the error messages.
function _profile_matrix(S::AbstractMatrix{<:Real}, who::AbstractString)
    size(S, 1) == size(S, 2) ||
        throw(ArgumentError("$who: profile must be square, got size $(size(S))"))
    n = size(S, 1)
    n >= 1 || throw(ArgumentError("$who: empty profile"))
    Sf = Matrix{Float64}(S)
    all(isfinite, Sf) ||
        throw(ArgumentError("$who: profile entries must be finite"))
    all(x -> x >= 0, Sf) ||
        throw(ArgumentError("$who: profile entries must be >= 0 (they are variances)"))
    scale = max(1.0, maximum(Sf))
    maximum(abs.(Sf .- Sf')) <= 1e-10 * scale ||
        throw(ArgumentError("$who: profile must be symmetric"))
    return (Sf .+ Sf') ./ 2
end

# Group identical rows of S: returns (representative row indices,
# group counts, reduced matrix Ared[a, b] = S[rep_a, rep_b] * cnt_b / n).
function _qve_groups(S)
    n = size(S, 1)
    seen = Dict{Vector{Float64},Int}()
    reps = Int[]
    counts = Int[]
    for i in 1:n
        row = S[i, :]
        g = get(seen, row, 0)
        if g == 0
            push!(reps, i)
            push!(counts, 1)
            seen[row] = length(reps)
        else
            counts[g] += 1
        end
    end
    G = length(reps)
    Ared = [S[reps[a], reps[b]] * counts[b] / n for a in 1:G, b in 1:G]
    return reps, counts, Ared
end

_qve_F(A, z, m) = 1.0 ./ m .+ z .+ A * m

# Newton iteration on F(m) = 0, with step damping that keeps every
# component in the upper half plane. Returns (m, residual, iters, ok).
function _qve_newton(A, z, m0; tol=1e-12, maxiter=60)
    m = copy(m0)
    F = _qve_F(A, z, m)
    res = maximum(abs.(F))
    it = 0
    while res > tol && it < maxiter
        it += 1
        Δ = try
            (A - Diagonal(1.0 ./ m .^ 2)) \ F
        catch
            return m, Inf, it, false
        end
        ω = 1.0
        mn = m .- Δ
        while any(x -> imag(x) <= 0, mn) && ω > 1e-4
            ω /= 2
            mn = m .- ω .* Δ
        end
        any(x -> imag(x) <= 0, mn) && return m, Inf, it, false
        m = mn
        F = _qve_F(A, z, m)
        res = maximum(abs.(F))
    end
    return m, res, it, res <= tol
end

# Plain (then damped) fixed-point iteration m <- -1 ./ (z + A m).
function _qve_fp(A, z, m0; tol=1e-11, maxiter=400)
    m = copy(m0)
    it = 0
    step = Inf
    while step > tol && it < maxiter
        it += 1
        mn = -1.0 ./ (z .+ A * m)
        if it > 60
            mn = m .+ 0.5 .* (mn .- m)
        end
        step = maximum(abs.(mn .- m))
        m = mn
    end
    res = maximum(abs.(_qve_F(A, z, m)))
    return m, res, it, step <= tol
end

# Anderson-accelerated fixed point (depth 5; history cleared and a
# plain step taken whenever an accelerated step more than quadruples
# the fixed-point residual). Returns (m, residual, iters, ok) with
# the residual measured on F.
function _qve_anderson(A, z, m0; tol=1e-10, maxiter=600, depth=5)
    f(m) = -1.0 ./ (z .+ A * m)
    m = copy(m0)
    fm = f(m)
    g = fm .- m
    res = maximum(abs.(g))
    it = 0
    DFs = Vector{ComplexF64}[]
    DGs = Vector{ComplexF64}[]
    while res > tol && it < maxiter
        it += 1
        mn = fm
        if !isempty(DFs)
            γ = try
                reduce(hcat, DGs) \ g
            catch
                nothing
            end
            if γ !== nothing && all(isfinite, γ)
                cand = fm .- reduce(hcat, DFs) * γ
                all(x -> imag(x) > 0, cand) && (mn = cand)
            end
        end
        fn = f(mn)
        gn = fn .- mn
        resn = maximum(abs.(gn))
        if resn > 4 * res && mn !== fm
            empty!(DFs)
            empty!(DGs)
            mn = fm
            fn = f(mn)
            gn = fn .- mn
            resn = maximum(abs.(gn))
        end
        push!(DFs, fn .- fm)
        push!(DGs, gn .- g)
        if length(DFs) > depth
            popfirst!(DFs)
            popfirst!(DGs)
        end
        m, fm, g, res = mn, fn, gn, resn
    end
    Fres = maximum(abs.(_qve_F(A, z, m)))
    return m, Fres, it, Fres <= 1e-8
end

# Fallback: solve at a ladder of heights eta descending geometrically
# to eps, warm-starting each rung from the previous one.
function _qve_ladder(A, x, eps, m0; B=1.0)
    etatop = max(eps, 0.02 * (1 + B))
    m = copy(m0)
    totit = 0
    η = etatop
    while true
        mm, res, it, ok = _qve_fp(A, x + im * η, m)
        totit += it
        m = mm
        η <= eps * 1.0001 && break
        η = max(eps, η / 4)
    end
    res = maximum(abs.(_qve_F(A, x + im * eps, m)))
    return m, res, totit
end

# Solve the grouped QVE at z = x + i eps with warm start m0.
# Returns (m, residual) with residual = max |F(m)|.
function _qve_solve(A, x, eps, m0; B=1.0)
    z = x + im * eps
    if size(A, 1) <= 16
        m, res, it, ok = _qve_newton(A, z, m0)
        ok && return m, res
        m0 = all(x -> imag(x) > 0, m) ? m : m0
    end
    m, res, it, ok = _qve_anderson(A, z, m0)
    ok && return m, res
    m, res, itl = _qve_ladder(A, x, eps, m; B=B)
    return m, res
end

"""
    profileWignerLaw(S; npoints=2001, eps=nothing)
    profileWignerLaw(sizes, B; npoints=2001, eps=nothing)

The limiting spectral distribution of a Wigner matrix with variance
profile `S`: an `n x n` Hermitian matrix whose entries are
independent (up to the symmetry) with

```math
\\mathbb{E}\\,|X_{ij}|^2 = S_{ij} / n,
```

in the package's Wigner normalization (the constant profile
`S = sigma^2` ones is `sigma * rand(GOE(n), norm=true)`, whose law is
[`semicircleLaw`](@ref)`(sigma)`). The limit is characterized by the
quadratic vector equation (Ajanki, Erdos & Kruger, "Quadratic vector
equations on complex upper half-plane", Probab. Theory Related
Fields 170 (2018), 683-747; arXiv:1506.05095): the vector Stieltjes
transform `m(z)` is the unique solution in the upper half plane of

```math
-\\frac{1}{m_i(z)} = z + \\frac{1}{n} \\sum_j S_{ij}\\, m_j(z),
\\qquad
\\rho(x) = \\frac{1}{\\pi n} \\sum_i \\operatorname{Im} m_i(x + i0),
```

and this function evaluates that density by Stieltjes inversion on a
Chebyshev grid and returns it as a [`FreeMeasure`](@ref) — like
[`silversteinLaw`](@ref), this law has no closed-form density, cdf,
or sampler, and the grid measure is the package's working
representation for such laws (`freeMean`, `freeVar`, `freeDensity`,
`freeAtoms` and `cauchyTransform` all apply). The grid covers
`[-B, B]` (plus a 4% pad), where `B = 2 sqrt(max_i (1/n) sum_j
S_ij)` bounds the support. `eps` is the Stieltjes regularization
(default: a twentieth of the mean grid spacing). The support, the
density, and the edges need not be symmetric-interval or
semicircle-like: block profiles produce multi-cluster laws and
cusps, all read off the same solve.

Two exact reductions anchor the solver. The constant profile
collapses to the scalar equation `sigma^2 m^2 + z m + 1 = 0`, i.e.
`semicircleLaw(sigma)`. A block-diagonal profile is the exact
mixture of the block semicircles — with the full-size normalization
above, a diagonal block of variance `b` covering a fraction `p` of
the coordinates contributes a semicircle of radius `2 sqrt(b p)`
with weight `p`. Coordinates whose profile row is identically zero
are exact zero eigenvalues: they are removed from the solve and
returned as an atom at 0 of mass `(zero rows)/n` (see
[`freeAtoms`](@ref)); the all-zero profile is `diracLaw(0)`.

The block form `profileWignerLaw(sizes, B)` builds the
block-constant profile whose `(a, b)` block (of `sizes[a] x
sizes[b]` entries) is constantly `B[a, b]` and delegates to the
matrix form. Identical rows of the profile share an `m` value by
uniqueness of the QVE solution, so the solve runs on row groups:
block profiles solve on the `K x K` block system exactly, at any
block sizes, while a fully inhomogeneous profile of size `n` solves
on its `n` groups (a few seconds at `n = 400` at the default
resolution; Newton for at most 16 groups, Anderson-accelerated
fixed-point iteration above that, an `eta`-laddered fixed point as
fallback).

The exact moments are `int x dmu = 0` (the law is symmetric: odd
moments of a centered Wigner matrix vanish for every profile) and
`int x^2 dmu = (1/n^2) sum_ij S_ij`, the mean profile entry.

Accuracy, calibrated against the exact anchors and formed-matrix
simulation before the test bands were set: the constant profile
reproduces the semicircle density to `1.7e-5` (sup over the
interior) at the default resolution, its second moment to `2.5e-5`
(`1.6e-6` and `2.8e-6` at `eps = 1e-5`); the block-diagonal mixture
matches its exact density to `3.3e-4` and its moments to `1e-4`;
solver residuals are at most `1e-12` (Newton, block/constant) and
`2e-10` (Anderson, inhomogeneous). Against simulation of a
two-block profile (`B = [2 0.5; 0.5 1]`, equal blocks) the law's
CDF matches the pooled empirical CDF to `0.001` at `n = 400`,
improving with `n`, and the predicted support edge `2.156` is
approached by the simulated largest eigenvalue from below at the
shallow-edge Tracy-Widom scale. [`randProfileWigner`](@ref) samples
the ensemble itself.

# Examples

```julia
# Two communities with within-variance 2 and 1, cross-variance 0.5:
# a single-cluster law of variance (2 + 1 + 2*0.5)/4 = 1.
mu = profileWignerLaw([100, 100], [2.0 0.5; 0.5 1.0])
freeMean(mu)   # 0
freeVar(mu)    # 1.0001 (exact: 1)

# A band profile: variance 1 within distance 20 of the diagonal.
n = 200
S = [abs(i - j) <= 20 ? 1.0 : 0.0 for i in 1:n, j in 1:n]
freeVar(profileWignerLaw(S))   # 0.19451 (exact value: mean(S) = 0.1945)
```
"""
function profileWignerLaw(S::AbstractMatrix{<:Real};
                            npoints::Integer=2001, eps=nothing)
    Sf = _profile_matrix(S, "profileWignerLaw")
    n = size(Sf, 1)
    reps, counts, Ared = _qve_groups(Sf)
    G = length(reps)
    zerogroup = [all(Sf[reps[a], :] .== 0.0) for a in 1:G]
    nz = findall(.!zerogroup)
    zeromass = sum(counts[zerogroup]) / n
    isempty(nz) && return diracLaw(0.0)
    A = Ared[nz, nz]
    wts = counts[nz] ./ n
    rmax = maximum(sum(Ared, dims=2))
    B = 2 * sqrt(rmax)
    half = B + 0.04 * B
    xs = 0.0 .- half .* cos.(range(0, π; length=npoints))
    ε = eps === nothing ? (xs[end] - xs[1]) / (20 * (npoints - 1)) : eps
    dens = zeros(npoints)
    mstate = fill(complex(0.0, 0.0), length(nz))
    # Continuation along the grid: march from the right end, where
    # m ~ -1/z is nearly exact, leftward through the support.
    for i in npoints:-1:1
        z = xs[i] + im * ε
        if i == npoints
            mstate = fill(-1 / z, length(nz))
        end
        m, res = _qve_solve(A, xs[i], ε, mstate; B=B)
        if res > 1e-6
            # Cold-start retry; keep whichever solve is better.
            m2, res2 = _qve_solve(A, xs[i], ε, fill(-1 / z, length(nz)); B=B)
            if res2 < res
                m, res = m2, res2
            end
        end
        mstate = m
        dens[i] = max(sum(wts .* imag.(m)) / π, 0.0)
    end
    atoms = zeromass > 0 ? [0.0 => zeromass] : Pair{Float64,Float64}[]
    return _density_measure(xs, dens, atoms)
end

function profileWignerLaw(sizes::AbstractVector{<:Integer},
                          B::AbstractMatrix{<:Real};
                          npoints::Integer=2001, eps=nothing)
    K = length(sizes)
    size(B) == (K, K) ||
        throw(ArgumentError("profileWignerLaw: block matrix B must be $(K)x$(K) for $K block sizes, got $(size(B))"))
    all(s -> s >= 1, sizes) ||
        throw(ArgumentError("profileWignerLaw: block sizes must be >= 1"))
    Bf = _profile_matrix(B, "profileWignerLaw")
    S = zeros(sum(sizes), sum(sizes))
    off = cumsum(vcat(0, collect(sizes)))
    for a in 1:K, b in 1:K
        S[off[a]+1:off[a+1], off[b]+1:off[b+1]] .= Bf[a, b]
    end
    return profileWignerLaw(S; npoints=npoints, eps=eps)
end

"""
    randProfileWigner([rng,] S; complex=false)

A random Hermitian matrix with variance profile `S` (see
[`profileWignerLaw`](@ref) for the limiting law): entries independent
up to the symmetry, `X_ij` centered Gaussian with
`E|X_ij|^2 = S_ij / n`, `n = size(S, 1)`. The real case follows the
GOE convention (diagonal variance `2 S_ii / n`, as in
`rand(GOE(n))`); the complex case follows the GUE convention
(diagonal variance `S_ii / n`, real). Returns a `Symmetric` (real) or
`Hermitian` (complex) matrix.

# Examples

```julia
X = randProfileWigner(fill(1.0, 50, 50))   # GOE-like, bulk on [-2, 2]
```
"""
function randProfileWigner(rng::AbstractRNG, S::AbstractMatrix{<:Real};
                           complex::Bool=false)
    Sf = _profile_matrix(S, "randProfileWigner")
    n = size(Sf, 1)
    if complex
        X = zeros(ComplexF64, n, n)
        for i in 1:n
            X[i, i] = sqrt(Sf[i, i] / n) * randn(rng)
            for j in i+1:n
                v = sqrt(Sf[i, j] / n) * (randn(rng) + im * randn(rng)) / sqrt(2)
                X[i, j] = v
                X[j, i] = conj(v)
            end
        end
        return Hermitian(X)
    else
        X = zeros(Float64, n, n)
        for i in 1:n
            X[i, i] = sqrt(2 * Sf[i, i] / n) * randn(rng)
            for j in i+1:n
                v = sqrt(Sf[i, j] / n) * randn(rng)
                X[i, j] = v
                X[j, i] = v
            end
        end
        return Symmetric(X)
    end
end

randProfileWigner(S::AbstractMatrix{<:Real}; complex::Bool=false) =
    randProfileWigner(Random.default_rng(), S; complex=complex)
