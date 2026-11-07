# Tracy-Widom approximation with an explicit p^(-2/3) correction for
# the largest eigenvalue of the *unbalanced* complex Jacobi ensemble,
# after Didier, Gkogkou, Lopez Santander & McLaughlin, "On the
# asymptotic distribution of the largest eigenvalue of unbalanced
# complex Jacobi ensembles", arXiv:2611.00577 (2026).
#
# Model: A ~ CW_p(I, m), B ~ CW_p(I, n) independent complex Wisharts,
# J = (A + B)^{-1} B, theta_max = largest eigenvalue of J.  The regime
# (eq. (1.8) of the paper) is
#
#     n - p = b * p,        m - p = a * p^(1 + nu),        nu > 0,
#
# with a, b > 0; nu = 0 is the balanced regime of Johnstone (2008).
# This file covers nu > 0 only, and only the complex ensemble.
#
# Results implemented:
#   * Location beta_p: right endpoint of the equilibrium support,
#     eq. (2.12).  Scale: Theorem 3.1 is stated in the coordinate
#     theta = beta_p + checkC_p p^(-2/3) s with the finite-p
#     coefficient checkC_p = (beta_p - alpha_p) / (2 (-h_p(1))^(2/3))
#     of eq. (4.23); the public scale is sigma_p = checkC_p p^(-2/3).
#     Corollary 3.2, eq. (3.5) (nu > 0 branch), gives the closed-form
#     limit checkC_p = 1 / (C p^nu) (1 + o(1)); the o(1) converges
#     slowly (relative error ~ 7 p^(-nu) at b = 1), so at every
#     simulable p the finite-p coefficient is the accurate one and
#     the limiting scale 1 / (C p^(nu + 2/3)) is NOT exposed as a
#     public scale (it survives only inside `_jacobi_tw_params`,
#     where the test suite checks the (4.24) consistency relation).
#   * Theorem 3.1: with phi_p the affine map (2.13) and chi_p the
#     local Airy map (2.14),
#         P(theta_max <= phi_p(chi_p^{-1}(s)))
#             = F_2(s) * (1 - 2 Chat_p p^(-2/3) v(s)) + O(p^(-1)),
#     where v(s) = ( (int_s^inf q^2)^2 - q(s)^2 ) / 2 for the
#     Hastings-McLeod Painleve II function q (the same q tabulated in
#     TracyWidom.jl), and Chat_p is the explicit coefficient of
#     eq. (4.7), built from h_p of eq. (5.45).
#
# Validation: standardized theta_max matches the TracyWidom(2)
# moments at p = 50-200 and the correction term matches its predicted
# size and p^(-2/3) decay in simulation; see the test suite and the
# changelog entry.

export jacobiTWScaling, jacobiTWCDF, jacobiTWThreshold

function _jacobi_tw_check(p::Integer, a::Real, b::Real, nu::Real)
    p >= 8 || throw(ArgumentError("dimension p must be at least 8, got p = $p"))
    a > 0 || throw(ArgumentError("regime parameter a must be positive, got a = $a"))
    b > 0 || throw(ArgumentError("regime parameter b must be positive, got b = $b"))
    nu > 0 || throw(ArgumentError(
        "the unbalanced Jacobi correction requires nu > 0 (nu = 0 is the " *
        "balanced regime of Johnstone (2008), not covered here), got nu = $nu"))
    return nothing
end

# Equilibrium endpoints alpha_p < beta_p, eq. (2.12).
function _jacobi_tw_endpoints(p::Integer, a::Real, b::Real, nu::Real)
    eta = (b + a * p^nu + 2)^2
    D = a^2 * p^(2 * nu) - b^2 - eta
    disc = sqrt(D^2 - 4 * eta * b^2)
    alpha = (-D - disc) / (2 * eta)
    beta = (-D + disc) / (2 * eta)
    return alpha, beta
end

# R(0) and R(1) for R(z) = ((z - alpha)(z - beta))^(1/2) with the
# branch R(z) ~ z at infinity: R(0) = -sqrt(alpha*beta) (0 < alpha),
# R(1) = +sqrt((1 - alpha)(1 - beta)).
function _jacobi_tw_h(p::Integer, a::Real, b::Real, nu::Real,
                      alpha::Real, beta::Real, zeta::Real)
    R0 = -sqrt(alpha * beta)
    R1 = sqrt((1 - alpha) * (1 - beta))
    s = (alpha + beta) / 2 + (beta - alpha) / 2 * zeta
    T1 = b / (R0 * s)
    T2 = a * p^nu / (R1 * (s - 1))
    return 0.5 * ((beta - alpha) / 2)^2 * sqrt(zeta + 1) * (T1 + T2)
end

# h_p(1) and the logarithmic derivative h_p'(1) / h_p(1), from (5.45)
# with s(1) = beta and s'(zeta) = (beta - alpha) / 2.
function _jacobi_tw_h1(p::Integer, a::Real, b::Real, nu::Real,
                       alpha::Real, beta::Real)
    R0 = -sqrt(alpha * beta)
    R1 = sqrt((1 - alpha) * (1 - beta))
    ds = (beta - alpha) / 2
    T1 = b / (R0 * beta)
    T2 = a * p^nu / (R1 * (beta - 1))
    h1 = 0.5 * ds^2 * sqrt(2) * (T1 + T2)
    dT1 = -T1 * ds / beta
    dT2 = -T2 * ds / (beta - 1)
    hratio = 1 / 4 + (dT1 + dT2) / (T1 + T2)
    return h1, hratio
end

# All constants of the approximation, internal.  Returns a NamedTuple
# with the equilibrium endpoints `alpha`, `beta`; the finite-p
# coefficient `checkC` of eq. (4.23) and the scale
# `sigma = checkC * p^(-2/3)`; the Theorem 3.1 correction coefficient
# `Chat` of eq. (4.7); and `h1 = h_p(1)`, `hratio = h_p'(1) / h_p(1)`
# used to build them.  It also carries the Corollary 3.2 constant `C`
# of eq. (3.5) and the limiting scale `sigmaC = 1 / (C p^(nu + 2/3))`:
# these are asymptotic-limit quantities, kept only so the test suite
# can check the consistency relation (4.24) — no public function
# returns them, because sigmaC agrees with sigma only to a slowly
# vanishing 1 + o(1) and is far too wide at practical p (see
# `jacobiTWScaling`).
function _jacobi_tw_params(p::Integer, a::Real, b::Real, nu::Real)
    _jacobi_tw_check(p, a, b, nu)
    alpha, beta = _jacobi_tw_endpoints(p, a, b, nu)
    sqb = sqrt(1 + b)
    C = 0.5 * a / sqb * (2 * sqrt(2) * (1 + b) / (2 + b + 2 * sqb))^(2 / 3)
    sigmaC = 1 / (C * p^(nu + 2 / 3))
    h1, hratio = _jacobi_tw_h1(p, a, b, nu, alpha, beta)
    c1 = (-h1)^(2 / 3)
    Chat = (0.5 + 0.4 * hratio) / (4 * c1)
    checkC = (beta - alpha) / (2 * c1)
    sigma = checkC * p^(-2 / 3)
    return (alpha=alpha, beta=beta, C=C, checkC=checkC, sigma=sigma,
            sigmaC=sigmaC, Chat=Chat, h1=h1, hratio=hratio)
end

"""
    jacobiTWScaling(p, a, b, nu) -> (location, scale)

Location and scale for the largest eigenvalue `θmax` of the
unbalanced complex Jacobi (MANOVA) matrix `J = (A + B)⁻¹ B`, with
`A ~ CW_p(I, m)`, `B ~ CW_p(I, n)` independent complex Wisharts, in
the regime (eq. (1.8) of Didier et al. 2026)

```math
n - p = b\\, p, \\qquad m - p = a\\, p^{1 + \\nu},
\\qquad a, b > 0, \\ \\nu > 0,
```

so one Wishart is much fatter than the other. The `location` is the
right endpoint `β_p` of the equilibrium support (eq. (2.12)) and the
`scale` is `σ_p = Č_p p^{-2/3}` with the *finite-p* coefficient
`Č_p = (β_p - α_p) / (2 (-h_p(1))^{2/3})` of eq. (4.23):
`(θmax - location) / scale` converges in distribution to
[`TracyWidom`](@ref)`(2)`. The standardization is validated against
simulation at `p = 50`–`200`: standardized mean and sd match the
Tracy-Widom(2) values -1.7711 and 0.9018 within Monte Carlo error at
`p ≥ 100`.

Use the finite-p coefficient, not the closed form of Corollary 3.2:
the corollary's constant `C` (eq. (3.5)) gives the *limiting* scale
`1 / (C p^{ν + 2/3})`, which agrees with `σ_p` only to `1 + o(1)`,
and the `o(1)` vanishes slowly (relative error ``\\sim 7 p^{-\\nu}``
at `b = 1`). At `ν = 1/2` the limiting scale is 2.36, 1.89 and 1.60
times `σ_p` at `p = 50`, `100`, `200` — far too wide for quantiles
or tests at any practical size — so it is deliberately not exposed;
this function returns the finite-p scale.

Only the unbalanced regime `ν > 0` is covered: `ν = 0` is the
balanced regime of Johnstone (2008), and the ensemble here is
complex only. See [`jacobiTWCDF`](@ref) and
[`jacobiTWThreshold`](@ref) for the corrected distribution function
of Theorem 3.1.

Reference: Didier, Gkogkou, Lopez Santander & McLaughlin, "On the
asymptotic distribution of the largest eigenvalue of unbalanced
complex Jacobi ensembles", arXiv:2611.00577 (2026), eqs. (1.8),
(2.12), (4.23).
"""
function jacobiTWScaling(p::Integer, a::Real, b::Real, nu::Real)
    prm = _jacobi_tw_params(p, a, b, nu)
    return (location=prm.beta, scale=prm.sigma)
end

# Q2(s) = int_s^inf q(t)^2 dt from the tabulated Hastings-McLeod q
# (trapezoid on the table grid; q(4.5) ~ 1e-6 makes the dropped tail
# below 1e-11).  Cross-checked in the tests against pdf/cdf, since
# F_2'(s) = F_2(s) * Q2(s).
function _jacobi_tw_Q2(s::Real)
    t = _tw_table()
    xs, q = t.s, t.q
    s >= xs[end] && return 0.0
    # trapezoid over whole cells above s, plus the partial cell
    j = clamp(floor(Int, (s - xs[1]) / (xs[2] - xs[1])) + 1, 1, length(xs) - 1)
    qs = _tw_lookup(q, xs, s)
    total = 0.5 * (qs^2 + q[j + 1]^2) * (xs[j + 1] - s)
    for k in (j + 1):(length(xs) - 1)
        total += 0.5 * (q[k]^2 + q[k + 1]^2) * (xs[k + 1] - xs[k])
    end
    return total
end

# v(s) of eq. (3.2).
function _jacobi_tw_v(s::Real)
    t = _tw_table()
    q = _tw_lookup(t.q, t.s, s)
    Q2 = _jacobi_tw_Q2(s)
    return 0.5 * Q2^2 - 0.5 * q^2
end

# Adaptive Simpson quadrature on [lo, hi].
function _jacobi_tw_quad(f, lo::Real, hi::Real; tol::Real=1e-12)
    lo == hi && return 0.0
    mid = (lo + hi) / 2
    flo, fmid, fhi = f(lo), f(mid), f(hi)
    whole = (hi - lo) / 6 * (flo + 4 * fmid + fhi)
    return _jacobi_tw_quad_rec(f, lo, hi, flo, fmid, fhi, whole, tol, 40)
end

function _jacobi_tw_quad_rec(f, lo, hi, flo, fmid, fhi, whole, tol, depth)
    mid = (lo + hi) / 2
    lmid, rmid = (lo + mid) / 2, (mid + hi) / 2
    flmid, frmid = f(lmid), f(rmid)
    left = (mid - lo) / 6 * (flo + 4 * flmid + fmid)
    right = (hi - mid) / 6 * (fmid + 4 * frmid + fhi)
    delta = left + right - whole
    if depth <= 0 || abs(delta) <= 15 * tol
        return left + right + delta / 15
    end
    return _jacobi_tw_quad_rec(f, lo, mid, flo, flmid, fmid, left, tol / 2, depth - 1) +
           _jacobi_tw_quad_rec(f, mid, hi, fmid, frmid, fhi, right, tol / 2, depth - 1)
end

# The local Airy map chi_p, in the real form valid near lambda = 1
# (h_p < 0 there): with G(lambda) = int between 1 and lambda of
# sqrt|zeta - 1| (-h_p(zeta)) dzeta (positive on both sides),
# chi_p(lambda) = sign(lambda - 1) p^(2/3) (3 G / 2)^(2/3).
# The substitution zeta = 1 +/- u^2 removes the square-root endpoint.
function _jacobi_tw_chi(p::Integer, a::Real, b::Real, nu::Real,
                        alpha::Real, beta::Real, lambda::Real)
    lambda == 1 && return 0.0
    if lambda > 1
        U = sqrt(lambda - 1)
        f = u -> 2 * u^2 * (-_jacobi_tw_h(p, a, b, nu, alpha, beta, 1 + u^2))
        G = _jacobi_tw_quad(f, 0.0, U)
        return p^(2 / 3) * (1.5 * G)^(2 / 3)
    else
        U = sqrt(1 - lambda)
        f = u -> 2 * u^2 * (-_jacobi_tw_h(p, a, b, nu, alpha, beta, 1 - u^2))
        G = _jacobi_tw_quad(f, 0.0, U)
        return -p^(2 / 3) * (1.5 * G)^(2 / 3)
    end
end

"""
    jacobiTWThreshold(s, p, a, b, nu)

The threshold `φ_p(χ_p⁻¹(s))` of Theorem 3.1 of Didier et al.
(2026): the value `t` such that `P(θmax ≤ t)` is approximated by
[`jacobiTWCDF`](@ref)`(s, p, a, b, nu)`, for the largest eigenvalue
`θmax` of the unbalanced complex Jacobi ensemble of
[`jacobiTWScaling`](@ref). Here `φ_p(λ) = (α_p + β_p)/2 +
(β_p - α_p) λ/2` is the affine map onto the equilibrium support
`[α_p, β_p]` (eq. (2.13)) and `χ_p` is the local Airy map of
eq. (2.14), inverted here by bisection.
`jacobiTWThreshold(0, ...)` is exactly the location `β_p`.

An approximate level-`u` quantile of `θmax` is
`jacobiTWThreshold(s_u, p, a, b, nu)` with `s_u` solving
`jacobiTWCDF(s_u, p, a, b, nu) = u`; to leading order
`s_u = quantile(TracyWidom(2), u)`.
"""
function jacobiTWThreshold(s::Real, p::Integer, a::Real, b::Real, nu::Real)
    _jacobi_tw_check(p, a, b, nu)
    alpha, beta = _jacobi_tw_endpoints(p, a, b, nu)
    chi = lam -> _jacobi_tw_chi(p, a, b, nu, alpha, beta, lam)
    lo, hi = 1.0, 1.0
    step = 0.05
    if s >= 0
        while chi(hi) < s
            hi += step
            hi < 1e6 || error("jacobiTWThreshold: failed to bracket s = $s")
        end
    else
        while chi(lo) > s
            lo -= step
            lo > -1e6 || error("jacobiTWThreshold: failed to bracket s = $s")
        end
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        chi(mid) < s ? (lo = mid) : (hi = mid)
        hi - lo < 1e-13 && break
    end
    lam = (lo + hi) / 2
    return (alpha + beta) / 2 + (beta - alpha) / 2 * lam
end

"""
    jacobiTWCDF(s, p, a, b, nu; corrected = true)

Theorem 3.1 of Didier et al. (2026): the corrected Tracy-Widom
approximation to the distribution of the largest eigenvalue
`θmax` of the unbalanced complex Jacobi ensemble of
[`jacobiTWScaling`](@ref),

```math
P\\bigl(\\theta_{\\max} \\le \\varphi_p(\\chi_p^{-1}(s))\\bigr)
\\ = \\ F_2(s) \\bigl(1 - 2\\, \\hat{C}_p\\, p^{-2/3}\\, v(s)\\bigr)
\\ + \\ O(p^{-1}),
```

where the threshold `φ_p(χ_p⁻¹(s))` is [`jacobiTWThreshold`](@ref),
`F_2` is the cdf of [`TracyWidom`](@ref)`(2)`, `Ĉ_p` is the
explicit coefficient of eq. (4.7), and
`v(s) = ( (∫_s^∞ q²)² - q(s)² ) / 2` for the Hastings-McLeod
Painlevé II function `q` (eq. (3.2); the same `q` tabulated for
`TracyWidom`). With `corrected = false`, returns the plain
Tracy-Widom value `F_2(s)`, whose error is the displayed
`p^{-2/3}` term itself.

The correction term is genuinely small at practical sizes — at most
about 0.003 of cdf at `p = 50`, `ν = 1/2`, shrinking like
`p^{-2/3}` (simulation confirms its predicted size and decay) — so
the theorem's `O(p^{-1})` remainder, not the displayed term, is the
dominant finite-p error; the standardization of
[`jacobiTWScaling`](@ref) is the part that matters most in practice.
"""
function jacobiTWCDF(s::Real, p::Integer, a::Real, b::Real, nu::Real;
                     corrected::Bool=true)
    _jacobi_tw_check(p, a, b, nu)
    F = cdf(TracyWidom(2), s)
    corrected || return F
    prm = _jacobi_tw_params(p, a, b, nu)
    return F * (1 - 2 * prm.Chat * p^(-2 / 3) * _jacobi_tw_v(s))
end
