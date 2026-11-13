# Tests for lssTest (Bai-Silverstein CLT for linear spectral statistics).
#
# The anchors are the calibration spec of the companion study
# (workspace research bai-silverstein-clt: paper + prototype
# bs_clt.py / results.json): the closed-form mean/variance table,
# verified there against the general contour quadrature, exact Wishart
# trace identities and the corrected-LRT anchor; the Rademacher
# fourth-moment witness; and the p = 200 simulation size calibration
# (empirical size 0.043-0.055 at nominal 5% over 2,500 reps).

using RandomMatrix
using Test, Random, LinearAlgebra, Statistics

@testset "lssTest closed-form anchors" begin
    # (mean, var) at gamma in {0.25, 0.5, 0.8}, real Gaussian case,
    # from the prototype's verified closed forms.
    anchors = Dict(
        0.25 => Dict(:x => (0.0, 0.5), :x2 => (0.25, 3.375),
                     :x3 => (0.9375, 18.392578125),
                     :log => (-0.14384103622589045, 0.5753641449035618)),
        0.5 => Dict(:x => (0.0, 1.0), :x2 => (0.5, 10.0),
                    :x3 => (2.25, 89.0625),
                    :log => (-0.34657359027997264, 1.3862943611198906)),
        0.8 => Dict(:x => (0.0, 1.6), :x2 => (0.8, 23.296),
                    :x3 => (4.32, 312.75264),
                    :log => (-0.8047189562170503, 3.218875824868201)),
    )
    # Build spectra of exact length p with n = p / gamma an integer:
    # gamma = 0.25 -> (p, n) = (50, 200); 0.5 -> (200, 400); 0.8 -> (200, 250).
    for (γ, p, n) in ((0.25, 50, 200), (0.5, 200, 400), (0.8, 200, 250))
        ev = fill(1.0, p) # any finite positive spectrum; mean/var do not depend on it
        for (f, (m, v)) in anchors[γ]
            r = lssTest(ev, n; f=f)
            @test r.mean ≈ m atol = 1e-10
            @test r.var ≈ v atol = 1e-8
            @test r.statistic ≈ (r.G - m) / sqrt(v)
            @test 0 <= r.pvalue <= 1
        end
    end
end

@testset "lssTest corrected-LRT anchor" begin
    # For g = x - 1 - log x the corrected likelihood-ratio statistic
    # has mean -log(1-gamma)/2 and var -2 log(1-gamma) - 2 gamma
    # (Bai-Jiang-Yao-Zheng anchor). With Cov(x, log x) = 2 gamma this
    # must be reproduced by the shipped :x and :log mean/var.
    for γ in (0.25, 0.5, 0.8)
        mx, vx = RandomMatrix._lss_mean_var(:x, γ, 1, nothing)
        ml, vl = RandomMatrix._lss_mean_var(:log, γ, 1, nothing)
        @test mx - ml ≈ -0.5 * log(1 - γ) atol = 1e-12
        @test vx + vl - 2 * (2γ) ≈ -2 * log(1 - γ) - 2γ atol = 1e-12
    end
end

@testset "lssTest beta=2 and fourth-moment corrections" begin
    ev = fill(1.0, 200)
    r1 = lssTest(ev, 400; f=:x3, beta=1)
    r2 = lssTest(ev, 400; f=:x3, beta=2)
    @test r2.mean == 0.0
    @test r2.var ≈ r1.var / 2
    # Rademacher entries (mu4 = 1, kappa2 = -2) at gamma = 0.5:
    # monomial mean scales by (mu4 - 2) = -1, var gains (mu4 - 3) * V2.
    rx2 = lssTest(ev, 400; f=:x2, mu4=1.0)
    @test rx2.mean ≈ -0.5 atol = 1e-12
    @test rx2.var ≈ 1.0 atol = 1e-12 # 10 - 2 * 4 * 0.5 * (1.5)^2
    # log x takes its own kappa2 coefficients (-gamma/2 mean, gamma var):
    # the paper's Rademacher witness values.
    rlog = lssTest(ev, 400; f=:log, mu4=1.0)
    @test rlog.mean ≈ 0.15342640972002736 atol = 1e-12
    @test rlog.var ≈ 0.3862943611198906 atol = 1e-12
    # Gaussian matching is mu4 = 3.
    @test lssTest(ev, 400; f=:x2, mu4=3.0).mean ≈ lssTest(ev, 400; f=:x2).mean atol = 1e-12
    @test_throws ArgumentError lssTest(ev, 400; f=:x2, beta=2, mu4=1.0)
    @test_throws ArgumentError lssTest(ev, 400; f=:x2, mu4=0.5)
end

@testset "lssTest input validation" begin
    ev = fill(1.0, 200)
    @test_throws ArgumentError lssTest(ev, 400; f=:x4)
    @test_throws ArgumentError lssTest(ev, 400; beta=3)
    @test_throws ArgumentError lssTest(ev, 400; sigma2=0.0)
    @test_throws ArgumentError lssTest(ev, 400; sigma2=-1.0)
    @test_throws ArgumentError lssTest(Float64[], 400)
    @test_throws ArgumentError lssTest(ev, 1)
    @test_throws ArgumentError lssTest([1.0, Inf, 2.0], 10)
    # log: gamma >= 1 and nonpositive eigenvalues throw
    @test_throws ArgumentError lssTest(fill(1.0, 200), 200; f=:log)
    @test_throws ArgumentError lssTest(vcat([0.0], fill(1.0, 99)), 400; f=:log)
    # sigma2 scaling: testing 2 * MP(0.5, 1) spectrum against sigma2 = 2
    # must equal testing the unit spectrum against sigma2 = 1.
    a = lssTest(2 .* ev, 400; f=:x2, sigma2=2.0)
    b = lssTest(ev, 400; f=:x2)
    @test a.G ≈ b.G
    @test a.statistic ≈ b.statistic
end

@testset "lssTest Rademacher degenerate trace" begin
    # Entries +-1: tr S = p exactly, so G(x) is identically 0.
    rng = MersenneTwister(20261113)
    p, n = 200, 400
    X = rand(rng, (-1.0, 1.0), p, n)
    ev = eigvals(Symmetric((X * X') ./ n))
    @test sum(ev) ≈ p atol = 1e-8
    r = lssTest(ev, n; f=:x, mu4=1.0)
    @test r.G ≈ 0 atol = 1e-8
end

@testset "lssTest Wishart simulation (size and moments)" begin
    # Seeded Wishart null at gamma = 0.5, p = 200 (the paper's
    # calibration cell: empirical size 0.0552 for :x2 over 2,500 reps,
    # mean 0.549 vs predicted 0.5, var ratio 1.039). Bands here are
    # Monte Carlo bands at 400 reps, wide enough to be stable.
    rng = MersenneTwister(20261113)
    p, n, reps = 200, 400, 400
    Gs = Vector{Float64}(undef, reps)
    rej = 0
    for i in 1:reps
        X = randn(rng, p, n)
        ev = eigvals(Symmetric((X * X') ./ n))
        r = lssTest(ev, n; f=:x2)
        Gs[i] = r.G
        rej += r.pvalue < 0.05
    end
    @test abs(mean(Gs) - 0.5) < 3.5 * sqrt(10.0 / reps)   # E G(x^2) = gamma exactly
    @test 0.75 < var(Gs) / 10.0 < 1.3                     # var ratio ~1.04 in the paper
    @test 0.015 <= rej / reps <= 0.095                    # nominal 0.05
    # Exact finite-n trace identities (Gaussian): E G(x^2) = gamma_n
    # and E G(x^3) = 3 gamma (1 + gamma) + 4 gamma / n; checked on a
    # smaller seeded ensemble against MC error.
    rng2 = MersenneTwister(20261114)
    p2, n2, reps2 = 50, 100, 2000
    s2 = 0.0; s3 = 0.0
    for _ in 1:reps2
        X = randn(rng2, p2, n2)
        ev = eigvals(Symmetric((X * X') ./ n2))
        s2 += lssTest(ev, n2; f=:x2).G
        s3 += lssTest(ev, n2; f=:x3).G
    end
    γ2 = p2 / n2
    @test abs(s2 / reps2 - γ2) < 3.5 * sqrt(10.0 * (γ2 / 0.5) / reps2) + 0.02
    @test abs(s3 / reps2 - (3γ2 * (1 + γ2) + 4γ2 / n2)) < 0.35
end
