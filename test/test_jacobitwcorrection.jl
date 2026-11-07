# Tests for the unbalanced complex Jacobi Tracy-Widom correction in
# src/JacobiTWCorrection.jl (arXiv:2611.00577): the public functions
# jacobiTWScaling, jacobiTWCDF, jacobiTWThreshold.  These check the
# transcribed constants against the paper's own internal consistency
# relations and limiting values, plus one small seeded simulation
# smoke test.  The full simulation study lives outside the package
# test suite.

using RandomMatrix
using Distributions: cdf, pdf
using LinearAlgebra
using Random
using Test

const RM = RandomMatrix

@testset "Jacobi TW correction (arXiv:2611.00577)" begin
    a, b, nu = 1.0, 1.0, 0.5

    @testset "exports and regime validation" begin
        # exactly the three public names are exported
        @test :jacobiTWScaling in names(RM)
        @test :jacobiTWCDF in names(RM)
        @test :jacobiTWThreshold in names(RM)
        @test !(:_jacobi_tw_params in names(RM))
        # the limiting scale sigmaC is not part of the public API
        @test !(:jacobiTWParams in names(RM))
        @test !(:jacobiTWLocationScale in names(RM))
        @test_throws ArgumentError RM._jacobi_tw_params(100, a, b, 0.0)
        @test_throws ArgumentError RM._jacobi_tw_params(100, -1.0, b, nu)
        @test_throws ArgumentError RM._jacobi_tw_params(100, a, 0.0, nu)
        @test_throws ArgumentError RM._jacobi_tw_params(4, a, b, nu)
        # validation fires through every public entry point
        @test_throws ArgumentError jacobiTWScaling(100, a, b, 0.0)
        @test_throws ArgumentError jacobiTWCDF(0.0, 100, a, b, -0.5)
        @test_throws ArgumentError jacobiTWThreshold(0.0, 4, a, b, nu)
    end

    @testset "endpoints and constants" begin
        for p in (50, 100, 200)
            prm = RM._jacobi_tw_params(p, a, b, nu)
            @test 0 < prm.alpha < prm.beta < 1
            @test prm.sigma > 0
            @test prm.Chat > 0
            @test prm.h1 < 0
            ls = jacobiTWScaling(p, a, b, nu)
            @test keys(ls) == (:location, :scale)
            @test ls.location == prm.beta
            @test ls.scale == prm.sigma
            loc, sc = ls
            @test (loc, sc) == (prm.beta, prm.sigma)
        end
        # the limiting scale sigmaC decays exactly like p^{-(nu + 2/3)}
        s1 = RM._jacobi_tw_params(100, a, b, nu).sigmaC
        s2 = RM._jacobi_tw_params(200, a, b, nu).sigmaC
        @test s1 / s2 ≈ 2^(nu + 2 / 3) rtol=1e-12
        # the finite-p scale sigma decays at the same rate up to the
        # slowly vanishing o(1) in checkC_p
        r1 = RM._jacobi_tw_params(100, a, b, nu).sigma
        r2 = RM._jacobi_tw_params(200, a, b, nu).sigma
        @test 1.5 < r1 / r2 < 2.3
        # location decays like p^{-nu}: beta_p * p^nu stabilizes,
        # with an O(p^{-nu}) relative error (constant ~ 1 at b = 1)
        l1 = RM._jacobi_tw_params(400, a, b, nu).beta * 400^nu
        l2 = RM._jacobi_tw_params(4000, a, b, nu).beta * 4000^nu
        @test l1 ≈ l2 rtol=0.15
    end

    @testset "h_p(1) against the paper's limit (5.46)" begin
        # h_inf for b = 1: -2 sqrt(2) (1+b) / (2 + b + 2 sqrt(1+b)).
        # The paper's error is O(p^{-nu}); at b = 1 the constant is
        # ~4, so at p = 200_000 the relative error is ~0.01.
        hinf = -2 * sqrt(2) * (1 + b) / (2 + b + 2 * sqrt(1 + b))
        prm = RM._jacobi_tw_params(200_000, a, b, nu)
        @test prm.h1 ≈ hinf rtol=0.02
        # and the error decays: it is smaller at 200_000 than at 2_000
        e1 = abs(RM._jacobi_tw_params(2_000, a, b, nu).h1 - hinf)
        e2 = abs(prm.h1 - hinf)
        @test e2 < e1 / 3
    end

    @testset "C against the (4.23) coefficient checkC_p" begin
        # checkC_p = (beta - alpha) / (2 (-h_p(1))^{2/3}) must equal
        # 1 / (C p^nu) (1 + o(1)) for nu > 0, eq. (4.24).  The o(1)
        # is slow at b = 1 (relative deficit ~ 7 p^{-nu}): check the
        # deficit shrinks at the predicted rate instead of a fixed
        # tolerance.
        deficit = function (p)
            prm = RM._jacobi_tw_params(p, a, b, nu)
            return 1 - prm.checkC * prm.C * p^nu
        end
        d100, d4000, d200k = deficit(100), deficit(4_000), deficit(200_000)
        @test 0 < d4000 < d100
        @test 0 < d200k < 0.03
        @test d4000 / d100 < (4_000 / 100)^(-nu) * 1.6
    end

    @testset "Q2 against pdf/cdf of TracyWidom(2)" begin
        # F_2'(s) = F_2(s) * Q2(s)
        for s in (-3.0, -2.0, -1.0, 0.0, 1.0)
            F = cdf(TracyWidom(2), s)
            f = pdf(TracyWidom(2), s)
            @test RM._jacobi_tw_Q2(s) ≈ f / F rtol=0.01
        end
    end

    @testset "threshold map" begin
        for p in (50, 100)
            prm = RM._jacobi_tw_params(p, a, b, nu)
            @test jacobiTWThreshold(0.0, p, a, b, nu) ≈ prm.beta atol=1e-10
            ts = [jacobiTWThreshold(s, p, a, b, nu) for s in -3.0:1.0:2.0]
            @test issorted(ts)
            @test all(0 .< ts .< 1)
            # thresholds concentrate at the scale sigma_p around beta_p
            @test abs(jacobiTWThreshold(2.0, p, a, b, nu) - prm.beta) < 6 * prm.sigma
            @test abs(jacobiTWThreshold(-3.0, p, a, b, nu) - prm.beta) < 6 * prm.sigma
        end
    end

    @testset "corrected cdf" begin
        p = 100
        ss = collect(-3.5:0.25:2.5)
        raw = [jacobiTWCDF(s, p, a, b, nu; corrected=false) for s in ss]
        cor = [jacobiTWCDF(s, p, a, b, nu) for s in ss]
        @test raw == [cdf(TracyWidom(2), s) for s in ss]
        @test all(0 .<= cor .<= 1)
        @test issorted(cor)
        # the correction is genuinely of order p^{-2/3}: it visibly
        # moves the cdf at p = 100 but by less than 0.1
        @test 0.001 < maximum(abs.(cor .- raw)) < 0.1
        # and it shrinks like p^{-2/3} between p = 100 and p = 800
        d1 = maximum(abs.([jacobiTWCDF(s, 100, a, b, nu) -
                           jacobiTWCDF(s, 100, a, b, nu; corrected=false) for s in ss]))
        d2 = maximum(abs.([jacobiTWCDF(s, 800, a, b, nu) -
                           jacobiTWCDF(s, 800, a, b, nu; corrected=false) for s in ss]))
        @test d1 / d2 ≈ (800 / 100)^(2 / 3) rtol=0.05
    end

    @testset "simulation smoke test (seeded, loose)" begin
        # theta_max of (A + B)^{-1} B, A ~ CW_p(I, m), B ~ CW_p(I, n),
        # at p = 40, nu = 1/2, a = b = 1: n = 2p, m = p + round(p^1.5).
        # Only the location/scale are checked, with Monte-Carlo-aware
        # tolerances; the sharp CDF check is the external study.
        rng = MersenneTwister(2_611_007)
        p = 40
        n = p + round(Int, b * p)
        m = p + round(Int, a * p^(1 + nu))
        prm = RM._jacobi_tw_params(p, a, b, nu)
        R = 300
        thetas = Vector{Float64}(undef, R)
        for r in 1:R
            X = randn(rng, ComplexF64, m, p) ./ sqrt(2)
            Y = randn(rng, ComplexF64, n, p) ./ sqrt(2)
            A = X' * X
            Bm = Y' * Y
            F = cholesky(Hermitian(A + Bm))
            M = Hermitian(F.L \ Bm / F.U)
            thetas[r] = eigmax(M)
        end
        z = (thetas .- prm.beta) ./ prm.sigma
        # TW_2 mean -1.7711, sd 0.9018; MC sd of the mean at R = 300
        # is ~0.052, of the sd ~0.037: tolerances are 6+ MC sigmas.
        @test abs(sum(z) / R - (-1.771086807411)) < 0.35
        @test abs(sqrt(sum((z .- sum(z) / R) .^ 2) / (R - 1)) - 0.9018) < 0.25
    end
end
