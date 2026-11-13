using RandomMatrix
using Distributions: Normal, Semicircle, cdf, Gamma
using LinearAlgebra
using Random
using Test
using Aqua

Random.seed!(20261024)

@testset "IID matrices" begin
    @test randDiagonal(2) !== nothing
    @test randDiagonal(4) isa Diagonal
    @test size(randDiagonal(4)) == (4, 4)
    @test randTriangular(1:3, 3, upper=false, Diag=false) isa LowerTriangular
    @test randTriangular(3) isa UpperTriangular
    @test size(randMatrix(10)) == (10, 10)
    @test size(randMatrix(3, 2)) == (3, 2)
    @test !isreal(randMatrix(10) |> eigvals)
end

@testset "Hermitian / symmetric ensembles" begin
    @test isreal(randHermitian(3) |> eigvals)
    @test ishermitian(randHermitian(5))
    @test issymmetric(randSymmetric(5))
    @test randHermitian(ComplexNormal(im, 2), 3, diag=Elliptic(0.1, c=im, R=9)) !== nothing
    G = rand(GOE(4))
    @test size(G) == (4, 4)
    @test issymmetric(G)
    @test size(rand(GOE(4), norm=true)) == (4, 4)
    U = rand(GUE(4))
    @test size(U) == (4, 4)
    @test ishermitian(U)
end

@testset "Unitary / orthogonal" begin
    U = randUnitary(5)
    @test U' * U ≈ I
    O = randOrthogonal(5)
    @test O' * O ≈ I
    @test isreal(O)
    H = rand(Haar(2, 4))
    @test H' * H ≈ I
    C = rand(COE(4))
    @test C' * C ≈ I
    P = randPermutation(6)
    @test size(P) == (6, 6)
    @test all(sum(P, dims=1) .== 1)
    @test all(sum(P, dims=2) .== 1)
    Pf = randPermutation(10, fix=7)
    @test sum(Pf[i, i] for i in 1:10) >= 7
end

@testset "Structured matrices" begin
    T = randToeplitz(4)
    @test all(T[i, j] == T[i + 1, j + 1] for i in 1:3, j in 1:3)
    H = randHankel(4)
    @test all(H[i, j] == H[i + 1, j - 1] for i in 1:3, j in 2:4)
    S1 = randStochastic(4, type=1)
    @test all(sum(S1, dims=2) .≈ 1)
    S2 = randStochastic(4, type=2)
    @test all(sum(S2, dims=1) .≈ 1)
    S3 = randStochastic(4)
    @test all(sum(S3, dims=2) .≈ 1)
    @test all(sum(S3, dims=1) .≈ 1)
    A = randErdosRenyi(6, 0.5)
    @test size(A) == (6, 6)
    E = randElliptic(4)
    @test size(E) == (4, 4)
    # r is a pair correlation: the valid range is (-1, 1] (r = 1 is the
    # Hermitian case). Outside it the call used to leak a PosDefException
    # from the bivariate-normal construction inside preNORTA.
    @test size(randElliptic(4, r=1.0)) == (4, 4)
    @test size(randElliptic(4, r=-0.999)) == (4, 4)
    @test_throws ArgumentError randElliptic(4, r=-1.0)
    @test_throws ArgumentError randElliptic(4, r=1.5)
    @test_throws ArgumentError randElliptic(4, r=-2.0)
    @test_throws ArgumentError randElliptic(4, r=NaN)
    @test_throws ArgumentError randElliptic(4, r=Inf)
end

@testset "Scalar distributions" begin
    @test pdf(MarchenkoPastur(rand()), 0) == 0
    @test pdf(MarchenkoPastur(0.5), -1.0) == 0
    @test pdf(MarchenkoPastur(0.5), 10.0) == 0
    @test pdf(MarchenkoPastur(0.5), 1.0) > 0
    @test rand(MarchenkoPastur(0.5)) >= 0
    @test eltype(rand(ComplexNormal(), 3)) == ComplexF64
    @test mean(ComplexNormal(1 + 2im, 3)) == 1 + 2im
    @test abs(rand(Circular(0, 1))) <= 1
    @test mean(Circular(2 + im, 5)) == 2 + im
    @test abs(rand(Elliptic(0.5, c=0, R=1))) <= 2
    @test Gaussian(1, 0, 1) isa Normal
    @test Gaussian(2, 0, 1) isa ComplexNormal
    @test stieltjes(Semicircle(2))(0) isa Number
end

@testset "Wachter law" begin
    # Constructor domain: gamma_i > 0 and gamma1 + gamma2 - gamma1*gamma2 > 0.
    @test_throws ArgumentError Wachter(0.0, 0.5)
    @test_throws ArgumentError Wachter(0.5, -1.0)
    @test_throws ArgumentError Wachter(3.0, 3.0)  # n1 + n2 <= p: A + B singular
    @test Wachter() == Wachter(0.5, 0.5)
    # Mean and variance closed forms.
    @test mean(Wachter(0.25, 0.5)) ≈ 2 / 3
    @test mean(Wachter(0.5, 0.25)) ≈ 1 / 3
    @test var(Wachter(0.5, 0.5)) ≈ 0.0625
    @test var(Wachter(0.25, 0.5)) ≈ 0.125^2 / 0.75^3
    # Support edges for (0.5, 0.5): (sqrt(3) -/+ 1)^2 / 8 form, 0.066987.../0.933012...
    d = Wachter(0.5, 0.5)
    @test pdf(d, 0.06) == 0
    @test pdf(d, 0.94) == 0
    @test pdf(d, -0.1) == 0
    @test pdf(d, 1.1) == 0
    @test pdf(d, 0.5) ≈ 1.1026577908435842 atol = 1e-12
    @test pdf(Wachter(0.25, 0.5), 0.5) ≈ 1.2732395447351628 atol = 1e-12
    # Swap symmetry: 1 - X for X ~ Wachter(g1, g2) is Wachter(g2, g1).
    @test pdf(Wachter(0.25, 0.5), 0.4) ≈ pdf(Wachter(0.5, 0.25), 0.6) atol = 1e-14
    @test mean(Wachter(0.25, 0.5)) ≈ 1 - mean(Wachter(0.5, 0.25))
    # cdf: endpoints, symmetry point, and a numerically integrated anchor.
    @test cdf(d, -1.0) == 0
    @test cdf(d, 0.0) == 0
    @test cdf(d, 1.0) == 1
    @test cdf(d, 0.5) ≈ 0.5 atol = 1e-10
    @test cdf(d, 0.25) ≈ 0.21634689593878545 atol = 1e-8
    @test cdf(Wachter(0.25, 0.5), 0.5) ≈ 0.23791808825216623 atol = 1e-8
    @test cdf(Wachter(1.0, 0.5), 0.8889) ≈ 1.0 atol = 1e-3  # edge touches 0 at gamma1 = 1
    @test cdf(Wachter(1.0, 0.5), 0.8) ≈ 0.9427508529513049 atol = 1e-10
    # Atoms: gamma1 > 1 puts mass 1 - 1/gamma1 at 0, gamma2 > 1 at 1.
    da = Wachter(1.5, 0.5)
    @test pdf(da, 0) ≈ 1 / 3
    @test pdf(da, 1) == 0
    @test cdf(da, 0.0) ≈ 1 / 3
    @test mean(da) ≈ 0.25
    db = Wachter(0.5, 1.5)
    @test pdf(db, 1) ≈ 1 / 3
    @test cdf(db, prevfloat(1.0)) ≈ 2 / 3 atol = 1e-6
    # Sampling: range, moments, atom frequencies (seeded, generous bands).
    Random.seed!(20261101)
    xs = rand(d, 20_000)
    @test all(0 .<= xs .<= 1)
    @test mean(xs) ≈ 0.5 atol = 0.01
    @test var(xs) ≈ 0.0625 atol = 0.005
    Random.seed!(20261102)
    xa = rand(da, 20_000)
    @test count(==(0.0), xa) / 20_000 ≈ 1 / 3 atol = 0.02
    @test mean(xa) ≈ 0.25 atol = 0.01
    # MANOVA simulation calibration: eigenvalues of (A + B)^{-1} A for
    # p = 120, n1 = 240 (gamma1 = 0.5), n2 = 480 (gamma2 = 0.25).
    Random.seed!(20261103)
    p, n1, n2 = 120, 240, 480
    evals = Float64[]
    for _ in 1:40
        X = randn(p, n1)
        Y = randn(p, n2)
        A = X * X'
        B = Y * Y'
        L = cholesky(Symmetric(A + B)).L
        M = L \ A / L'
        append!(evals, eigvals(Symmetric((M + M') / 2)))
    end
    dm = Wachter(0.5, 0.25)
    @test mean(evals) ≈ mean(dm) atol = 0.01
    @test var(evals) ≈ var(dm) atol = 0.005
    for (lo, hi) in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8)]
        emp = count(x -> lo <= x < hi, evals) / length(evals)
        theo = cdf(dm, hi) - cdf(dm, lo) + (lo == 0.0 ? cdf(dm, 0.0) : 0.0)
        @test emp ≈ theo atol = 0.03
    end
    # fitWachter: method-of-moments inversion of the law's mean/variance.
    # The inversion is algebraically exact — the fitted law's moments
    # reproduce the input spectrum's moments to roundoff.
    Random.seed!(20261108)
    xs = rand(Wachter(0.5, 0.25), 5_000)
    f = fitWachter(xs)
    @test mean(Wachter(f.γ₁, f.γ₂)) ≈ mean(xs) atol = 1e-12
    @test var(Wachter(f.γ₁, f.γ₂)) ≈ var(xs; corrected = false) atol = 1e-12
    @test f.γ₁ ≈ 0.5 atol = 0.03   # seeded fit: (0.5044, 0.2468)
    @test f.γ₂ ≈ 0.25 atol = 0.03
    # Pooled MANOVA spectrum above (true gamma1 = 0.5, gamma2 = 0.25):
    # seeded fit (0.5042, 0.2523).
    fm = fitWachter(evals)
    @test fm.γ₁ ≈ 0.5 atol = 0.02
    @test fm.γ₂ ≈ 0.25 atol = 0.02
    # Atom regime (gamma1 = 1.5 > 1): seeded fit (1.5022, 0.4998).
    Random.seed!(20261112)
    xa2 = rand(da, 20_000)
    fa = fitWachter(xa2)
    @test fa.γ₁ ≈ 1.5 atol = 0.02
    @test fa.γ₂ ≈ 0.5 atol = 0.02
    # Degenerate / impossible spectra throw.
    @test_throws ArgumentError fitWachter(Float64[])
    @test_throws ArgumentError fitWachter([0.5, 1.2])
    @test_throws ArgumentError fitWachter(fill(0.5, 10))        # zero variance
    @test_throws ArgumentError fitWachter(zeros(10))            # single atom at 0
    @test_throws ArgumentError fitWachter([0.0, 1.0, 0.0, 1.0]) # Bernoulli bound
    @test_throws ArgumentError fitWachter([0.5, NaN])
end

@testset "Spacing and transformations" begin
    @test wignersurmise(2) ≈ 0.1357605281502967
    @test wignersurmise(2, beta=2) ≈ 0.07962814352725955
    E = sort(randn(50))
    @test length(unfoldedSpacings(E)) == 49
    @test numberVariance(float.(collect(1:20)), 2.0) >= 0
    A = randHermitian(6)
    R = Resolvent(A)
    @test R(0.5 + 0.1im) * (A - (0.5 + 0.1im) * I) ≈ I atol = 1e-8
    @test (1 ± 0.5) == [0.5, 1.5]
end

@testset "Sampling" begin
    A = rand(3, 4)
    B = rand(4, 2)
    S = randSampling(A, B, k=2)
    @test size(S) == (4, 2)
end

@testset "Spiked covariance / BBP" begin
    γ = 0.5
    @test bbpThreshold(γ) ≈ 1 + sqrt(γ)
    @test spikeLocation(4.0, γ) ≈ 4 * (1 + γ / 3)
    @test spikeOverlap(4.0, γ) ≈ (1 - γ / 9) / (1 + γ / 3)
    @test estimateSpike(spikeLocation(4.0, γ), γ) ≈ 4.0 atol = 1e-10
    # Subcritical spike: sticks to the bulk edge, zero overlap, no Gaussian variance.
    @test spikeLocation(1.3, γ) ≈ (1 + sqrt(γ))^2
    @test spikeOverlap(1.3, γ) == 0.0
    @test isnan(spikeVariance(1.3, γ, 400))
    @test spikeVariance(4.0, γ, 400) ≈ 2 * 16 * (1 - γ / 9) / 400
    # Rectangular case.
    @test svdThreshold(0.25) ≈ 0.25^0.25
    @test svdSpikeLocation(2.0, 0.25) ≈ sqrt(5 * (1 + 0.25 / 4))
    @test collect(svdSpikeOverlaps(2.0, 0.25)) ≈ [(16 - 0.25) / (4 * 4.25), (16 - 0.25) / (4 * 5)]
    @test svdSpikeOverlaps(0.5, 0.25) == (0.0, 0.0)
    # Simulation: outlier location within fluctuation scale; subcritical sticks near edge.
    S = randSpikedCovariance(200, 400, [4.0])
    @test abs(eigmax(S) - spikeLocation(4.0, γ)) < 3.5 * sqrt(spikeVariance(4.0, γ, 400))
    S2 = randSpikedCovariance(200, 400, [1.3])
    @test eigmax(S2) < (1 + sqrt(γ))^2 + 0.35
end

@testset "Spiked MANOVA / BBP" begin
    γ₁, γ₂ = 0.5, 0.5
    ρ = sqrt(γ₁ + γ₂ - γ₁ * γ₂)
    x₋ = γ₂ * (γ₁ - ρ)^2 / (γ₁ + γ₂)^2
    x₊ = γ₂ * (γ₁ + ρ)^2 / (γ₁ + γ₂)^2
    @test manovaThreshold(γ₁, γ₂) ≈ (1 + ρ) / (1 - γ₂)
    # Supercritical upper spike: phi(6) = 16.5 and t = phi (γ₂ = γ₁),
    # so the MANOVA location is 16.5 / 17.5.
    @test manovaSpikeLocation(6.0, γ₁, γ₂) ≈ 16.5 / 17.5
    # The map is continuous at both critical values: it returns the
    # Wachter support edges.
    θ₊ = manovaThreshold(γ₁, γ₂)
    θ₋ = (1 - ρ) / (1 - γ₂)
    @test manovaSpikeLocation(θ₊, γ₁, γ₂) ≈ x₊ atol = 1e-12
    @test manovaSpikeLocation(θ₋, γ₁, γ₂) ≈ x₋ atol = 1e-12
    # Subcritical spikes stick to the nearer edge.
    @test manovaSpikeLocation(2.0, γ₁, γ₂) ≈ x₊
    @test manovaSpikeLocation(0.5, γ₁, γ₂) ≈ x₋
    # Supercritical lower spike, phi and the t/(1+t) map spelt out.
    λlow = 0.1 * (0.1 - 0.5) / (0.05 - 1)
    @test manovaSpikeLocation(0.1, γ₁, γ₂) ≈ λlow / (1 + λlow)
    # Asymmetric ratios, checked against the map spelt out.
    θ, a, b = 5.0, 0.25, 0.5
    λa = θ * (θ - 1 + a) / ((1 - b) * θ - 1)
    ta = (b / a) * λa
    @test manovaSpikeLocation(θ, a, b) ≈ ta / (1 + ta)
    @test manovaThreshold(a, b) ≈ (1 + sqrt(a + b - a * b)) / (1 - b)
    @test_throws ArgumentError manovaThreshold(0.5, 1.0)
    @test_throws ArgumentError manovaThreshold(0.0, 0.5)
    @test_throws ArgumentError manovaSpikeLocation(-1.0, γ₁, γ₂)
    # Fluctuation variance: deterministic anchors are the p-scaled
    # values from the DJO Prop. 11 formula, validated against
    # simulation grids before the bands were fixed (see the simulation
    # variance check at the end of this testset).
    @test manovaSpikeVariance(6.0, γ₁, γ₂, 100) * 100 ≈ 0.0058475635152019974 atol = 1e-12
    @test manovaSpikeVariance(0.1, γ₁, γ₂, 100) * 100 ≈ 0.0038577611830804938 atol = 1e-12
    @test manovaSpikeVariance(5.0, 0.25, 0.5, 200) * 200 ≈ 0.0017342138515128755 atol = 1e-12
    # Independent spelling of the formula for an asymmetric case:
    # Fisher variance from h = theta - 1, then the (g2/g1)/(1+t)^2
    # Jacobian of the map t -> t/(1+t), t = (g2/g1) phi.
    θv, a2, b2 = 4.0, 0.5, 0.25
    hv = θv - 1
    denv = (1 - b2) * θv - 1
    VFv = 2 * (a2 + b2 - a2 * b2) * hv^2 * θv^2 * (hv^2 - b2 * θv^2 - a2) / denv^4
    tv = (b2 / a2) * (θv * (θv - 1 + a2) / denv)
    @test manovaSpikeVariance(θv, a2, b2, 100) ≈ (b2 / a2)^2 / (1 + tv)^4 * VFv / 100 atol = 1e-14
    # Inside the critical window (and at its edges) there is no
    # Gaussian variance at the 1/p scale.
    @test isnan(manovaSpikeVariance(2.0, γ₁, γ₂, 100))
    @test isnan(manovaSpikeVariance(θ₊, γ₁, γ₂, 100))
    @test isnan(manovaSpikeVariance(θ₋, γ₁, γ₂, 100))
    # Complex data halves the variance; bad inputs throw.
    @test manovaSpikeVariance(6.0, γ₁, γ₂, 100; real=false) ≈
        manovaSpikeVariance(6.0, γ₁, γ₂, 100) / 2
    @test_throws ArgumentError manovaSpikeVariance(-1.0, γ₁, γ₂, 100)
    @test_throws ArgumentError manovaSpikeVariance(6.0, 0.5, 1.0, 100)
    @test_throws ArgumentError manovaSpikeVariance(6.0, γ₁, γ₂, 0)
    # Simulation: eigenvalues of (A + B)^{-1} A with A ~ W_p(Σ, n₁),
    # Σ = diag(θ, 1, ..., 1), B ~ W_p(I, n₂).
    function manova_eigs(θ, p, n₁, n₂)
        G = randn(n₁, p)
        G[:, 1] .*= sqrt(θ)
        A = Matrix(Symmetric(G' * G))
        H = randn(n₂, p)
        B = Matrix(Symmetric(H' * H))
        C = cholesky(Symmetric(A + B))
        M = (C.L \ A) / C.U
        return eigvals(Symmetric((M + M') / 2))
    end
    avg = xs -> sum(xs) / length(xs)  # Statistics is loaded further down this file
    p, n₁, n₂, reps = 100, 200, 200, 6
    top = [maximum(manova_eigs(6.0, p, n₁, n₂)) for _ in 1:reps]
    @test avg(top) ≈ manovaSpikeLocation(6.0, γ₁, γ₂) atol = 0.02
    stuck = [maximum(manova_eigs(2.0, p, n₁, n₂)) for _ in 1:reps]
    @test x₊ - 0.06 < avg(stuck) < x₊ + 0.05
    bot = [minimum(manova_eigs(0.1, p, n₁, n₂)) for _ in 1:reps]
    @test avg(bot) ≈ manovaSpikeLocation(0.1, γ₁, γ₂) atol = 0.015
    # Simulation: empirical variance of the outlier against
    # manovaSpikeVariance, both sides of the window. The band is
    # generous (120 reps; the 2026-11-06 calibration grids at
    # p = 100-200 with 300-1500 reps gave ratios 0.86-1.10 across
    # nine (theta, g1, g2) points, Fisher and MANOVA scales alike).
    empvar(xs) = (m = avg(xs); sum((xs .- m) .^ 2) / (length(xs) - 1))
    p₂, n₃ = 80, 160
    tops = [maximum(manova_eigs(6.0, p₂, n₃, n₃)) for _ in 1:120]
    @test 0.45 < empvar(tops) / manovaSpikeVariance(6.0, γ₁, γ₂, p₂) < 2.0
    bots = [minimum(manova_eigs(0.1, p₂, n₃, n₃)) for _ in 1:120]
    @test 0.45 < empvar(bots) / manovaSpikeVariance(0.1, γ₁, γ₂, p₂) < 2.0
end

using Statistics: mean, std

@testset "Covariance cleaning" begin
    p, n = 300, 600
    γ = p / n
    X = randn(p, n) / sqrt(n)
    S = Symmetric(X * X')
    ev = eigvals(S)
    @test fitMarchenkoPastur(ev, γ) ≈ 1.0 atol = 0.06
    @test fitMarchenkoPastur(ev, γ; method=:median) ≈ 1.0 atol = 0.06
    # MP median across regimes: for gamma > 1 the law has an atom at 0 of
    # mass 1 - 1/gamma, so the median is 0 for gamma >= 2 and lies inside
    # the continuous part for 1 < gamma < 2. Check against an independent
    # midpoint-rule CDF of the density (plus the atom).
    function mp_cdf(γ, x)
        a, _ = RandomMatrix._mp_edges(γ, 1.0)
        atom = γ > 1 ? 1 - 1 / γ : 0.0
        x <= a && return atom
        d = MarchenkoPastur(γ, 1.0)
        npts = 100_000
        h = (x - a) / npts
        return atom + h * sum(pdf(d, a + (i - 0.5) * h) for i in 1:npts)
    end
    for γm in (0.25, 0.5, 1.0, 1.5)
        @test mp_cdf(γm, RandomMatrix._mp_median(γm)) ≈ 0.5 atol = 2e-3
    end
    @test RandomMatrix._mp_median(2.0) == 0.0
    @test RandomMatrix._mp_median(3.0) == 0.0
    # Median matching at gamma = 1 (previously sigma^2 = Inf) and in (1, 2).
    X1 = randn(500, 500) / sqrt(500)
    ev1 = eigvals(Symmetric(X1 * X1'))
    @test fitMarchenkoPastur(ev1, 1.0; method=:median) ≈ 1.0 atol = 0.08
    X2 = randn(600, 400) / sqrt(400)
    ev2 = eigvals(Symmetric(X2 * X2'))
    @test fitMarchenkoPastur(ev2, 1.5; method=:median) ≈ 1.0 atol = 0.08
    @test_throws ArgumentError fitMarchenkoPastur(ev2, 2.5; method=:median)
    cl = clipEigenvalues(ev, γ)
    @test sum(cl) ≈ sum(ev) atol = 1e-8
    ξ = rieEigenvalues(ev, γ)
    @test abs(mean(ξ) - 1) < 0.10
    @test std(ξ) < 0.13
    # Spiked population: losses ordered sample > clip > RIE, RIE near oracle.
    d = vcat([6.0, 3.0], fill(1.5, 48), fill(0.7, p - 50))
    Q = Matrix(qr(randn(p, p)).Q)
    Σ = Symmetric(Q * Diagonal(d) * Q')
    X2 = Q * Diagonal(sqrt.(d)) * Q' * randn(p, n) / sqrt(n)
    S2 = Symmetric(X2 * X2')
    F = eigen(S2)
    ora = [dot(F.vectors[:, i], Σ * F.vectors[:, i]) for i in 1:p]
    loss(M) = sum(abs2, M - Σ)
    Srie = F.vectors * Diagonal(rieEigenvalues(F.values, γ)) * F.vectors'
    Sora = F.vectors * Diagonal(ora) * F.vectors'
    Sclip = cleanCovariance(Matrix(S2), n; method=:clip)
    @test loss(Srie) < loss(Matrix(Sclip)) < loss(Matrix(S2))
    @test loss(Srie) < 1.6 * loss(Sora)
    # RIE bracketing of an isolated spike: xi < l < lambda.
    ξ2 = rieEigenvalues(F.values, γ)
    @test ξ2[end] < 6.0 < F.values[end]
    @test_throws ArgumentError cleanCovariance(Matrix(S2), div(n, 2) - 1; method=:rie)
end

@testset "Linear shrinkage (Ledoit-Wolf / OAS)" begin
    # Exact anchors on fixed data matrices (rows = variables, columns =
    # observations), computed independently in Python: a direct numpy
    # implementation of Ledoit & Wolf (2004) agreed with scikit-learn
    # 1.9.1's ledoit_wolf to <= 3e-16 on all three datasets (intensities
    # and matrices). The OAS anchors follow Chen et al. (2010) Eq. (23)
    # WITH the 2/p terms; scikit-learn's oas omits them (its docstring
    # says so) and gives 0.8792 / 1.0 / 0.9578 on datasets A / B / C
    # instead of the values below.
    XA = [1.2 -0.4 0.7 2.1 -1.3 0.3;
          0.5 1.8 -0.9 0.2 1.1 -0.6;
          -1.0 0.3 1.4 -0.2 0.8 1.9;
          0.9 -1.5 0.1 1.6 -0.7 0.4]
    @test ledoitWolfShrinkage(XA) ≈ 0.4106843530339546 atol = 1e-10
    @test oasShrinkage(XA) ≈ 0.7906303094510234 atol = 1e-10
    @test Matrix(ledoitWolf(XA)) ≈ [1.1155896627359558 -0.2799249323088716 -0.2570071015935254 0.5690169969038816;
                                    -0.2799249323088716 0.9212791980279846 -0.25929888466505996 -0.3251057985762684;
                                    -0.2570071015935254 -0.25929888466505996 0.9662963655045576 -0.17253852552839222;
                                    0.5690169969038816 -0.3251057985762684 -0.17253852552839222 1.0193347737315017] atol = 1e-10
    @test Matrix(oas(XA)) ≈ [1.0446928026739652 -0.09945060301076389 -0.0913084483783037 0.20215806787451182;
                             -0.09945060301076389 0.9756589630401777 -0.09212266384154968 -0.11550227928618544;
                             -0.0913084483783037 -0.09212266384154968 0.9916524810682246 -0.06129879273295037;
                             0.20215806787451182 -0.11550227928618544 -0.06129879273295037 1.0104957532176324] atol = 1e-10
    XB = [2.0 0.5 -1.0 1.5 0.0;
          -0.5 1.0 0.5 -1.5 2.0;
          1.0 -2.0 0.5 0.5 -1.0]
    @test ledoitWolfShrinkage(XB) ≈ 0.3850459032766104 atol = 1e-10
    @test oasShrinkage(XB) ≈ 0.9567736142588255 atol = 1e-10
    @test Matrix(ledoitWolf(XB)) ≈ [1.1964733991472363 -0.5104119002804134 0.22753301578765417;
                                    -0.5104119002804134 1.393258710098721 -0.5780568509199863;
                                    0.22753301578765417 -0.5780568509199863 1.270267890754043] atol = 1e-10
    XC = [0.3 1.1 -0.8 0.6;
          -1.2 0.4 0.9 -0.1;
          1.5 -0.6 0.2 1.0;
          0.7 0.7 -1.4 0.5;
          -0.9 1.3 0.8 -1.1] # p = 5 > n = 4: S is singular.
    @test ledoitWolfShrinkage(XC) ≈ 0.4696509105739964 atol = 1e-10
    @test oasShrinkage(XC) ≈ 0.8991967509906638 atol = 1e-10

    # Structural properties on seeded random data, including p > n:
    # Symmetric PSD outputs, intensities in [0, 1], exact trace
    # preservation, and consistency of the estimator with its intensity.
    rng = MersenneTwister(20261030)
    for (p, n) in ((40, 120), (120, 40), (30, 30))
        X = randn(rng, p, n)
        Xc = X .- mean(X; dims=2)
        S = (Xc * Xc') / n
        m = tr(S) / p
        for (est, int) in ((ledoitWolf, ledoitWolfShrinkage), (oas, oasShrinkage))
            Σ̂ = est(X)
            ρ = int(X)
            @test Σ̂ isa Symmetric
            @test 0 <= ρ <= 1
            @test eigmin(Σ̂) > 0 # rho > 0 here, so the estimate is PD even for p > n
            @test tr(Σ̂) ≈ tr(S) atol = 1e-8
            @test Matrix(Σ̂) ≈ (1 - ρ) * S + ρ * m * I atol = 1e-10
        end
    end

    # Identity population (Sigma = I): the target IS the population
    # covariance, so both intensities are near 1 and the Frobenius loss
    # against I collapses relative to the sample covariance.
    p, n = 100, 100
    X = randn(rng, p, n)
    Xc = X .- mean(X; dims=2)
    S = (Xc * Xc') / n
    @test ledoitWolfShrinkage(X) > 0.9
    @test oasShrinkage(X) > 0.9
    @test sum(abs2, Matrix(ledoitWolf(X)) - I) < 0.01 * sum(abs2, S - I)
    @test sum(abs2, Matrix(oas(X)) - I) < 0.01 * sum(abs2, S - I)

    # Spread diagonal population: both estimators beat the sample
    # covariance in Frobenius loss, with moderate intensities.
    d = vcat([8.0, 4.0, 2.0], range(0.4, 1.6; length=47))
    Σ = Diagonal(d)
    X2 = Diagonal(sqrt.(d)) * randn(rng, length(d), 200)
    X2c = X2 .- mean(X2; dims=2)
    S2 = (X2c * X2c') / 200
    loss(M) = sum(abs2, M - Σ)
    @test 0.1 < ledoitWolfShrinkage(X2) < 0.4 # 0.200 at this seed
    @test 0.1 < oasShrinkage(X2) < 0.4 # 0.196 at this seed
    @test loss(Matrix(ledoitWolf(X2))) < loss(S2)
    @test loss(Matrix(oas(X2))) < loss(S2)

    # Ledoit-Wolf (2003) constant-correlation target. Reference
    # intensities and matrices from an independent implementation of
    # the papers' formulas (the pi / rho / gamma sums written out
    # entrywise), separate from the package code; scikit-learn has
    # no constant-correlation variant to compare against.
    @test ledoitWolfShrinkage(XA; target=:constant_correlation) ≈ 0.2614421011797 atol = 1e-10
    @test ledoitWolfShrinkage(XB; target=:constant_correlation) ≈ 0.2489178887215023 atol = 1e-10
    @test ledoitWolfShrinkage(XC; target=:constant_correlation) ≈ 0.40189467737310935 atol = 1e-10
    Σcc_A = [1.1922222222222223 -0.411847911115839 -0.385771629132592 0.6464581529422336;
        -0.411847911115839 0.8625 -0.3791272100991172 -0.4641360306085983;
        -0.385771629132592 -0.3791272100991172 0.9388888888888888 -0.275389129435672;
        0.6464581529422336 -0.4641360306085983 -0.275389129435672 1.0288888888888889]
    Σcc_B = [1.14 -0.7334061130731484 0.1757046021364959;
        -0.7334061130731484 1.46 -0.8216702000857754;
        0.1757046021364959 -0.8216702000857754 1.26]
    @test Matrix(ledoitWolf(XA; target=:constant_correlation)) ≈ Σcc_A atol = 1e-10
    @test Matrix(ledoitWolf(XB; target=:constant_correlation)) ≈ Σcc_B atol = 1e-10
    # The default target is the identity target, unchanged.
    @test ledoitWolf(XA) == ledoitWolf(XA; target=:identity)
    @test ledoitWolfShrinkage(XA) == ledoitWolfShrinkage(XA; target=:identity)

    # Structural properties on seeded random data with varied scales,
    # including p > n: Symmetric PD output, intensity in [0, 1], the
    # estimator equals (1 - rho) S + rho F with F rebuilt from S, and
    # the diagonal of S is preserved (the target shares it).
    rngcc = MersenneTwister(20261031)
    for (p, n) in ((40, 120), (120, 40), (30, 30))
        X = randn(rngcc, p, n) .* (0.5 .+ rand(rngcc, p))
        Xc = X .- mean(X; dims=2)
        S = (Xc * Xc') / n
        v = diag(S)
        sd = sqrt.(v)
        R = S ./ (sd * sd')
        rbar = (sum(R) - p) / (p * (p - 1))
        F = rbar .* (sd * sd')
        for i in 1:p
            F[i, i] = v[i]
        end
        Σ̂ = ledoitWolf(X; target=:constant_correlation)
        ρ = ledoitWolfShrinkage(X; target=:constant_correlation)
        @test Σ̂ isa Symmetric
        @test 0 <= ρ <= 1
        @test eigmin(Σ̂) > 0 # rho > 0 here, so PD even for p > n
        @test Matrix(Σ̂) ≈ (1 - ρ) .* S .+ ρ .* F atol = 1e-10
        @test diag(Σ̂) ≈ v atol = 1e-12
    end

    # Equicorrelated population (correlation 0.5, varied scales): the
    # target family contains the population, so the estimate beats
    # the sample covariance in Frobenius loss (same protocol as the
    # identity-population check above).
    p, n = 60, 300
    d = collect(range(0.5, 2.0; length=p))
    Σeq = Diagonal(d) * (fill(0.5, p, p) + Diagonal(fill(0.5, p))) * Diagonal(d)
    Xeq = cholesky(Σeq).L * randn(rng, p, n)
    Xeqc = Xeq .- mean(Xeq; dims=2)
    Seq = (Xeqc * Xeqc') / n
    @test sum(abs2, Matrix(ledoitWolf(Xeq; target=:constant_correlation)) - Σeq) <
        sum(abs2, Seq - Σeq)

    # Degenerate and invalid inputs.
    X0 = fill(2.5, 4, 7) # constant data: S = 0, the target itself
    @test Matrix(ledoitWolf(X0)) == zeros(4, 4)
    @test Matrix(oas(X0)) == zeros(4, 4)
    @test ledoitWolfShrinkage(X0) == 1.0 # degenerate convention (see docstring)
    @test oasShrinkage(X0) == 1.0
    @test ledoitWolf([1 2 3; 4 5 6; 7 8 10]) isa Symmetric{Float64} # integer input
    @test_throws ArgumentError ledoitWolf(randn(rng, 5, 1)) # n = 1
    @test_throws ArgumentError oas(reshape([1.0], 1, 1))
    @test_throws ArgumentError ledoitWolfShrinkage(zeros(0, 4)) # no variables
    @test_throws ArgumentError ledoitWolf([1.0 NaN; 2.0 3.0; 4.0 5.0])
    @test_throws ArgumentError oasShrinkage([1.0 Inf; 2.0 3.0; 4.0 5.0])

    # Degenerate and invalid inputs for the constant-correlation
    # target specifically.
    @test Matrix(ledoitWolf(X0; target=:constant_correlation)) == zeros(4, 4)
    @test ledoitWolfShrinkage(X0; target=:constant_correlation) == 1.0 # gamma = 0 convention
    @test Matrix(ledoitWolf(reshape(1.0:5.0, 1, 5); target=:constant_correlation)) ==
        fill(2.0, 1, 1) # p = 1: S = F, the estimate is S itself
    @test ledoitWolfShrinkage(reshape(1.0:5.0, 1, 5); target=:constant_correlation) == 1.0
    @test ledoitWolf([1 2 4; 3 5 4]; target=:constant_correlation) isa Symmetric{Float64}
    @test_throws ArgumentError ledoitWolf(XA; target=:bogus) # unknown target
    @test_throws ArgumentError ledoitWolfShrinkage(XA; target=:bogus)
    @test_throws ArgumentError ledoitWolf(XA[:, 1:1]; target=:constant_correlation) # n = 1
    @test_throws ArgumentError ledoitWolf(fill(NaN, 3, 4); target=:constant_correlation)
end

@testset "Tracy-Widom" begin
    using Distributions: cdf, pdf, quantile
    @test mean(TracyWidom(1)) ≈ -1.2065335745820
    @test mean(TracyWidom(2)) ≈ -1.771086807411
    @test mean(TracyWidom(4)) ≈ -2.306884893241
    @test var(TracyWidom(2)) ≈ 0.8131947928329
    # Published quantile anchors (Johnstone / El Karoui / Bejan tables).
    @test quantile(TracyWidom(1), 0.05) ≈ -3.18 atol = 0.02
    @test quantile(TracyWidom(1), 0.95) ≈ 0.98 atol = 0.02
    @test quantile(TracyWidom(2), 0.5) ≈ -1.81 atol = 0.02
    @test quantile(TracyWidom(2), 0.95) ≈ -0.23 atol = 0.02
    @test quantile(TracyWidom(4), 0.05) ≈ -3.4556 atol = 0.01
    @test quantile(TracyWidom(4), 0.95) ≈ -1.0904 atol = 0.01
    @test quantile(TracyWidom(4), 0.99) ≈ -0.5447 atol = 0.01
    d2 = TracyWidom(2)
    @test cdf(d2, quantile(d2, 0.3)) ≈ 0.3 atol = 1e-3
    xs = -6:0.005:3
    @test sum(pdf.(d2, xs)) * 0.005 ≈ 1.0 atol = 0.01
    @test_throws ArgumentError cdf(TracyWidom(3), 0.0)
    @test isfinite(rand(TracyWidom(3)))
    μ, σ = twScaling(400, 200)
    @test μ ≈ (sqrt(399) + sqrt(200))^2 / 400
    @test σ > 0
    # Johnstone's Tracy-Widom test for the largest eigenvalue:
    # deterministic anchors from the TW quantiles.
    μt, σt = twScaling(400, 100)
    r = twTest(μt + σt * quantile(TracyWidom(1), 0.95), 400, 100)
    @test keys(r) == (:statistic, :pvalue)
    @test r.statistic ≈ quantile(TracyWidom(1), 0.95)
    @test r.pvalue ≈ 0.05 atol = 2e-3
    @test twTest(μt, 400, 100).pvalue ≈ 1 - cdf(TracyWidom(1), 0.0)
    r2 = twTest(μt + σt * quantile(TracyWidom(2), 0.95), 400, 100; beta=2)
    @test r2.statistic ≈ quantile(TracyWidom(2), 0.95)
    @test r2.pvalue ≈ 0.05 atol = 2e-3
    # The p-value decreases as lambda grows and stays in [0, 1].
    ps = [twTest(λ, 400, 100).pvalue for λ in (2.1, 2.246, 2.3, 2.5)]
    @test issorted(ps; rev=true) && all(p -> 0 <= p <= 1, ps)
    @test_throws ArgumentError twTest(2.3, 400, 100; beta=4)
    @test_throws ArgumentError twTest(2.3, 400, 100; beta=3)
    @test_throws ArgumentError twTest(2.3, 1, 100)
    @test_throws ArgumentError twTest(2.3, 400, 0)
    @test_throws ArgumentError twTest(NaN, 400, 100)
    @test_throws ArgumentError twTest(Inf, 400, 100)
    # Simulation calibration under the null (identity population
    # covariance): the scaled statistic follows TW_1 / TW_2 and the
    # test rejects at about the nominal rate — slightly conservatively
    # at these sizes (a 400-trial calibration at (400, 100) gave mean
    # statistic -1.24 against the TW_1 mean -1.207, rejection 0.033 at
    # the 5% level, mean p-value 0.504).
    rngtw = MersenneTwister(20261101)
    zs = map(1:250) do _
        X = randn(rngtw, 400, 100)
        twTest(eigmax(Symmetric(X' * X) ./ 400), 400, 100).statistic
    end
    @test -1.5 < mean(zs) < -0.9
    @test 0.005 < mean(z -> z > quantile(TracyWidom(1), 0.95), zs) < 0.11
    @test 0.38 < mean(1 .- cdf.(TracyWidom(1), zs)) < 0.62
    zc = map(1:150) do _
        X = randn(rngtw, ComplexF64, 400, 100)
        twTest(eigmax(Hermitian(X' * X) ./ 400), 400, 100; beta=2).statistic
    end
    @test -2.1 < mean(zc) < -1.45
    # The beta = 4 branch of the DE sampler: the raw rescaled edge is
    # the stochastic-Airy variable (mean -2.05520, var 0.41092), and
    # the sampler's 2^(1/6) multiplication must land it on the classic
    # TracyWidom(4) moments. Seeded; bands carry the finite-N bias at
    # N = 1000 (~+0.01-0.03 on the mean) and the Monte Carlo error
    # (se ~ 0.011 at 4000 reps). The wrong direction would give
    # mean ~ -1.83 and var ~ 0.26, far outside both bands.
    rng4 = Xoshiro(20261106)
    xs4 = [RandomMatrix._tw_tridiagonal_sample(rng4, 4; N=1000) for _ in 1:4000]
    @test -2.306884893241 - 0.08 < mean(xs4) < -2.306884893241 + 0.08
    @test 0.5177237207726 - 0.10 < var(xs4) < 0.5177237207726 + 0.10
end

@testset "MANOVA Tracy-Widom test" begin
    using Distributions: cdf, quantile
    # Real scaling recomputed independently from Johnstone's eq. (6).
    γa = 2 * asin(sqrt((min(50, 50) - 1 / 2) / (100 + 50 - 1)))
    φa = 2 * asin(sqrt((max(50, 50) - 1 / 2) / (100 + 50 - 1)))
    μr = 2 * log(tan((φa + γa) / 2))
    σr = (16 / (100 + 50 - 1)^2 / (sin(φa + γa)^2 * sin(φa) * sin(γa)))^(1 / 3)
    @test manovaTWScaling(50, 100, 50) == (μr, σr)
    # Complex scaling: the Theorem 2 weighted constants, pinned at the
    # values of the calibration prototype (an independent implementation
    # of the same formulas, validated by simulation below and in the
    # 2026-11-09 calibration: 1500 reps at three configurations gave
    # z mean -1.849 / -1.764 / -1.785 against the TW_2 mean -1.771).
    μc, σc = manovaTWScaling(50, 100, 50; beta=2)
    @test μc ≈ 2.079395707874801 atol = 1e-12
    @test σc ≈ 0.12651634469655962 atol = 1e-12
    # Deterministic anchors from the TW quantiles, both betas.
    invlogit(w) = 1 / (1 + exp(-w))
    for (beta, μ, σ) in ((1, μr, σr), (2, μc, σc))
        q95 = quantile(TracyWidom(beta), 0.95)
        r = manovaTWTest(invlogit(μ + σ * q95), 50, 100, 50; beta=beta)
        @test keys(r) == (:statistic, :pvalue)
        @test r.statistic ≈ q95
        @test r.pvalue ≈ 0.05 atol = 2e-3
        @test manovaTWTest(invlogit(μ), 50, 100, 50; beta=beta).pvalue ≈
              1 - cdf(TracyWidom(beta), 0.0)
    end
    # The p-value decreases as theta grows and stays in [0, 1].
    ps = [manovaTWTest(θ, 50, 100, 50).pvalue for θ in (0.85, 0.88, 0.90, 0.92)]
    @test issorted(ps; rev=true) && all(p -> 0 <= p <= 1, ps)
    @test_throws ArgumentError manovaTWTest(0.9, 50, 100, 50; beta=4)
    @test_throws ArgumentError manovaTWTest(0.9, 50, 100, 50; beta=3)
    @test_throws ArgumentError manovaTWTest(0.0, 50, 100, 50)
    @test_throws ArgumentError manovaTWTest(1.0, 50, 100, 50)
    @test_throws ArgumentError manovaTWTest(1.2, 50, 100, 50)
    @test_throws ArgumentError manovaTWTest(NaN, 50, 100, 50)
    @test_throws ArgumentError manovaTWTest(0.9, 1, 100, 50)
    @test_throws ArgumentError manovaTWTest(0.9, 50, 1, 50)
    @test_throws ArgumentError manovaTWTest(0.9, 50, 100, 1)
    # Angle fractions leave (0, 1) when max(p, n) >= m + n - 1/2.
    @test_throws ArgumentError manovaTWTest(0.5, 100, 10, 10)
    @test_throws ArgumentError manovaTWTest(0.5, 100, 10, 10; beta=2)
    # Simulation calibration under the null at (p, m, n) = (50, 100, 50):
    # the scaled statistic follows TW_1 / TW_2 and the test rejects at
    # about the nominal rate (calibration prototype, 2000 / 1500 reps:
    # real mean -1.145, rejection 0.047; complex mean -1.764, rejection
    # 0.047 — bands here carry the smaller in-suite rep counts).
    function manova_theta_max(rng, p, m, n; complex::Bool=false)
        if complex
            X = (randn(rng, m, p) .+ im .* randn(rng, m, p)) ./ sqrt(2)
            Y = (randn(rng, n, p) .+ im .* randn(rng, n, p)) ./ sqrt(2)
            A, B = Hermitian(X' * X), Hermitian(Y' * Y)
            C = cholesky(Hermitian(Matrix(A + B)))
            return eigmax(Hermitian(C.L \ Matrix(B) / C.L'))
        end
        X = randn(rng, m, p); Y = randn(rng, n, p)
        A, B = Symmetric(X' * X), Symmetric(Y' * Y)
        C = cholesky(Symmetric(Matrix(A + B)))
        return eigmax(Symmetric(C.L \ Matrix(B) / C.L'))
    end
    rngm = MersenneTwister(20261109)
    zs = [manovaTWTest(manova_theta_max(rngm, 50, 100, 50), 50, 100, 50).statistic
          for _ in 1:250]
    @test -1.55 < mean(zs) < -0.85
    @test 0.005 < mean(z -> z > quantile(TracyWidom(1), 0.95), zs) < 0.11
    zc = [manovaTWTest(manova_theta_max(rngm, 50, 100, 50; complex=true),
                        50, 100, 50; beta=2).statistic for _ in 1:200]
    @test -2.05 < mean(zc) < -1.5
end

@testset "Beta ensembles" begin
    using Distributions: insupport
    # API shape: multivariate Distributions types whose rand returns
    # the sorted spectrum.
    dh = BetaHermite(2, 40)
    @test length(dh) == 40
    λ0 = rand(MersenneTwister(7), dh)
    @test length(λ0) == 40 && issorted(λ0)
    @test size(rand(MersenneTwister(7), dh, 3)) == (40, 3)
    @test rand(MersenneTwister(7), dh) == λ0   # seeded reproducibility
    @test mean(dh) == zeros(40)
    @test insupport(dh, λ0)
    @test !insupport(dh, reverse(λ0))
    dl = BetaLaguerre(1, 60, 30)
    @test length(dl) == 30
    μ0 = rand(MersenneTwister(7), dl)
    @test issorted(μ0) && minimum(μ0) >= 0
    @test insupport(dl, μ0)
    # Validation.
    @test_throws ArgumentError BetaHermite(0, 5)
    @test_throws ArgumentError BetaHermite(-2, 5)
    @test_throws ArgumentError BetaHermite(NaN, 5)
    @test_throws ArgumentError BetaHermite(2, 0)
    @test_throws ArgumentError BetaLaguerre(0, 10, 5)
    @test_throws ArgumentError BetaLaguerre(1, 10, 0)
    @test_throws ArgumentError BetaLaguerre(1, 0, 5)
    # Degenerate sizes: n = 1 Hermite is a single normal; p = 1
    # Laguerre is a scaled chi-square with E[lam] = n exactly.
    @test length(rand(MersenneTwister(3), BetaHermite(2, 1))) == 1
    rng1 = MersenneTwister(20261108)
    @test 44 < mean(rand(rng1, BetaLaguerre(2, 50, 1))[1] for _ in 1:400) < 56
    # Simulation calibration (bands fixed from a 2000/400-rep
    # calibration at N = 200, n = 400, p = 100: trace fingerprints
    # matched the exact identities to 1e-3, bulk moments to ~0.01,
    # edge means to ~0.09 at beta = 1, 2).
    #
    # Hermite trace identity E[sum lam^2] = 2N/beta + N(N-1) is exact,
    # and the scaled bulk moments tend to the semicircle values 1, 2.
    rngh = MersenneTwister(20261109)
    for β in (0.5, 1.0, 2.0, 4.0, 3.7)
        N = 120
        specs = [rand(rngh, BetaHermite(β, N)) for _ in 1:300]
        exact = 2N / β + N * (N - 1)
        @test 0.99 < mean(sum(abs2, λ) for λ in specs) / exact < 1.01
        @test 0.94 < mean(mean(abs2, λ) for λ in specs) / N < 1.06
        @test 1.8 < mean(mean(λ .^ 4) for λ in specs) / N^2 < 2.2
    end
    # Hermite edge vs TracyWidom at beta = 1, 2 (calibration means
    # -1.296, -1.762 at N = 200; the bands carry the finite-N bias).
    rnge = MersenneTwister(20261110)
    for (β, lo, hi) in ((1.0, -1.55, -0.9), (2.0, -1.98, -1.58))
        w = [200^(1 / 6) * (maximum(rand(rnge, BetaHermite(β, 200))) - 2 * sqrt(200))
             for _ in 1:300]
        @test lo < mean(w) < hi
    end
    # Laguerre trace identity E[sum lam] = n p is exact; on the 1/n
    # scale the bulk tends to MarchenkoPastur(p/n) (mean 1, second
    # moment 1 + p/n) and the lower edge to (1 - sqrt(p/n))^2.
    rngl = MersenneTwister(20261111)
    for β in (0.5, 1.0, 2.0, 4.0, 3.7)
        specs = [rand(rngl, BetaLaguerre(β, 200, 50)) for _ in 1:300]
        @test 0.99 < mean(sum, specs) / (200 * 50) < 1.01
        @test 0.97 < mean(mean(μ) for μ in specs) / 200 < 1.03
        @test 1.18 < mean(mean(abs2, μ) for μ in specs) / 200^2 < 1.32
        @test 0.15 < mean(minimum(μ) for μ in specs) / 200 < 0.38
    end
    # Singular Laguerre, n < p: the spectrum is p - n exact zeros
    # plus the n positive eigenvalues of the dual (Gram-swapped)
    # ensemble, so the trace identity E[sum lam] = n p still holds
    # and the positive part carries the dual's bulk moments
    # (MarchenkoPastur(n/p) on the 1/p scale; same bands as the
    # (200, 50) block above, whose draws the positive part equals
    # in law).
    dls = BetaLaguerre(1, 30, 60)
    @test length(dls) == 60
    μs0 = rand(MersenneTwister(7), dls)
    @test length(μs0) == 60 && issorted(μs0)
    @test count(==(0.0), μs0) == 30 && all(>(0), μs0[31:end])
    @test insupport(dls, μs0)
    @test size(rand(MersenneTwister(7), dls, 3)) == (60, 3)
    @test rand(MersenneTwister(7), dls) == μs0   # seeded reproducibility
    # The positive part is exactly a draw of the swapped ensemble:
    # same bidiagonal builder, same RNG stream.
    @test μs0[31:end] == rand(MersenneTwister(7), BetaLaguerre(1, 60, 30))
    # Rank-one edge: n = 1 leaves a single positive eigenvalue.
    μ1 = rand(MersenneTwister(3), BetaLaguerre(2, 1, 5))
    @test count(==(0.0), μ1) == 4 && μ1[end] > 0
    rngs = MersenneTwister(20261113)
    for β in (0.5, 1.0, 2.0, 4.0, 3.7)
        specs = [rand(rngs, BetaLaguerre(β, 50, 200)) for _ in 1:300]
        @test all(s -> count(==(0.0), s) == 150, specs)
        @test 0.99 < mean(sum, specs) / (50 * 200) < 1.01
        pos = [s[151:end] for s in specs]
        @test 0.97 < mean(mean, pos) / 200 < 1.03
        @test 1.18 < mean(mean(abs2, μ) for μ in pos) / 200^2 < 1.32
    end
    # Laguerre edge vs Johnstone scaling at beta = 1 (calibration
    # mean -1.224 at (400, 100), 400 reps).
    μt, σt = twScaling(400, 100)
    rngw = MersenneTwister(20261112)
    z = [(maximum(rand(rngw, BetaLaguerre(1, 400, 100))) / 400 - μt) / σt
         for _ in 1:150]
    @test -1.55 < mean(z) < -0.9
end

@testset "Free probability" begin
    sc = semicircleLaw(1.0)
    @test freeMean(sc) ≈ 0.0 atol = 1e-10
    @test freeVar(sc) ≈ 1.0 atol = 0.02
    @test rTransform(sc, 0.1im) ≈ 0.1im atol = 5e-4
    fp = freePoissonLaw(0.5)
    @test freeMean(fp) ≈ 1.0 atol = 0.01
    @test freeVar(fp) ≈ 0.5 atol = 0.02
    # The Chebyshev discretization resolves the lower-edge density
    # spike for gamma near 1 (uniform grid: mean 2.3% high at 0.99,
    # all-NaN weights at gamma = 1, where pdf(MarchenkoPastur(1), 0)
    # is NaN).
    for g in (0.99, 1.0, 1.01, 2.0)
        fpg = freePoissonLaw(g)
        @test all(isfinite, fpg.weights)
        @test freeMean(fpg) ≈ 1.0 atol = 2e-3
        @test freeVar(fpg) ≈ g atol = 2e-3
    end
    @test freeMean(freePoissonLaw(1.0; scale=2.0)) ≈ 2.0 atol = 4e-3
    @test freeVar(freePoissonLaw(1.0; scale=2.0)) ≈ 4.0 atol = 8e-3
    @test rTransform(fp, 0.1) ≈ 1 / (1 - 0.05) atol = 1e-3
    @test sTransform(fp, 0.3) ≈ 1 / (1 + 0.5 * 0.3) atol = 1e-3
    # Semicircle boxplus semicircle: variances add.
    ss = freeAdd(semicircleLaw(1.0), semicircleLaw(1.0))
    @test freeMean(ss) ≈ 0.0 atol = 1e-8
    @test freeVar(ss) ≈ 2.0 atol = 0.05
    # Atom rule: MP_3 boxplus MP_3 has an atom of mass 1/3 at 0.
    aa = freeAdd(freePoissonLaw(3.0), freePoissonLaw(3.0))
    @test length(freeAtoms(aa)) == 1 && freeAtoms(aa)[1][1] == 0.0 && freeAtoms(aa)[1][2] ≈ 1 / 3
    # Shift by a Dirac: mean shifts, variance unchanged.
    sh = freeAdd(fp, diracLaw(1.0))
    @test freeMean(sh) ≈ 2.0 atol = 0.01
    @test freeVar(sh) ≈ 0.5 atol = 0.02
end

@testset "Free multiplicative convolution" begin
    mm = freeMul(freePoissonLaw(0.5; npoints=401), freePoissonLaw(0.25; npoints=401); npoints=401)
    @test freeMean(mm) ≈ 1.0 atol = 0.05
    @test freeVar(mm) ≈ 0.75 atol = 0.08   # m2 = 1 + gamma + delta
    # A Dirac factor dilates exactly.
    dil = freeMul(diracLaw(2.0), freePoissonLaw(0.5; npoints=401))
    @test freeMean(dil) ≈ 2.0 atol = 0.01
    @test freeVar(dil) ≈ 2.0 atol = 0.02
    # Atom at 0 under boxtimes: max of the two masses.
    at = freeMul(freePoissonLaw(3.0; npoints=401), freePoissonLaw(0.5; npoints=401); npoints=401)
    @test length(freeAtoms(at)) == 1 && freeAtoms(at)[1][1] == 0.0 && freeAtoms(at)[1][2] ≈ 2 / 3
    @test_throws ArgumentError freeMul(semicircleLaw(1.0), freePoissonLaw(0.5))
end

@testset "Free multiplicative convolution (exact moment anchors)" begin
    # Exact moments of MP_g boxtimes MP_d from the S-transform
    # (series reversion of chi(w) = w / ((1+w)(1+g w)(1+d w))):
    # m1 = 1, m2 = 1 + g + d, m3 as tabulated. Tolerances are about
    # 3x the errors measured at the default resolution (2026-10-28):
    # (0.5, 0.25) errs 0.5/0.9/1.5%, (1, 0.5) errs 5.4/6.0/7.4% — the
    # latter is the eps^(1/3) boundary-leak regime documented in
    # freeMul's docstring.
    mom(μ, k) = sum(w * x^k for (x, w) in zip(μ.nodes, μ.weights))
    a1 = freeMul(freePoissonLaw(0.5), freePoissonLaw(0.25))
    @test mom(a1, 1) ≈ 1.0 atol = 0.02
    @test mom(a1, 2) ≈ 1.75 atol = 0.05
    @test mom(a1, 3) ≈ 3.9375 atol = 0.18
    a2 = freeMul(freePoissonLaw(1.0), freePoissonLaw(0.5))
    @test mom(a2, 1) ≈ 1.0 atol = 0.1
    @test mom(a2, 2) ≈ 2.5 atol = 0.25
    @test mom(a2, 3) ≈ 8.25 atol = 0.85
    # MP_1 boxtimes MP_1 (the x^(-2/3) singularity case): regression
    # guard — finite weights and a mean in the documented band.
    a3 = freeMul(freePoissonLaw(1.0), freePoissonLaw(1.0))
    @test all(isfinite, a3.weights)
    @test mom(a3, 1) ≈ 1.0 atol = 0.15
end

@testset "Free compression" begin
    mom(μ, k) = sum(w * x^k for (x, w) in zip(μ.nodes, μ.weights))
    mp(k, γ) = k == 1 ? 1.0 : k == 2 ? 1 + γ : 1 + 3γ + γ^2
    fp = freePoissonLaw(0.5)
    # Validation and the two exact special cases.
    @test_throws ArgumentError freeCompress(fp, 0.0)
    @test_throws ArgumentError freeCompress(fp, -0.5)
    @test_throws ArgumentError freeCompress(fp, 1.5)
    id = freeCompress(fp, 1.0)
    @test id.nodes == fp.nodes && id.weights == fp.weights && id.atoms == fp.atoms
    d = freeCompress(diracLaw(2.0), 0.3)
    @test freeMean(d) == 2.0 && freeVar(d) == 0.0 && freeAtoms(d) == [2.0 => 1.0]
    # Headline anchor: MP_gamma compressed by tau is MP_{tau*gamma}
    # (R-transform identity R_{mu_tau}(w) = R_mu(tau*w); cumulants
    # scale as kappa_n -> tau^(n-1) kappa_n). Tolerances are about 3x
    # the errors measured at the default resolution (2026-11-06):
    # atom-free cases err <= 0.9% on m3, atom cases <= 1.3%.
    for (γ, τ, tol) in ((0.5, 0.5, (0.005, 0.01, 0.03)),
                        (0.5, 0.25, (0.005, 0.01, 0.03)),
                        (0.25, 0.8, (0.005, 0.01, 0.02)),
                        (1.0, 0.5, (0.01, 0.02, 0.05)),
                        (2.0, 0.75, (0.01, 0.03, 0.15)),
                        (3.0, 0.5, (0.01, 0.03, 0.3)))
        c = freeCompress(freePoissonLaw(γ), τ)
        for k in 1:3
            @test mom(c, k) ≈ mp(k, τ * γ) atol = tol[k]
        end
        # ... and it agrees with a freshly built MP_{tau*gamma} law.
        ref = freePoissonLaw(τ * γ)
        @test freeMean(c) ≈ freeMean(ref) atol = tol[1]
        @test freeVar(c) ≈ freeVar(ref) atol = tol[2]
    end
    # Atom at 0: MP_2 (mass 1/2) compressed by 3/4 keeps an atom of
    # (1/2 - 1/4)/(3/4) = 1/3; compressed by 1/2 the atom disappears
    # exactly at tau*gamma = 1.
    at1 = freeAtoms(freeCompress(freePoissonLaw(2.0), 0.75))
    @test length(at1) == 1 && at1[1][1] == 0.0 && at1[1][2] ≈ 1 / 3
    @test isempty(freeAtoms(freeCompress(freePoissonLaw(2.0), 0.5)))
    at2 = freeAtoms(freeCompress(freePoissonLaw(3.0), 0.5))
    @test length(at2) == 1 && at2[1][1] == 0.0 && at2[1][2] ≈ 1 / 3
    # Atoms away from 0 follow the same threshold rule, and the mean
    # is preserved exactly (kappa_1 is not scaled).
    cm = freeCompress(discreteLaw([1.0, 3.0], [0.7, 0.3]), 0.75)
    atm = freeAtoms(cm)
    @test length(atm) == 2 && atm[1][1] == 1.0 && atm[1][2] ≈ 0.6 &&
          atm[2][1] == 3.0 && atm[2][2] ≈ 1 / 15
    @test freeMean(cm) ≈ 1.6 atol = 0.01
    cz = freeCompress(discreteLaw([0.0, 2.0], [0.8, 0.2]), 0.5)
    atz = freeAtoms(cz)
    @test length(atz) == 1 && atz[1][1] == 0.0 && atz[1][2] ≈ 0.6
    @test freeMean(cz) ≈ 0.4 atol = 0.01
    # Semicircle: variance sigma^2 -> tau * sigma^2.
    cs = freeCompress(semicircleLaw(2.0), 0.5)
    @test freeMean(cs) ≈ 0.0 atol = 1e-8
    @test freeVar(cs) ≈ 2.0 atol = 0.06
    # Bernoulli (+-1): no atoms survive at tau = 1/2, and the moments
    # follow cumulant scaling (m2 = tau, m4 = 2 tau^2 - tau^3).
    cb = freeCompress(discreteLaw([-1.0, 1.0], [0.5, 0.5]), 0.5)
    @test isempty(freeAtoms(cb))
    @test mom(cb, 2) ≈ 0.5 atol = 0.01
    @test mom(cb, 4) ≈ 0.375 atol = 0.01
    # Compression commutes with dilation: MP_0.5 at scale 2,
    # compressed by 1/2, is MP_0.25 at scale 2.
    cd = freeCompress(freePoissonLaw(0.5; scale=2.0), 0.5)
    @test freeMean(cd) ≈ 2.0 atol = 0.01
    @test freeVar(cd) ≈ 1.0 atol = 0.03
    # Simulation anchor: the ESD moments of a principal submatrix
    # retaining half the coordinates of a Wishart matrix (gamma = 1/2)
    # track the compressed law MP_{1/4}. Seeded; the band is ~5x the
    # Monte Carlo sd of the 10-rep mean measured at these sizes.
    Random.seed!(20261106)
    p, N, reps = 400, 800, 10
    msub = div(p, 2)
    emp = zeros(reps, 3)
    for r in 1:reps
        X = randn(p, N) ./ sqrt(N)
        S = X * X'
        ev = eigvals(Symmetric(S[1:msub, 1:msub]))
        for k in 1:3
            emp[r, k] = mean(ev .^ k)
        end
    end
    cc = freeCompress(freePoissonLaw(0.5), 0.5)
    for k in 1:3
        @test mean(emp[:, k]) ≈ mom(cc, k) atol = (0.02, 0.03, 0.06)[k]
    end
end

@testset "Silverstein equation (general-population law)" begin
    # Anchors calibrated before the bands below were set
    # (hidden_files/silverstein_proto.jl + silverstein_calib.jl):
    # H = delta_1 reproduces MarchenkoPastur (edges exact, density to
    # ~1e-3 interior), the closed-form moments hold to <= 0.3% at the
    # default resolution across all population shapes below (0.9% for
    # the delta_1 mean at gamma = 1, whose density diverges at 0), and the
    # two-mass support splitting matches formed-matrix simulation.
    # H = delta_1 is MarchenkoPastur: moments gamma-exact.
    for γ in (0.25, 0.5, 1.0, 2.0)
        F = silversteinLaw([1.0], [1.0], γ)
        @test freeMean(F) ≈ 1.0 atol = 0.02
        @test freeVar(F) ≈ γ atol = 0.03 * γ + 0.01
    end
    # Density against the closed form (freeDensity's own smoothing
    # included; the largest deviation sits at the lower edge).
    F = silversteinLaw([1.0], [1.0], 0.5)
    for x in (0.5, 1.0, 1.5, 2.0, 2.5)
        @test freeDensity(F, [x])[1] ≈ pdf(MarchenkoPastur(0.5), x) atol = 0.008
    end
    # Outside the support the density vanishes (MP edges 0.0858, 2.914).
    @test freeDensity(F, [3.5])[1] < 0.005
    # Exact atom rule: max(H({0}), 1 - 1/gamma).
    @test freeAtoms(silversteinLaw([1.0], [1.0], 2.0)) == [0.0 => 0.5]
    @test isempty(freeAtoms(silversteinLaw([1.0], [1.0], 0.5)))
    @test freeAtoms(silversteinLaw([0.0, 2.0], [0.3, 0.7], 0.5)) == [0.0 => 0.3]
    @test freeAtoms(silversteinLaw([0.0, 2.0], [0.3, 0.7], 3.0)) ==
        [0.0 => 1 - 1 / 3]
    # Closed-form moments for multi-mass populations:
    # m1 = E_H[t], m2 = E_H[t^2] + gamma (E_H[t])^2.
    for (pts, ms, γ) in (([1.0, 4.0], [0.5, 0.5], 0.1),
                         ([1.0, 4.0], [0.5, 0.5], 1.5),
                         ([0.5, 2.0, 5.0], [0.3, 0.4, 0.3], 0.8),
                         ([0.0, 2.0], [0.3, 0.7], 0.5))
        F = silversteinLaw(pts, ms, γ)
        μ1 = sum(ms .* pts)
        μ2 = sum(ms .* pts .^ 2) + γ * μ1^2
        @test freeMean(F) ≈ μ1 atol = 0.04 * μ1
        @test freeVar(F) ≈ μ2 - μ1^2 atol = 0.04 * (μ2 - μ1^2)
    end
    # Two-mass support splitting: at gamma = 0.1 the inverse-map
    # edges are [0.555, 1.408] U [2.485, 6.052] -- the gap midpoint
    # carries only the smoothing tails, the cluster interiors do not.
    F2 = silversteinLaw([1.0, 4.0], [0.5, 0.5], 0.1)
    @test freeDensity(F2, [1.95])[1] < 0.02
    @test freeDensity(F2, [0.9])[1] > 0.3
    @test freeDensity(F2, [4.0])[1] > 0.08
    # Formed-matrix check of the splitting (seeded): no eigenvalues
    # in the predicted gap, extremes at the predicted outer edges.
    Random.seed!(20261113)
    p, γ = 300, 0.1
    n = round(Int, p / γ)
    sq = sqrt.(vcat(fill(1.0, p ÷ 2), fill(4.0, p - p ÷ 2)))
    X = randn(p, n) ./ sqrt(n)
    ev = eigvals(Symmetric((sq .* X) * (sq .* X)'))
    @test count(x -> 1.55 < x < 2.35, ev) / p < 0.01
    @test 0.35 < ev[1] < 0.75
    @test 5.6 < ev[end] < 6.5
    # At gamma = 0.5 the same population has a single interval:
    # the former gap is populated.
    @test freeDensity(silversteinLaw([1.0, 4.0], [0.5, 0.5], 0.5), [1.95])[1] > 0.05
    # FreeMeasure entry point: H the Marcenko-Pastur grid law itself
    # (E_H[t] = 1, E_H[t^2] = 1.5), gamma = 0.4: m1 = 1, var = 0.9.
    F3 = silversteinLaw(freePoissonLaw(0.5), 0.4)
    @test freeMean(F3) ≈ 1.0 atol = 0.02
    @test freeVar(F3) ≈ 0.9 atol = 0.03
    # Distributions entry point: H = Gamma(2, 1.5) (mean 3,
    # E[t^2] = 13.5), gamma = 0.5: m1 = 3, var = 18 - 9 = 9. The
    # bands include the quantile-discretization error of the
    # unbounded population (measured 3.006 / 9.006 at hpoints = 401).
    F4 = silversteinLaw(Gamma(2, 1.5), 0.5)
    @test freeMean(F4) ≈ 3.0 atol = 0.1
    @test freeVar(F4) ≈ 9.0 atol = 0.45
    # Degenerate population and validation.
    @test silversteinLaw([0.0], [1.0], 0.5).nodes == [0.0]
    @test_throws ArgumentError silversteinLaw([1.0], [1.0], 0.0)
    @test_throws ArgumentError silversteinLaw([1.0], [1.0], -1.0)
    @test_throws ArgumentError silversteinLaw([-1.0, 2.0], [0.5, 0.5], 0.5)
    @test_throws ArgumentError silversteinLaw([1.0], [-1.0], 0.5)
    @test_throws ArgumentError silversteinLaw(Float64[], Float64[], 0.5)
end

@testset "RIE finite-size bias (characterized)" begin
    # Seeded characterization of the bias documented in rieEigenvalues:
    # at the default broadening eta = p^(-1/2) the mean cleaned eigenvalue
    # under the null is biased up by ~ c(gamma) p^(-1/2) (c(0.5) ~ 1.3,
    # so ~ 1.09 at p = 200, gamma = 0.5); eta = p^(-2/3) cuts the bias
    # roughly threefold. Bands sit several standard errors around the
    # measured values, tight enough to catch a prescription change.
    Random.seed!(20261027)
    p, n = 200, 400
    γ = p / n
    X = randn(p, n) / sqrt(n)
    ev = eigvals(Symmetric(X * X'))
    ξd = rieEigenvalues(ev, γ)
    @test 1.04 < mean(ξd) < 1.15
    ξ23 = rieEigenvalues(ev, γ; η=p^(-2 / 3))
    @test abs(mean(ξ23) - 1) < abs(mean(ξd) - 1) - 0.02
    @test abs(mean(ξ23) - 1) < 0.07
    # The ASCII spelling `eta` is an alias for the Greek keyword `η`.
    @test rieEigenvalues(ev, γ; eta=0.05) == rieEigenvalues(ev, γ; η=0.05)
    @test rieEigenvalues(ev, γ; eta=p^(-2 / 3)) == ξ23
end

@testset "Variance-profile Wigner law (QVE)" begin
    # Anchors calibrated before the bands below were set
    # (hidden_files/qve_proto.jl + qve_calib1-5.jl): the constant
    # profile reproduces the semicircle (grid second moment to
    # 2.5e-5 at sigma = 1; freeDensity carries its own eps = 0.01
    # smoothing, deviations <= 1.6e-3), the block-diagonal mixture is
    # exact with fraction-weighted radii 2 sqrt(b p) (moments to
    # 1e-4), the two-block law matches formed-matrix simulation
    # (pooled sup CDF distance 0.001 at n = 400, edge 2.156 approached
    # from below), and solver residuals are <= 2e-10 across constant,
    # block, band, and fully inhomogeneous profiles.
    # Constant profile = semicircleLaw.
    for σ in (1.0, 1.7)
        mu = profileWignerLaw(fill(σ^2, 30, 30))
        @test freeMean(mu) ≈ 0.0 atol = 1e-8
        @test freeVar(mu) ≈ σ^2 rtol = 5e-4
        for x in (-1.0, 0.0, 0.7)
            @test freeDensity(mu, [x])[1] ≈ pdf(Semicircle(2σ), x) atol = 0.004
        end
        # Outside the support the density vanishes.
        @test freeDensity(mu, [2.5σ])[1] < 0.005
    end
    # Block-diagonal profile: exact mixture, block b on fraction p
    # contributing a semicircle of radius 2 sqrt(b p) with weight p:
    # 0.6 * SC(sqrt(1.35)) + 0.4 * SC(sqrt(0.256)).
    mu = profileWignerLaw([60, 40], [2.25 0.0; 0.0 0.64])
    @test freeMean(mu) ≈ 0.0 atol = 1e-8
    @test freeVar(mu) ≈ 0.6 * 1.35 + 0.4 * 0.256 atol = 5e-4
    mix(x) = 0.6 * pdf(Semicircle(2 * sqrt(1.35)), x) +
             0.4 * pdf(Semicircle(2 * sqrt(0.256)), x)
    for x in (0.0, 0.5, 2.0)
        @test freeDensity(mu, [x])[1] ≈ mix(x) atol = 0.008
    end
    # Block form and matrix form are the same computation.
    Sb = zeros(20, 20)
    Sb[1:12, 1:12] .= 2.0; Sb[13:20, 13:20] .= 1.0
    Sb[1:12, 13:20] .= 0.3; Sb[13:20, 1:12] .= 0.3
    mb = profileWignerLaw([12, 8], [2.0 0.3; 0.3 1.0])
    @test mb.nodes == profileWignerLaw(Sb).nodes
    @test mb.weights == profileWignerLaw(Sb).weights
    # Two-block profile vs formed-matrix simulation (seeded): pooled
    # moments and CDF against the law, largest eigenvalue below the
    # predicted edge 2.156 at the shallow-edge TW scale.
    Random.seed!(20261113)
    mu2 = profileWignerLaw([150, 150], [2.0 0.5; 0.5 1.0])
    @test freeVar(mu2) ≈ 1.0 atol = 5e-4
    S300 = zeros(300, 300)
    S300[1:150, 1:150] .= 2.0; S300[151:300, 151:300] .= 1.0
    S300[1:150, 151:300] .= 0.5; S300[151:300, 1:150] .= 0.5
    allex = Float64[]
    lmax = Float64[]
    for r in 1:15
        ev = eigvals(randProfileWigner(S300))
        append!(allex, ev)
        push!(lmax, ev[end])
    end
    @test abs(mean(allex .^ 2) - 1.0) < 0.012
    supd = 0.0
    for x in range(-2.3, 2.3; length=47)
        ecdf = count(v -> v <= x, allex) / length(allex)
        lcdf = sum(w for (nd, w) in zip(mu2.nodes, mu2.weights) if nd <= x)
        supd = max(supd, abs(ecdf - lcdf))
    end
    @test supd < 0.01
    @test 2.0 < mean(lmax) < 2.2
    # Band profile: second moment is exactly mean(S); the Anderson
    # solver path (200 groups) lands on it.
    n = 200
    Sband = [abs(i - j) <= 20 ? 1.0 : 0.0 for i in 1:n, j in 1:n]
    @test freeVar(profileWignerLaw(Sband)) ≈ sum(Sband) / n^2 atol = 5e-4
    # Fully inhomogeneous profile (Anderson path): moments and sanity.
    Random.seed!(20261114)
    n = 60
    Sr = 0.5 .+ rand(n, n)
    Sr = (Sr + Sr') / 2
    mr = profileWignerLaw(Sr)
    @test freeMean(mr) ≈ 0.0 atol = 1e-6
    @test freeVar(mr) ≈ sum(Sr) / n^2 atol = 5e-4
    @test all(isfinite, mr.nodes) && all(w -> w >= 0, mr.weights)
    # Zero rows: exact atom at 0, rest of the law untouched.
    Sz = fill(1.0, 30, 30)
    Sz[7, :] .= 0.0
    Sz[:, 7] .= 0.0
    mz = profileWignerLaw(Sz)
    @test freeAtoms(mz) == [0.0 => 1 / 30]
    @test freeVar(mz) ≈ sum(Sz) / 900 atol = 5e-4
    @test profileWignerLaw(zeros(4, 4)).nodes == [0.0]
    # Sampler: types, symmetry, and entry variances (seeded).
    @test randProfileWigner(fill(1.0, 40, 40)) isa Symmetric
    @test randProfileWigner(fill(1.0, 40, 40); complex=true) isa Hermitian
    Random.seed!(20261115)
    S60 = zeros(60, 60)
    S60[1:30, 1:30] .= 2.0; S60[31:60, 31:60] .= 1.0
    S60[1:30, 31:60] .= 0.5; S60[31:60, 1:30] .= 0.5
    acc = 0.0
    for r in 1:400
        acc += randProfileWigner(S60)[1, 31]^2
    end
    @test 0.0055 < acc / 400 < 0.0115
    # Validation.
    @test_throws ArgumentError profileWignerLaw(ones(3, 4))
    @test_throws ArgumentError profileWignerLaw([1.0 -0.5; -0.5 1.0])
    @test_throws ArgumentError profileWignerLaw([1.0 0.9; 0.1 1.0])
    @test_throws ArgumentError profileWignerLaw([1.0 NaN; NaN 1.0])
    @test_throws ArgumentError profileWignerLaw([10, 10], [1.0 0.0 0.0])
    @test_throws ArgumentError profileWignerLaw([0, 10], [1.0 0.0; 0.0 1.0])
    @test_throws ArgumentError profileWignerLaw([10, 10], [1.0 -0.5; -0.5 1.0])
    @test_throws ArgumentError randProfileWigner([1.0 -1.0; -1.0 1.0])
end

@testset "Aqua quality checks" begin
    # Ambiguities, unbound type parameters, undefined exports, stale and
    # compat-bounded dependencies, type piracy, persistent tasks, and a
    # docstring for every exported name. Aqua is a test-only dependency
    # (see [extras]/[targets] in Project.toml), never a runtime dep.
    Aqua.test_all(RandomMatrix)
end

include("test_spectralformfactor.jl")

include("test_jacobitwcorrection.jl")
