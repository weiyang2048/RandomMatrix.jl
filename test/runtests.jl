using RandomMatrix
using Distributions: Normal, Semicircle
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

@testset "Aqua quality checks" begin
    # Ambiguities, unbound type parameters, undefined exports, stale and
    # compat-bounded dependencies, type piracy, persistent tasks, and a
    # docstring for every exported name. Aqua is a test-only dependency
    # (see [extras]/[targets] in Project.toml), never a runtime dep.
    Aqua.test_all(RandomMatrix)
end
