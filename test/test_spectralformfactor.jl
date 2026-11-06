@testset "Spectral form factor" begin
    # Exact anchors for g(t) = |sum_j cis(-t E_j)|^2 / N^2.
    @test spectralFormFactor([0.0, 1.0], π) ≈ 0.0 atol = 1e-24
    @test spectralFormFactor(collect(0:4), 2π) ≈ 1.0 atol = 1e-12
    @test spectralFormFactor([1.0, 5.0, 9.0], 0.0) == 1.0
    @test spectralFormFactor([3.7], 2.3) ≈ 1.0 atol = 1e-12
    E = [0.3, 1.1, 2.9, 4.2, 8.0]
    @test spectralFormFactor(E, [0.0, π]) ≈
        [spectralFormFactor(E, 0.0), spectralFormFactor(E, π)]
    ts = collect(range(0.0, 6.0, length=25))
    g = spectralFormFactor(E, ts)
    @test length(g) == length(ts) && all(x -> -1e-12 <= x <= 1 + 1e-12, g)
    # unfold=true must not mutate the caller's vector.
    Eraw = [5.0, 1.0, 3.0, 2.0, 4.0]
    Ecopy = copy(Eraw)
    gu = spectralFormFactor(Eraw, 0.7; unfold=true)
    @test Eraw == Ecopy && 0.0 <= gu <= 1.0
    @test_throws ArgumentError spectralFormFactor(Float64[], 1.0)

    # Ensemble decomposition: identity total = connected + disconnected,
    # and the trivial single-spectrum case (disconnected = total).
    rng = MersenneTwister(20261114)
    specs = [sort(rand(rng, 20)) .* 20 for _ in 1:5]
    parts = ensembleSpectralFormFactor(specs, ts)
    @test parts.total ≈ parts.connected .+ parts.disconnected
    one = ensembleSpectralFormFactor([specs[1]], 1.3)
    @test one.connected == 0.0
    @test one.disconnected ≈ one.total ≈ spectralFormFactor(specs[1], 1.3)
    @test_throws ArgumentError ensembleSpectralFormFactor(Vector{Float64}[], 1.0)
    @test_throws ArgumentError ensembleSpectralFormFactor([[1.0, 2.0], [1.0]], 1.0)

    # Structural calibration (seeded; bands fixed from the M = 200,
    # N = 100 calibration run of 2026-11-06, which gave dip ratios
    # 0.13 / 0.07 of the plateau for beta = 1 / 2, plateau ratios
    # 0.94 / 0.95 at the Heisenberg time 2pi, and a flat Poisson
    # plateau with minimum 0.84 of 1/N over t >= 0.5).
    N = 100
    M = 200
    tgrid = collect(range(0.1, 6.4, length=64))
    plateau = 1 / N
    att(v, t) = v[argmin(abs.(tgrid .- t))]

    # Poisson levels: straight to the plateau, no dip, no ramp.
    rngP = MersenneTwister(20261111)
    specP = [sort(rand(rngP, N)) .* N for _ in 1:M]
    pP = ensembleSpectralFormFactor(specP, tgrid)
    @test minimum(pP.total[tgrid .>= 0.5]) > 0.45 * plateau
    @test 0.7 * plateau < mean(pP.total) < 1.4 * plateau
    @test 0.65 * plateau < mean(pP.connected) < 1.4 * plateau

    # Unfolded beta-ensemble spectra: dip below the plateau, a ramp,
    # and saturation at 1/N at the Heisenberg time.
    for (β, seed) in ((1.0, 20261112), (2.0, 20261113))
        rngβ = MersenneTwister(seed)
        specβ = [rand(rngβ, BetaHermite(β, N)) for _ in 1:M]
        pβ = ensembleSpectralFormFactor(specβ, tgrid; unfold=true)
        @test minimum(pβ.total) < 0.35 * plateau
        @test att(pβ.total, 3.0) > att(pβ.total, 1.0) > minimum(pβ.total)
        @test 0.3 * plateau < att(pβ.total, 3.0) < 0.85 * plateau
        @test 0.7 * plateau < pβ.total[end] < 1.3 * plateau
        @test 0.65 * plateau < pβ.connected[end] < 1.3 * plateau
    end
end
