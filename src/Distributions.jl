export  ComplexNormal, rand

struct  ComplexNormal <:ContinuousUnivariateDistribution
    μ::ComplexF64
    σ::Float64
    function ComplexNormal(μ=0,σ=1) new(μ,σ) end
end

Base.rand(rng::AbstractRNG, d::ComplexNormal) = randn(rng,ComplexF64)*d.σ+d.μ;
Base.eltype(::Type{ComplexNormal}) = ComplexF64

function Base.rand(d::ComplexNormal)
    randn(ComplexF64)*d.σ+d.μ
end

