export  ComplexNormal, rand

struct  ComplexNormal <:ContinuousUnivariateDistribution
    μ::ComplexF64
    σ::Float64
    function ComplexNormal(μ=0,σ=1) new(μ,σ) end
end

function Base.rand(d::ComplexNormal)
    randn(ComplexF64)*d.σ+d.μ
end
