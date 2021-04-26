export ComplexNormal, Gaussian, Circular
        
        # re-export    
        mean, var

"""
```julia
ComplexNormal(μ=0,σ=1) 
```
"""
struct  ComplexNormal <:ContinuousUnivariateDistribution
    μ::ComplexF64
    σ::Float64
    function ComplexNormal(μ=0,σ=1) new(μ,σ) end
end

Base.eltype(::Type{ComplexNormal}) = ComplexF64
Base.rand(rng::AbstractRNG, d::ComplexNormal) = randn(rng,ComplexF64)*d.σ+d.μ;

mean(d::ComplexNormal)=d.μ
var(d::ComplexNormal)=d.σ

"""
```julia
Gaussian(beta=1,μ=0,σ=1)
````
- `beta` 1 for Real Gaussian, 2 for Complex Gaussian 
"""
struct Gaussian <: ContinuousUnivariateDistribution
    beta::Int8
    μ::Union{ComplexF64,Float64}
    σ::Float64
    function Gaussian(beta=1,μ=0,σ=1)  beta==1 ? Normal(μ,σ) : ComplexNormal(μ,σ) end
end


"""
```julia
Circular(c=0,R=1)
```
- The uniform distribution on the complex disk with center `c` and radius `R`

```julia
rand(Circular(1,10),200)
```
> Generates 200 uniform r.v.s on the complex disk of center 1 radius 10.
"""
struct  Circular <:ContinuousUnivariateDistribution
    c::ComplexF64 
    R::Float64  
    function Circular(c=0,R=1) new(c,R) end
end

Base.eltype(::Type{Circular}) = ComplexF64
function Base.rand(rng::AbstractRNG, d::Circular) 
    x_1 = rand(rng,Uniform(-d.R,d.R))
    r = sqrt(d.R^2-x_1^2)
    x_2 = rand(rng,Uniform(-r,r))
    return x_1+x_2*im + d.c
end
