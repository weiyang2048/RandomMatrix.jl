

#################################################
"""
```julia
randHermitian(d, n; Diag, norm )

randHermitian(n; norm)
```
- `d` : entry distribution
- `n`  : dimensions 
- `norm` : default is set to `false`, if `norm` set to `true`, then the matrix will be normalized with n^(-1/2).  
- `Diag` : the distribution for diagonal entries, by default `Diag=d`. 
    To use a different distribution (say Binomial) for digonal elements, set `Diag = Binomial(1,0.5)`

# Examples
Generates a 2 by 2 random Hermitian matrix with entries from the Standard Complex Gaussian.
```julia
randHermitian(2)
```
Generate a random 2 by 2 Symmetric Matrix with entries  `Poisson(2)` rvs. 
*Need to import the `Distributions` package for `Poisson(2)`*
```julia
randHermitian(Poisson(2),2)
# or equivalently 
randSymmetric(Poisson(2),2)

# Entries uniformly from {1,2,3,...,10}
randHermitian(1:10,2)

# Entries either -1 or pi with equal probability
randHermitian([-1,pi],2)
``` 
"""
function randHermitian(d::D, n::Int; diag=d::D, norm = false::Bool)  where D<:S
    
    M = randTriangular(d,n,Diag=Diag)

    if norm
        M/=sqrt(n)
    end
    
    return Hermitian(convert(Matrix,M))
end


function randHermitian(n::Int; norm = false::Bool)
    return randHermitian(ComplexNormal(), n, diag=Normal(), norm = norm)
end


"""
```julia
randSymmetric(d::D, n::Int; Diag = d::D,  norm = false::Bool) where T<:S

randSymmetric(n::Int; norm = false::Bool)
```
- `d` : entry distribution
- `n` : dimensions 
- `norm` : default is set to `false`, if `norm` set to `true`, then the matrix will be normalized with n^(-1/2).  
- `Diag` : the distribution for diagonal entries, by default `Diag=d`. 
    To use a different distribution (say Binomial) for digonal elements, set `Diag = Binomial(1,0.5)`
```julia
# Examples

# Generates a 2 by 2 random Symmetric matrix with entries from the Standard Gaussian.
randSymmetric(2)
``` 
"""
function  randSymmetric(d::D, n::Int; diag = d::D, norm = false::Bool)  where D<:S

    M = randTriangular(d,n,diag=diag)

    if norm
        M/=sqrt(n)
    end
    
    return Symmetric(convert(Matrix,M),M)
end


function randSymmetric(n::Int; norm = false::Bool)
    return randSymmetric(Normal(), n, norm = norm)
end

struct GOE <: ContinuousMatrixDistribution
    n::Int
end

rand(rng::AbstractRNG, M::GOE) = randSymmetric(Normal(),M.n,diag=Normal(0,sqrt(2)),norm=true)

struct GUE <: ContinuousMatrixDistribution
    n::Int
end

rand(rng::AbstractRNG, M::GUE) = randHermitian(ComplexNormal(),M.n,diag=Normal(),norm=true)
