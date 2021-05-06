

#################################################
"""
```julia
randHermitian(d, n; diag, norm )

randHermitian(n; norm)
```
- `d` : entry distribution
- `n`  : dimensions 
- `norm` : default is set to `false`, if `norm` set to `true`, then the matrix will be normalized with ``n^{-1/2}``.  
- `diag` : the distribution for diagonal entries, by default `diag=d`. 
    To use a different distribution (say Binomial) for digonal elements, set `diag = Binomial(1,0.5)`

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
    
    M = randTriangular(d,n,diag=diag)

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
randSymmetric(d, n; Diag, norm) 

randSymmetric(n; norm)
```
- `d` : entry distribution
- `n` : dimensions 
- `norm` : default  `false`, if `norm` set to `true`, then the matrix will be normalized with n^(-1/2).  
- `diag` : default `diag = d`, the distribution for diagonal entries. 
    To use a different distribution (say Binomial) for digonal elements, set `diag = Binomial(1,0.5)`
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
    
    return Symmetric(convert(Matrix,M))
end


function randSymmetric(n::Int; norm = false::Bool)
    return randSymmetric(Normal(), n, norm = norm)
end

"""
```julia
GOE <: ContinuousMatrixDistribution
GOE(n)
```

- `n` : dimension

# Examples

Sample a 3 by 3 random matrix from GOE(5)
```julia
julia> rand(GOE(3))

3×3 Symmetric{Float64, Matrix{Float64}}:
  0.216246   0.148277  -0.71106
  0.148277  -0.341516  -0.204024
 -0.71106   -0.204024  -0.338064

```
"""
struct GOE <: ContinuousMatrixDistribution
    n::Int
end

rand(rng::AbstractRNG, M::GOE) = randSymmetric(Normal(),M.n,diag=Normal(0,sqrt(2)),norm=true)

"""
```julia
GUE <: ContinuousMatrixDistribution
GUE(n)
```

- `n` : dimension
"""
struct GUE <: ContinuousMatrixDistribution
    n::Int
end

rand(rng::AbstractRNG, M::GUE) = randHermitian(ComplexNormal(),M.n,diag=Normal(),norm=true)
