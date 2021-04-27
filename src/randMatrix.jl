export randMatrix
"""
```julia
randMatrix(d::T, n::Int, m = n::Int  ; norm = false::Bool) where T<:Union{Distribution{Univariate},DataType,AbstractArray, Tuple}

randMatrix(n::Int, m = n :: Int; norm = false::Bool)
```
- `d` : entry distribution
- `n` , 'm' : dimensions, if `m` is not provided, by default `m=n` 
- `norm` : default is set to `false`, if `norm` set to `true`, then the matrix will be normlaized with n^(-1/2).  

```julia
# Examples

# Generates a 2 by 2 random  matrix with entries from the Standard  Gaussian.
randMatrix(2)

# Generates a 3 by 2 random  matrix with entries uniformly from {1,2,3,...,10}.
randMatrix(1:10,3,2)

# Generate a normalized random 2 by 2  Matrix with entries  `Poisson(2)` rvs. 
# Need to import the `Distributions` package for `Poisson(2)`
randMatrix(Poisson(2),2,norm = true)
``` 

"""
function randMatrix(d::T, n::Int, m = n ::Int ; norm = false::Bool)  where T<:Union{Distribution{Univariate}
                                            ,DataType,AbstractArray, Tuple}
 
    M = reshape([rand(d) for i = 1:n*m],(n,m))

    if norm
        M/=sqrt(min(n,m))
    end
    
    return M
end

function randMatrix(n::Int, m = n ::Int; norm = false::Bool)
    return randMatrix(Normal(), n, m; norm = norm)
end