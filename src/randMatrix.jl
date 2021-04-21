export randMatrix
"""
```julia
randMatrix(d::T, n::Int, m = n ::Int ; ; norm = false::Bool) where T<:Union{Distribution{Univariate},DataType,AbstractArray, Tuple}

randMatrix(n::Int, m = n :: Int; norm = false::Bool)
```
- If `norm` set to `true`, then the matrix will be normlaized with n^(-1/2).  
# Examples
```julia
randMatrix(2)
``` 
>Generates a 2 by 2 random  matrix with entries from the Standard  Gaussian.
```julia
randMatrix(1:10,3,2)
``` 
>Generates a 3 by 2 random  matrix with entries uniform in {1,2,3,...,10}.
```julia
randMatrix(Poisson(2),2,norm = true)
``` 

>Generate a normalized random 2 by 2  Matrix with entries  `Poisson(2)` rvs. 
>*Need to import the `Distributions` package for `Poisson(2)`*
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