export randMatrix
"""
```julia
randMatrix(d::T, n::Int; norm = false::Bool) where T<:Union{Distribution{Univariate},DataType,AbstractArray, Tuple}
randMatrix(n::Int; norm = false::Bool)
```
- If `norm` set to `true`, then the matrix will be normlaized with n^(-1/2).  
# Examples
```julia
randMatrix(2)
``` 
>Generates a 2 by 2 random  matrix with entries from the Standard  Gaussian.
```julia
randMatrix(Poisson(2),2,norm = true)
``` 

>Generate a normalized random 2 by 2  Matrix with entries  `Poisson(2)` rvs. 
>*Need to import the `Distributions` package for `Poisson(2)`*
"""
function randMatrix(d::T, n::Int; norm = false::Bool)  where T<:Union{Distribution{Univariate}
                                            ,DataType,AbstractArray, Tuple}
 
    M = reshape([rand(d) for i = 1:n^2],(n,n))

    if norm
        M/=sqrt(n)
    end
    
    return M
end

function randMatrix(n::Int; norm = false::Bool)
    return randMatrix(Normal(), n; norm = norm)
end