export randHermitian, randSymmetric

"""
```julia
randHermitian(d::T, n::Int; norm = false::Bool, complex=true::Bool) where T<:Union{Distribution{Univariate},DataType,AbstractArray, Tuple}

randHermitian(n::Int; norm = false::Bool)
```
- If `norm` set to `true`, then the matrix will be normlaized with n^(-1/2).  
- If one knows that all entries will be real, set `complex=false`,
    or equivalently use `randSymmetric`
# Examples
```julia
randHermitian(2)
``` 
>Generates a 2 by 2 random Hermitian matrix with entries from the Standard Complex Gaussian.
```julia
randHermitian(Poisson(2),2)
``` 
or
```julia
randSymmetric(Poisson(2),2)
```
>Generate a random 2 by 2 Symmetric Matrix with entries  `Poisson(2)` rvs. 
>*Need to import the `Distributions` package for `Poisson(2)`*
```julia
randHermitian(1:10,2)
``` 
```julia
randHermitian([-1,pi],2)
```
"""
function randHermitian(d::T, n::Int; norm = false::Bool, complex=true::Bool)  where T<:Union{Distribution{Univariate}
                                            ,DataType,AbstractArray, Tuple}
    if complex
        M = zeros(ComplexF64,n,n)
    else
        M =zeros(n,n)
    end
    for i in 1:n, j in i:n
        M[i,j] = rand(d)
    end

    if norm
        M/=sqrt(n)
    end
    
    return Hermitian(M)
end


function randHermitian(n::Int; norm = false::Bool)
    return randHermitian(ComplexNormal(), n; norm = norm)
end


"""
```julia
randSymmetric(d::T, n::Int; norm = false::Bool) where T<:Union{Distribution{Univariate},DataType,AbstractArray, Tuple}

randSymmetric(n::Int; norm = false::Bool)
```
- Essentially equivalent to `randHermitian` with `complex = false`
- If `norm` set to `true`, then the matrix will be normlaized with n^(-1/2).  
# Examples
```julia
randSymmetric(2)
``` 
>Generates a 2 by 2 random Symmetric matrix with entries from the Standard Gaussian.
"""
function  randSymmetric(d::T, n::Int; norm = false::Bool)  where T<:Union{Distribution{Univariate}
    ,DataType,AbstractArray, Tuple}
    return randHermitian(d, n; norm = norm, complex=false)
end

function randSymmetric(n::Int; norm = false::Bool)
    return randSymmetric(Normal(), n; norm = norm)
end
