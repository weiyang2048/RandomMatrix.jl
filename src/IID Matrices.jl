
"""
```julia
randDiagonal(d, n) 

randDiagonal(n) 
```
- `d` : entry distribution
- `n` : dimension

```julia
# Examples

# generates a 5 by 5 diagonal matrix, with non-zero elements from `Normal(0,1)`
randDiagonal(5)

# generates a 5 by 5 diagonal matrix, with non-zero elements from `Poisson(2)`
randDiagonal(Poisson(2),5)
```
"""
function randDiagonal(d::D, n::Int)  where D<:S
    Diagonal(rand(d,n))
end

randDiagonal(n::Int) =  randDiagonal(Normal(), n::Int)

###############################
"""
```julia 
randTriangular(d::D, n::Int; Diag=d::D ,diag=true, upper = true) 

randTriangular(n::Int;diag=true, upper = true)
```
- `d` : entry distribution
- `n` : dimension
- `Diag` : default `Diag = d`, diagonal entry distribution
- `diag` : default `diag = true`, `true` includes diagonal, `false` with diagonal entries 0
- `upper` : default `upper = true`, `true` gives upper triangular, false gives lower triangular
"""
function randTriangular(d::D, n::Int; Diag=d::D ,diag=true, upper = true)  where D<:S
    M = UpperTriangular(zeros(eltype(d),n,n)) 
    for i in 1:n, j in i+1:n
        M[i,j] = rand(d)
    end
    if diag
        M+=randDiagonal(Diag,n)
    end

    if !upper
        M=transpose(M)
    end
    M
end
randTriangular(n::Int;diag=true, upper = true) = randTriangular(Normal(),n,diag=diag,upper=upper)


"""
```julia
randMatrix(d::D, n::Int, m = n::Int; norm = false::Bool) where D<:S

randMatrix(n::Int, m = n::Int; norm = false::Bool)
```
- `d` : entry distribution
- `n`,`m` : dimensions, if `m` is not provided, by default `m=n` 
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
function randMatrix(d::D, n::Int, m = n::Int ; norm = false::Bool)  where D<:S
 
    M =  rand(d,n,m)

    if norm
        M/=sqrt(min(n,m))
    end
    
    return M
end

function randMatrix(n::Int, m = n::Int; norm = false::Bool)
    return randMatrix(Normal(), n,m ; norm = norm)
end