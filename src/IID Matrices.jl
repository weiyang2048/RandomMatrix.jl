
"""
```julia
randDiagonal(d, n) 

randDiagonal(n) 
```
- `d` : default `Normal()`, entry distribution
- `n` : dimension

```julia
# Examples

# generates a 3 by 3 diagonal matrix, with non-zero elements from `Normal(0,1)`
randDiagonal(3)
3×3 Diagonal{Float64, Vector{Float64}}:
 0.440359   ⋅         ⋅
  ⋅        1.94832    ⋅
  ⋅         ⋅       -0.52536

# generates a 5 by 5 diagonal matrix, with non-zero elements from `Poisson(2)`
julia> randDiagonal(Poisson(2),5)
5×5 Diagonal{Int64, Vector{Int64}}:
 1  ⋅  ⋅  ⋅  ⋅
 ⋅  0  ⋅  ⋅  ⋅
 ⋅  ⋅  0  ⋅  ⋅
 ⋅  ⋅  ⋅  3  ⋅
 ⋅  ⋅  ⋅  ⋅  3
```
"""
function randDiagonal(d::D, n::Int)  where D<:S
    Diagonal(rand(d,n))
end

randDiagonal(n::Int) =  randDiagonal(Normal(), n::Int)


"""
```julia 
randTriangular(d , n ;  diag , Diag, upper ) 

randTriangular(n;diag, upper)
```
- `d` : entry distribution
- `n` : dimension
- `diag` : default `diag = d`, diagonal entry distribution
- `Diag` : default `Diag = true`, `true` includes diagonal, `false` with diagonal entries 0
- `upper` : default `upper = true`, `true` gives upper triangular, `false` gives lower triangular

```julia
# Examples 

# Generate an upper triangular matrix with entries Standard Normal
julia> randTriangular(3)
3×3 UpperTriangular{Float64, Matrix{Float64}}:
 -0.572757  -0.459518   -1.60622
   ⋅         0.0216834  -0.416529
   ⋅          ⋅         -1.00807

# Generate a 3 by 3 strictly lower triangular matrix, with nonzero entries uniform from ``\\{1,2,3\\}`` 
julia> randTriangular(1:3,3,upper=false,Diag=false)
3×3 LowerTriangular{Int64, Transpose{Int64, Matrix{Int64}}}: 
 0  ⋅  ⋅
 3  0  ⋅
 3  2  0

```

"""
function randTriangular(d::D, n::Int; diag=d::D, Diag=true, upper = true)  where D<:S
    M = UpperTriangular(zeros(eltype(d),n,n)) 
    for i in 1:n, j in i+1:n
        M[i,j] = rand(d)
    end
    if Diag
        M += randDiagonal(diag,n)
    end

    if !upper
        M = transpose(M)
    end
    M
end
randTriangular(n::Int; Diag=true, upper = true) = randTriangular(Normal(),n,Diag=Diag,upper=upper)


"""
```julia
randMatrix(d::D, n::Int, m = n::Int; norm = false::Bool) where D<:S

randMatrix(n::Int, m = n::Int; norm = false::Bool)
```
- `d` : entry distribution
- `n`,`m` : default `m = n` , dimensions

Keyword arguments:
- `norm` : default `false`, if `norm` set to `true`, then the matrix will be normlaized with ``min{n,m}^(-1/2)``.  

```julia
# Examples

# Generates a 2 by 2 random  matrix with entries from the Standard  Gaussian.
julia> randMatrix(2)
2×2 Matrix{Float64}:
 1.74043  -1.30317
 0.72765   0.639943

# Generates a 3 by 2 random  matrix with entries uniformly from {1,2,3,...,10}.
julia> randMatrix(1:10,3,2)
3×2 Matrix{Int64}:
  1  3
  6  4
 10  1

# Generate a normalized random 2 by 2  Matrix with entries  `Poisson(2)` rvs. 
# Need to import the `Distributions` package for `Poisson(2)`
randMatrix(Poisson(2),2,norm = true)
2×2 Matrix{Float64}:
 1.41421   0.0
 0.707107  1.41421
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