

#################################################
"""
```julia
randHermitian(d, n; diag, norm )

randHermitian(n; norm)
```
- `d` : entry distribution
- `n`  : dimensions 
- `norm` : default `false`, if `norm` set to `true`, then the matrix will be normalized with ``n^{-1/2}``.  
- `diag` : default `diag = d`, diagonal entry distribution. 
    To use a different distribution (say Circular(2)) for digonal elements, set `diag = Circular(2)`.  
    The diagonal entries will always be forced to have imgainary part `0`.

# Examples
Generates a 2 by 2 random Hermitian matrix with off-diagonal entries from the Standard Complex Gaussian, and Standard Normal on the diagonal.
```julia
randHermitian(2)

2×2 LinearAlgebra.Hermitian{ComplexF64, Matrix{ComplexF64}}:
 -0.385798+0.0im       0.789436-0.671441im
  0.789436+0.671441im  -2.00856+0.0im
```
Generate a 3 by 3 Hermitian matrix, with off-diagonal entries `Circular(1)` and diagonal entries uniformly `-1` or `1`.
```julia
randHermitian(Circular(1),3,diag = (-1,1))

3×3 LinearAlgebra.Hermitian{ComplexF64, Matrix{ComplexF64}}:
     -1.0+0.0im        0.785839+0.0137306im  1.61019-0.207893im
 0.785839-0.0137306im       1.0+0.0im        1.30573+0.723762im
  1.61019+0.207893im    1.30573-0.723762im       1.0+0.0im
```
Generate a random 2 by 2 Symmetric Matrix with entries  `Poisson(2)` rvs. 
This is also be done with `randSymmetric(Poisson(2),3)`
```julia
randHermitian(Poisson(2),3)

3×3 LinearAlgebra.Hermitian{Int64, Matrix{Int64}}:
 0  0  6
 0  3  1
 6  1  4
```
Entries uniformly from ``\\{1,2,3,...,10\\}``
```julia
randHermitian(1:10,2)

2×2 Hermitian{Int64, Matrix{Int64}}:
 10  7
  7  6
```

Entries either -1 or pi with equal probability
```julia
randHermitian([-1,pi],3)

3×3 Hermitian{Float64, Matrix{Float64}}:
  3.14159  -1.0      -1.0
 -1.0      -1.0       3.14159
 -1.0       3.14159   3.14159
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
- `norm` : default  `false`, if `norm` set to `true`, then the matrix will be normalized with ``n^{-1/2}``.   
- `diag` : default `diag = d`, the distribution for diagonal entries. 
    To use a different distribution (say Binomial) for digonal elements, set `diag = Binomial(1,0.5)`

# Examples
Generates a 3 by 3 random Symmetric matrix with entries from the Standard Gaussian.
```julia
randSymmetric(3)

3×3 Symmetric{Float64, Matrix{Float64}}:
 -0.230698  -1.72846     0.306362
 -1.72846    0.0845915  -0.0116108
  0.306362  -0.0116108  -0.559046
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

Generate a 3 by 3 random matrix from GOE(3)
```julia
rand(GOE(3))

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
# Examples

Generate a 3 by 3 random matrix from GUE(3)
```julia
rand(GUE(3))

3×3 Hermitian{ComplexF64, Matrix{ComplexF64}}:
 -0.0127056+0.0im          -0.200015+0.00787052im  0.333224-0.157845im
  -0.200015-0.00787052im  0.00753986+0.0im         -0.37598-0.202447im
   0.333224+0.157845im      -0.37598+0.202447im    0.298683+0.0im
```
"""
struct GUE <: ContinuousMatrixDistribution
    n::Int
end

rand(rng::AbstractRNG, M::GUE) = randHermitian(ComplexNormal(),M.n,diag=Normal(),norm=true)
