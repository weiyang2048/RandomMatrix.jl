

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
- See also [`GUE`](@ref)

# Examples
Generates a 2 by 2 random Hermitian matrix with off-diagonal entries from the Standard Complex Gaussian, and Standard Normal on the diagonal.
```julia
randHermitian(2)

2×2 Hermitian{ComplexF64, Matrix{ComplexF64}}:
  0.382095+0.0im        -0.708469-0.0636734im
 -0.708469+0.0636734im   0.336952+0.0im
```
Generate a 3 by 3 Hermitian matrix, with off-diagonal entries [`Circular`](@ref)(1) and diagonal entries uniformly `-1` or `1`.
```julia
randHermitian(Circular(1),3,diag = (-1,1))

3×3 Hermitian{ComplexF64, Matrix{ComplexF64}}:
     1.0+0.0im       1.56259-0.676099im  1.39468-0.295073im
 1.56259+0.676099im     -1.0+0.0im       1.53369+0.296583im
 1.39468+0.295073im  1.53369-0.296583im     -1.0+0.0im
```
Generate a random 2 by 2 Symmetric Matrix with entries  `Poisson(2)` rvs. 
This is also be done with `randSymmetric(Poisson(2),3)`
```julia
randHermitian(Poisson(2),3)

3×3 Hermitian{Int64, Matrix{Int64}}:
 3  1  0
 1  1  2
 0  2  1
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

##########################################################
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
- See also [`GOE`](@ref)

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

#########################################################

"""
```julia
GOE <: ContinuousMatrixDistribution
GOE(n) 
```
- `n` : dimension
- The **Gaussian Orthogonal Ensemble** (`GOE`) is an ensemble of random ``n \\times n`` Symmetric matrices 
    ``M_{n}`` in which the upper-triangular entries are iid with distribution ``N(0,1)_{\\mathbf{R}}``, 
    and the diagonal entries are iid with distribution ``N(0,2)_{\\mathbf{R}}``, and independent of the upper-triangular ones

```julia
rand(M::GOE, norm::bool)
```
 -  `norm` : default `false`,  if `norm` set to `true`, then the matrix will be normlaized with ``\\operatorname{min}(n,m)^{-1/2}``.  

# Examples

Generate a 3 by 3 random matrix from GOE(3)
```julia
rand(GOE(3))

3×3 Symmetric{Float64, Matrix{Float64}}:
 -1.62391   -0.451433    0.863883
 -0.451433   0.0271799  -0.524854
  0.863883  -0.524854   -0.00930624
```

```julia
rand(GOE(3),norm=true)

3×3 Symmetric{Float64, Matrix{Float64}}:
  0.302141   0.152634   -0.711679
  0.152634  -0.0629327   0.103075
 -0.711679   0.103075    1.51861
```

"""
struct GOE <: ContinuousMatrixDistribution
    n::Int
end

rand( M::GOE; norm = false::Bool) = randSymmetric(Normal(),M.n,diag=Normal(0,sqrt(2)),norm=norm)

"""
```julia
GUE <: ContinuousMatrixDistribution
GUE(n)
```
- `n` : dimension
- The **Gaussian Unitary Ensemble** (`GUE`) is an ensemble of random ``n \\times n`` Hermitian matrices 
    ``M_{n}`` in which the upper-triangular entries are iid with distribution ``N(0,1)_{\\mathbf{C}}``, 
    and the diagonal entries are iid with distribution ``N(0,1)_{\\mathbf{R}}``, and independent of the upper-triangular ones

```julia
rand(M::GUE, norm::bool) 
```
- `norm` : default `false`,  if `norm` set to `true`, then the matrix will be normlaized with ``\\operatorname{min}(n,m)^{-1/2}``.  

# Examples

Generate a 3 by 3 random matrix from GUE(3)
```julia
rand(GUE(3))

3×3 Hermitian{ComplexF64, Matrix{ComplexF64}}:
 -0.883413+0.0im         1.09872+0.874884im     -0.1985-1.04778im
   1.09872-0.874884im    1.55483+0.0im        -0.488532+1.18694im
   -0.1985+1.04778im   -0.488532-1.18694im   -0.0823873+0.0im
```

```julia
rand(GUE(2),norm=true)
2×2 Hermitian{ComplexF64, Matrix{ComplexF64}}:
 -0.457089+0.0im       -0.672713-0.102234im
 -0.672713+0.102234im   0.380126+0.0im
```
"""
struct GUE <: ContinuousMatrixDistribution
    n::Int
end

rand( M::GUE; norm=false::Bool) = randHermitian(ComplexNormal(),M.n,diag=Normal(),norm=norm)
