"""
```julia
randUnitary(n::Int)
```
- Generates a n by n random Unitary matrix
- Equivalent to  `rand(Haar(2,n))`, see [`Haar`](@ref)
- For orthogonal matrices, use `randOrthogonal` or `rand(Haar(1,n))` instead

# Examples

Generate a 3 by 3 random Unitary matrix 
```julia
randUnitary(3)

3×3 Matrix{ComplexF64}:
 -0.149398+0.0572715im  -0.0935861+0.629201im  -0.257255-0.709625im
  0.337035-0.342606im     -0.36366+0.599236im  -0.100838+0.517231im
  -0.17097+0.845103im   -0.0767105+0.313259im   0.247081+0.3025im
```
"""
function randUnitary(n::Int)
    A = randn(ComplexF64,n,n)
    Q,R = qr(A)
    Q,R = convert(Matrix,Q),convert(Matrix,R)
    Q = Q*Diagonal(sign.(diag(R)))
    return Q
end
"""
```julia
randOrthogonal(n::Int)
```
- Generates a `n` by `n` random Orthogonal matrix
- Equivalent to `rand(Haar(1,n))`, see [`Haar`](@ref)
- For unitary matrices, use `randUnitary` or `rand(Haar(2,n))` instead
 
# Examples

Generates a 3 by 3 random Orthogonal matrix 
```julia
randOrthogonal(3)

3×3 Matrix{Float64}:
 -0.875553  0.112448   0.469853
 -0.147915  0.863441  -0.482277
  0.459921  0.491757   0.739356
```
"""
function randOrthogonal(n::Int)
    A = randn(n,n)
    Q,R = qr(A)
    Q = Q*Diagonal(sign.(diag(R)))
    return Q
end
"""
```julia
Haar(beta,n)
```
- Uniform distribution on O(n) (`beta = 1`), U(n) (`beta = 2`)
- `beta`: 1 for Orthogonal, 2 for Unitary
- `n`: dimension

```julia
# Examples

# Generate a 100 by 100 random Unitary Matrix uniformly from U(n)
rand(Haar(2,100))


# Generate a 100 by 100 random Orthogonal Matrix uniformly from O(n)
rand(Haar(1,100))
```
"""
struct Haar <: ContinuousMatrixDistribution
    beta::Int
    n::Int
    Haar(beta,n) = beta in (1,2) ? new(beta,n) : error("Only take beta = 1 for orthogonal, beta = 2 for unitary")
end
Base.size(d::Haar) = d.n
rand(d::Haar) = d.beta == 1 ? randOrthogonal(d.n) : randUnitary(d.n)

"""
```julia
randPermutation(n; fix) 
```
- `n` : dimension
- `fix` : a keyword argument, default is set to `fix = 0`. If `fix = x`, `randPermutation(n,x)` will have atleast `x` fixed points
 
# Examples 

Generates a random 5 by 5 permutation matrix
```julia
randPermutation(5)

5×5 SparseArrays.SparseMatrixCSC{Int8, Int64} with 5 stored entries:
 ⋅  ⋅  ⋅  ⋅  1
 ⋅  1  ⋅  ⋅  ⋅
 1  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅
 ⋅  ⋅  ⋅  1  ⋅
```

Generates  Generates a random 10 by 10 permutation matrix with atleast 7 fix points
```julia
randPermutation(10, fix = 7)

10×10 SparseArrays.SparseMatrixCSC{Int8, Int64} with 10 stored entries: 
 1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅
 ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋮              ⋮
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅
 ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅
 ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1

```
"""
function randPermutation(n::Int; fix = 0::Int) 
    O = sample(1:n,n-fix,replace=false) # randomly choosing flexible indices
    E = shuffle(O) 
    M = spzeros(Int8,n,n)
    for i in 1:n-fix
      M[O[i],E[i]] = 1
    end
    for i in setdiff(1:n,O)
      M[i,i] = 1
    end
    return M
end

