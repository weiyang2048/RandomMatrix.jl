export randUnitary, randOrthogonal , Haar, randPermutation

"""
```julia
randUnitary(n::Int)
```
- Generates a n by n random Unitary matrix
- Equivalent to run `rand(Haar(2,n))`
- For orthogonal matrices, use `randOrthogonal` or `rand(Haar(1,n))` instead

```julia
# Examples

# Generate a 3 by 3 random Unitary matrix 
randUnitary(3)
# or
rand(Haar(2,3)) 
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
- Generates a n by n random Orthogonal matrix
- Equivalent to run `rand(Haar(1,n))`
- For unitary matrices, use `randUnitary` or `rand(Haar(2,n))` instead
 
```julia
# Examples

# Generates a 3 by 3 random Orthogonal matrix 
randOrthogonal(3)
# or
rand(Haar(1,3))
```
"""
function randOrthogonal(n::Int)
    A = randn(n,n)
    Q,R = qr(A)
    Q,R = convert(Matrix,Q),convert(Matrix,R)
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
 
```julia
# Examples 

# Generates a random 5 by 5 permutation matrix
randPermutation(5)

# Generates  Generates a random 100 by 100 permutation matrix with atleast 10 fix points
randPermutation(100,fix=10)
```
"""
function randPermutation(n::Int; fix = 0::Int) 
    O = sample(1:n,n-fix,replace=false) # randly choosing flexible indices
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

