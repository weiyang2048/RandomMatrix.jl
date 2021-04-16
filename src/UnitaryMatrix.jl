export randUnitary, randOrthogonal , Haar

"""
- To generate a 3 by 3 **random Unitary matrix** run 
  - ```julia
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
- To generate a 3 by 3 **random Orthogonal matrix** run 
  - ```julia
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

struct Haar <: ContinuousMatrixDistribution
    beta::Int
    n::Int
    Haar(beta,n) = beta in (1,2) ? new(beta,n) : error("Only take beta = 1 for orthogonal, beta = 2 for unitary")
end
Base.size(d::Haar) = d.n
Base.rand(d::Haar) = d.beta == 1 ? randOrthogonal(d.n) : randUnitary(d.n)