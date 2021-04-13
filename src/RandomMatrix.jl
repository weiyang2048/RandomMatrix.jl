module RandomMatrix

using LinearAlgebra, StatsBase

export RandomUnitaryMatrix, RandomOrthogonalMatrix 

function RandomUnitaryMatrix(n::Int)
    A = randn(ComplexF64,n,n)
    Q = convert(Matrix,qr(A).Q)
    return Q
end
function RandomOrthogonalMatrix(n::Int)
    A = randn(n,n)
    Q = convert(Matrix,qr(A).Q)
    return Q
end


export Norm2SamplingMatrix


"""
Norm2SamplingMatrix(A::Matrix,B=Matrix(A')::Matrix; k::Int)

Generate a random norm-square sampling Matrix S.
S is usually used to sample k    columns of A and rows of B

# Examples
```julia-repl
S = Norm2SamplingMatrix(rand(2,3),rand(3,2),k=2)

3×2 Matrix{Float64}:
 0.0      0.0
 0.0      1.2439
 1.15342  0.0
```
"""
function Norm2SamplingMatrix(A::Matrix,B=Matrix(A')::Matrix; k::Int)
    n, m = size(A)
    if size(B)[1] != m
        return "The matrices can not be multiplied "
    else
        Lens = [norm(A[:,i])*norm(B[i,:]) for i = 1:m ]
        Total = sum(Lens)
        temp = zeros(m,k)
        for (i,j) in enumerate([sample([1:m...], Weights(Lens) ) for i = 1 : k])
            temp[j,i] = 1/sqrt(Lens[j]/Total*k) # normalizing so that E[SS']=I
        end
        return temp
    end
end


end # module
