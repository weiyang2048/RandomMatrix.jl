module RandomMatrix

using LinearAlgebra

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

end # module
