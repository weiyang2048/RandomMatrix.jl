using RandomMatrix, LinearAlgebra
using Test

@test randDiagonal(2) !== nothing
@test randTriangular(randTriangular(1:3,3,upper=false,Diag=false)) !== nothing
@test !isreal(randMatrix(10)|>eigvals)


@test isreal(randHermitian(3)|>eigvals)