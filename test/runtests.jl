using RandomMatrix, LinearAlgebra
using Test

@test isreal(randMatrix(10)|>eigvals) == false