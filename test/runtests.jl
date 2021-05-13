using RandomMatrix, LinearAlgebra
using Test

@test randDiagonal(2) !== nothing
@test randTriangular(1:3,3,upper=false,Diag=false) !== nothing
@test !isreal(randMatrix(10)|>eigvals)


@test isreal(randHermitian(3)|>eigvals)
@test randHermitian(ComplexNormal(im,2),3,diag=Elliptic(0.1,c=im,R=9)) !==nothing

@test pdf(MarchenkoPastur(rand()),0) == 0