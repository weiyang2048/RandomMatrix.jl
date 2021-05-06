# Matrices with IID Entries

## Diagonal Matrices

```@docs
randDiagonal
```

## TriangularMatrices

```@docs
randTriangular
```

## Full Matrices

```@docs
randMatrix
```


## Some Random Matrix Theory
```@example 
using Plots, RandomMatrix, LinearAlgebra
gr() # hide
Plots.reset_defaults() # hide
N = 500
M = randMatrix(N)
colors = [:red,:green,:blue,:purple]
@gif for n = (1:50...,51:10:N...,N:-10:51...,50:1...)
     
    M[1:n,1:n]/sqrt(n)|>eigvals|>x->scatter(x,ylims=(-1.25,1.25), xlims=(-1.25,1.25),ratio=1,label="n = $(n)",size=(600,600))

    plot!([exp(θ*im) for θ=0:0.01:2pi],label="",lw=3,c=[rand(colors) for _=0:0.01:2pi])
        
    title!("Circular Law for IID Matrices")
    
end 
```
