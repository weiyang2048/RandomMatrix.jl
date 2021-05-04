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



```@example 
using Plots, RandomMatrix, LinearAlgebra
gr() # hide
Plots.reset_defaults() # hide

M = randMatrix(500)
@gif for n = 1:1:500
     
    M[1:n,1:n]/sqrt(n)|>
        eigvals|>
        x->scatter(x, ratio=1,
            ylims=(-1.5,1.5), xlims=(-1.5,1.5),
            label="n = $(n)", axis=false,
            size = (500, 500))
    title!("Circular Law")
end 
```
