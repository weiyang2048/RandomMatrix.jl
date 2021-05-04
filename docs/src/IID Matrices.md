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



```@example index
using Plots, RandomMatrix, LinearAlgebra
gr() # hide
Plots.reset_defaults() # hide

@gif for n = 1:1:500
     
    randMatrix(n,norm = true)|>
        eigvals|>
        x->scatter(x, ratio=1,
            ylims=(-1.5,1.5), xlims=(-1.5,1.5),
            label="n = $(n)", axis=false)

    title!("Circular Law")
end 
```
