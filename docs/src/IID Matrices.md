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


```@setup index
using Plots; gr()
Plots.reset_defaults()
```

```@example index
using RandomMatrix, LinearAlgebra

@gif for n = 1:10:500
     
    randMatrix(n,norm = true)|>
    eigvals|>
    x->scatter(x,ratio=1,xlims=(-1.5,1.5),label="n = $(n)",axis=false)

    title!("Circular Law")
end 
```
