# Matrices with IID Entries

# Diagonal Matrices

```@docs
randDiagonal
```

# TriangularMatrices

```@docs
randTriangular
```

# Full Matrices

```@docs
randMatrix
```



```@eval 
using RandomMatrix, LinearAlgebra
using Plots; gr()
Plots.reset_defaults()

@gif for n = 1:10:500
     
    randMatrix(n,norm = true)|>
    eigvals|>
    x->scatter(x,ratio=1,xlims=(-1.5,1.5),label="n = $(n)",axis=false)

    title!("Circular Law")
end 
```
