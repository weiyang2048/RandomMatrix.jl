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
using RandomMatrix, Plots

anim = @animate for n = 1:10:1000
     
    randMatrix(n,norm = true)|>
    eigvals|>
    x->scatter(x,ratio=1,xlims=(-1.5,1.5),label="n = $(n)")

    title!("Circular Law")
end

gif(anim,raw"Circular Law.gif",fps=20)
```
![](Circular Law.gif)