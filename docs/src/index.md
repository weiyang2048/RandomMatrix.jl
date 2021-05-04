# RandomMatrix.jl Documentation

A package for Random Matrix Theory.

Hoping to add-in enough functionalities and register in Aug 2021. If there is any functionality you want me to implement, please raise an issue.
```@example
using Plots, RandomMatrix, LinearAlgebra
gr() # hide
Plots.reset_defaults() # hide
M = randMatrix(1000)
H = Hermitian(M)

l=@layout[c c]
colors = [:red,:blue,:green,:purple]
@gif for n = 1:1:1000
     
  p1 = M[1:n,1:n]/sqrt(n)|>
        eigvals|>
        x->scatter(x,
            ylims=(-1.5,1.5), xlims=(-1.5,1.5),
            label="n = $(n)", axis=false, ratio=1, c=rand(colors))
  title!("Circular Law")
    
  p2 = H[1:n,1:n]/sqrt(n)|>
        eigvals|>
        x->histogram(x,norm=true,
        ylims=(0,0.5), xlims=(-2.5,2.5),
        nbins = ceil(Int,sqrt(n)), label="n = $(n)",c=rand(colors))
  title!("Semicircle Law")  
  
  plot(p1,p2,size = (1000, 500))
end 

```