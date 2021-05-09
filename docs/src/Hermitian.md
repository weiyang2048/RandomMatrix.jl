# Hermitian and Symmetric Matrices

## Hermitian
```@docs
randHermitian
```
## GUE
```@docs
GUE
```

## Symmetric
```@docs
randSymmetric
```

## GOE
```@docs
GOE
```


## RMT: Semicircle Law

```@eval
using Plots, RandomMatrix, LinearAlgebra, Distributions

N = 500
H = randHermitian(N)

colors = [:red,:green,:blue,:purple]
ani = @animate for n = (1:50...,51:10:N...,N:-10:51...,50:1...)

H[1:n,1:n]/sqrt(n)|> eigvals|> x->histogram(x,norm=true, ylims=(0,1/pi+0.2), xlims=(-2.25,2.25),nbins = ceil(Int,sqrt(n)),label="n = $(n)")
        plot!(x->pdf(Semicircle(2),x),-2:0.01:2,lw=3,label="",c=[rand(colors) for _=-2:0.01:2])
        title!("Semicircle Law for Hermitian Matrices")

end 
gif(anim, "H1.gif", fps = 10)
```
![](H1.gif)