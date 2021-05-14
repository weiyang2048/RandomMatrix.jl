# Hermitian and Symmetric Matrices

## Hermitian
```@docs
randHermitian
```


## Symmetric
```@docs
randSymmetric
```



## RMT: Semicircle Law

```@eval
using Plots, RandomMatrix, LinearAlgebra, Distributions

N = 1000
H = randHermitian(N)

colors = [:red,:green,:blue,:purple]
ani = @animate for n = (1:50...,51:10:N...,N:-10:51...,50:1...)

H[1:n,1:n]/sqrt(n)|> eigvals|> x->histogram(x,norm=true, ylims=(0,1/pi+0.2), xlims=(-2.25,2.25),nbins =ceil(Int,n/10),label="n = $(n)")
        plot!(x->pdf(Semicircle(2),x),-2:0.01:2,lw=3,label="",c=[rand(colors) for _=-2:0.01:2])
        title!("Semicircle Law for Hermitian Matrices")

end 
gif(ani, "H1.gif", fps = 10); nothing
```
![](H1.gif)

```@eval
using Plots, RandomMatrix, LinearAlgebra, Distributions
N = 2000
colors = [:red,:green,:blue,:purple]
anim = @animate for n = (1:2:50...,51:20:N...,N:-20:51...,50:-2:1...)
    M = RandomHermitian(Normal(0,1), n) 
    histogram(eigvals(M), normed=true, nbins=100,ratio = 7, label="n=$n")
    plot!(x->1/2pi*sqrt(4-x^2), -2:0.001:2, linewidth = 5, label="Semicircle",c=[rand(colors) for _=-2:0.01:2])
    xlims!(-2.5,2.5)
    ylims!(0,0.5)
    title!("Semicircular Law")
end

gif(anim,"Semicircle.gif",fps=20); nothing
```
![](Semicircle.gif)