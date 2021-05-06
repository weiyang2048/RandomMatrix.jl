```@docs
randHermitian

randSymmetric

randUnitary

randOrthogonal

Haar

randPermutation

randEllipic

resolvent

ComplexNormal

Gaussian

MarchenkoPastur

±
```

```@example 
using Plots
@gif for i =1 :10
    plot(rand(10))
end
```



```@example 
using Plots
q = @animate for i =1 :10
    plot(rand(10))
end
gif(q,"q.gif")
```
![](q.gif)