```@docs
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
q = @animate for i =1 :10
    plot(rand(10))
end
gif(q,"p.gif")
```


 
```@example 
using Plots
q = @animate for i =1 :10
    plot(rand(10))
end
gif(q,"q.gif")
```
![](q.gif)![](p.gif)