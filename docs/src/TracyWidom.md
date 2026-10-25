# Tracy–Widom Distributions

The Tracy–Widom law ``F_\beta`` is the limiting distribution of the
largest eigenvalue of a Gaussian ``\beta``-ensemble, and — after the
[`twScaling`](@ref) centering — of the largest eigenvalue of a Wishart
matrix. The distribution is computed from the Hastings–McLeod solution
of Painlevé II; published means are ``-1.20653`` (``\beta = 1``),
``-1.77109`` (``\beta = 2``) and ``-2.30688`` (``\beta = 4``).

```@docs
TracyWidom
twScaling
```
