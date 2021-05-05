export ±

 
"""
```julia
±(a,b)
```
- returns (a-b,a+b)
```julia
# Examples

1 ± 0.5  # returns (0.5,1.5)
```
"""
±(a,b) = [a-b,a+b]