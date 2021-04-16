export resolvent, qresolvent

"""
```julia
resolvent(A)
```
return the resolvent ***function*** of A
"""
function resolvent(A::Matrix)
    return function f(z::Number)
    (A-z*I)^(-1)
end
end

"""
```julia
qresolvent(A)
```
returns the quaternion resolvent ***function*** of A.
"""
function qresolvent(A::Matrix)
    return function f(z::Number,η::Number)
    ([-η*I A-z*I 
    A'-z'*I -η*I])^(-1)
end
end