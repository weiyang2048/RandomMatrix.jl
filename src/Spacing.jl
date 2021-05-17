export wignersurmise,unfoldedeigvals ,unfoldedSpacings

"""
```julia
wignersurmise(s ; beta)  
```
- `beta` : default `1`, 

    `1` for the spacing density corresponds to GOE(2) 

    ``p_1(s)=\\frac{\\pi s}{2} e^{-\\pi s^{2} / 4}``

    `2` for the spacing density corresponds to GUE(2) 

    ``p_2(s)=\\frac{32 s^{2}}{\\pi^{2}} e^{-4 s^{2} / \\pi}``

-  For more info see [Wigner surmise on Wikipedia](https://en.wikipedia.org/wiki/Wigner_surmise)


# Examples 
Evaluate Wigner surmise for GOE at `2`
```julia
wignersurmise(2)

0.1357605281502967
```

Evaluate Wigner surmise for GUE at `2`
```julia
wignersurmise(2,beta=2)

0.02239541536704175
```
"""
function wignersurmise(s::Real; beta = 1 :: Int)  
    if beta == 1
        return (pi/2) * s * exp(-pi * s^2 / 4)
    elseif beta == 2
        return (3/pi)^2 * s^2 * exp(- 4 * s^2 / pi)
    end
end

"""
```julia
unfoldedeigvals(E; deg, goe)
```
- `E` : The array of eigenvalues
- `deg` : default `10`z
- `goe` : default `false`
"""
function unfoldedeigvals(E::Array; deg=10::Int, goe = false::Bool)
    if goe
        return length(E)*cdf(Semicircle(2),E)
    end
    
    return polyfit(E,(x->sum(E.<x)).(E),deg).(E)

end
function unfoldedSpacings(E::Array;deg=10::Int, goe = false::Bool)
    return diff(unfoldedeigvals(E,deg=deg,goe=goe))
end
