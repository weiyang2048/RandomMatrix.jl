export wignersurmise, unfoldedeigvals, unfoldedSpacings, numberVariance

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

0.07962814352725955
```
"""
function wignersurmise(s::Real; beta=1::Int)
    if beta == 1
        return (pi / 2) * s * exp(-pi * s^2 / 4)
    elseif beta == 2
        return 32 * (s / pi)^2 * exp(-4 * s^2 / pi)
    end
end

"""
```julia
unfoldedeigvals(E; deg, goe)
```
- `E` : The array of eigenvalues
- `deg` : default `10`
- `goe` : default `false`
"""
function unfoldedeigvals(E::Array; deg=10::Int, goe=false::Bool)
    sort!(E)
    if goe
        return length(E) * cdf(Semicircle(2), E)
    end

    return polyfit(E, (x -> sum(E .< x)).(E), deg).(E)

end


"""
```julia
unfoldedSpacings(E; deg, goe)
```
- `E` : The array of eigenvalues
- `deg` : default `10`
- `goe` : default `false`
```
"""
function unfoldedSpacings(E::Array; deg=10::Int, goe=false::Bool)
    return diff(unfoldedeigvals(E, deg=deg, goe=goe))
end


function numberVariance(E::Array, l::Float64)
    n = length(E)
    E = unfoldedeigvals(E)
    stat = 0
    for (i, e) in enumerate(E)
        n_e_l = 0
        temp1, temp2 = i - 1, i + 1
        while temp1 >= 1 && E[temp1] >= e - l / 2
            temp1 -= 1
            n_e_l += 1
        end
        while temp2 <= n && E[temp2] <= e + l / 2
            temp2 += 1
            n_e_l += 1
        end
        stat += (n_e_l - l)^2
    end

    return stat / n
end