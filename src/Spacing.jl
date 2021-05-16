export wignersurmise,unfoldedeigvals ,unfoldedSpacings

function wignersurmise(x::Real; beta = 1 :: Int)  
    if beta == 1
        return (pi/2) * x * exp(-pi * x^2 / 4)
    elseif beta == 2
        return (3/pi)^2 * x^2 * exp(- 4 * x^2 / pi)
    end
end

function unfoldedeigvals(E::Array; deg=3::Int, goe = false::Bool)
    if goe
        return length(E)*cdf(Semicircle(2),E)
    end
    
    return polyfit(E,(x->sum(E.<x)).(E),deg).(E)

end
function unfoldedSpacings(E::Array;deg=3::Int, goe = false::Bool)
    return diff(unfoldedeigvals(E,deg=deg,goe=goe))
end
