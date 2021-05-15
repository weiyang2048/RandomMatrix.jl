export wignersurmise, unfoldedSpacings

function wignersurmise(x::Real; beta = 1 :: Int)  
    if beta == 1
        return (pi/2) * x * exp(-pi * x^2 / 4)
    elseif beta == 2
        return (3/pi)^2 * x^2 * exp(- 4 * x^2 / pi)
    end
end

function unfoldedSpacings(E::Array,deg=3::Int; goe = false::Bool)
    if goe
        f(e) = begin
            temp = 4-e^2
            if temp>=0
                length(E)* ( 1/2 + e/(pi*4)*sqrt(temp)+1/pi*atan(e/sqrt(temp)) )  
            end      
        end
        return diff(filter(x->x!==nothing,f.(E)))
    end
    p(x) =polyfit(E[1:end-1],diff(E),deg)(x)
        
    D = p.(E[1:end-1])
    diff(E)./D 
end

