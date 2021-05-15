export wignersurmise, unfold

function wignersurmise(x::Real; beta = 1 :: Int)  
    if beta == 1
        return (pi/2) * x * exp(-pi * x^2 / 4)
    elseif beta == 2
        return (3/pi)^2 * x^2 * exp(- 4 * x^2 / pi)
    end
end

function unfold(E::Array,deg=3::Int;type=:nothing ::Symbol)
    f(x) = begin
        if sym == :GOE
            return wignersurmise(x)
        elseif sym == :GUE
            return wignersurmise(x,beta=2)
        else
            return polyfit(E[1:end-1],diff(E),deg)(x)
        end
    end

    D = f.(E[1:end-1])
    diff(E)./D 
end

