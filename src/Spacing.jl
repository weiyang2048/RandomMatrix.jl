export wignersurmise

function wignersurmise(x::Real, beta = 1 :: Int)  
    if beta == 1
        return (pi/2) * x * exp(-pi * x^2 / 4)
    elseif beta == 2
        return (3/pi)^2 * x^2 * exp(- 4 * x^2 / pi)
    end
end