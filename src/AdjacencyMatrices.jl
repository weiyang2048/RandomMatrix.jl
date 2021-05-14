export randErdosRenyi

function randErdosRenyi(n::Int, p = 0.5::Float64 )
    randHermitian(Bernoulli(p),n)
end

