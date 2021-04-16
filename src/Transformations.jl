export resolvent, qresolvent


function resolvent(A::Matrix)
    return function f(z::Number)
    (A-z*I)^(-1)
end
end

function qresolvent(A::Matrix)
    return function f(z::Number,η::Number)
    ([-η*I A.-z 
    A'.-z' -η*I])^(-1)
end
end