export RandomHermitianMatrix


function RandomHermitianMatrix(d::Distribution{Univariate}, n::Int)
    M = zeros(n,n)
    for i in 1:n, j in 1:i-1
        M[i,j] = rand(d)
        M[j,i] = M[i,j]'
    end
    for i in 1:n
        M[i,j] = rand(d)
    return M
end

function RandomHermitianMatrix(n::Int)
    RandomHermitianMatrix(Normal(), n)
end
