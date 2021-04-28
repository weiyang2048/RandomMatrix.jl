export preNORTA

"""
- `d` : pair distribution
- `ρ` : entries
- `preNORTA` : Look for a `phat` that when used in NORTA will provided  
- NORTA method for generating Random Vectors with Arbitrary Marginal Distribution 
    fand CorrelationMatrices first appears in the following paper
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.281&rep=rep1&type=pdf
    The generated vector entries usually have a smaller correlation than what is aimed for.
"""
struct preNORTA 
    d::T where T<:Union{Distribution{Univariate},DataType,AbstractArray, Tuple}
    ρ::Float64
    ρhat::Float64
    function preNORTA(d,ρ,ρhat=ρ)
        L , H = minmax(sign(ρ)*0.999,0) # Lower bound, Higher bound
        while true
            N = [rand(MvNormal([1 ρhat;ρhat 1])) for i=1:10000]
            X = []
            Y = []

            for i in N
               append!(X,quantile(d,cdf(Normal(),i[1]))) 
               append!(Y,quantile(d,cdf(Normal(),i[2]))) 
            end
             
            temp = cor(X,Y)
            if abs(temp-ρ) <0.001 || H - L <0.001
                break
            end
            if temp < ρ
                ρhat, L = (ρhat+H)/2 ,  (L+ρhat)/2
            else
                ρhat, H = (ρhat+L)/2 , (H+ρhat)/2
            end
        end
        new(d,ρ,ρhat)
    end
end 