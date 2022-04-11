export normview, plotEigStats

"""
```julia
# Example

# The resolvent of a random Hermitian matrix is approximately diagonal, 
# see Section 3 in https://arxiv.org/pdf/1903.10060.pdf
 normview(resolvent(randHermitian(1000,norm=true))(0.5+0.1im))
```
"""
function normview(M::AbstractMatrix)
    M = norm.(M)
    heatmap(M,
        yflip=true,
        c=cgrad(:binary, 50, categorical=true),
        ratio=1,
        xlims=(0.5, size(M)[1] + 0.5), ylims=(0.5, size(M)[2] + 0.5)
    )
end

function poissonPMF(k)
    return exp(-k)
end

function plotEigStats(M::AbstractMatrix)

    E = eigvals(M)
    S = unfoldedSpacings(E, deg=10)
    begin
        layout = @layout [grid(1, 2)]
        plot(layout=layout)
        histogram!(E, subplot=2, title="Eigenvalues", legend=false, normed=true)
        histogram!(S, subplot=1, nbins=100, title="Unfolded Spacings", normed=true, label="histogram")
        plot!(0:0.1:2, wignersurmise.(0:0.1:2), subplot=1, label="Wigner Surmise")
        plot!(0:0.1:3, poissonPMF.(0:0.1:2), subplot=1, label="Exponential(1)")
    end
end



