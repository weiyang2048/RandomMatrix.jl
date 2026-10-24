module RandomMatrixPlotsExt

using RandomMatrix
using RandomMatrix: unfoldedSpacings, wignersurmise, numberVariance
using Plots: heatmap, cgrad, @layout, histogram!, grid, plot, plot!
using LinearAlgebra: eigvals, norm

"""
```julia
# Example

# The resolvent of a random Hermitian matrix is approximately diagonal,
# see Section 3 in https://arxiv.org/pdf/1903.10060.pdf
 normview(Resolvent(randHermitian(1000,norm=true))(0.5+0.1im))
```
"""
function RandomMatrix.normview(M::AbstractMatrix)
    M = norm.(M)
    heatmap(M,
        yflip=true,
        c=cgrad(:binary, 50, categorical=true),
        ratio=1,
        xlims=(0.5, size(M)[1] + 0.5), ylims=(0.5, size(M)[2] + 0.5)
    )
end

poissonPMF(k) = exp(-k)

function RandomMatrix.plotEigStats(M::AbstractMatrix)
    n = size(M)[1]
    E = eigvals(M)
    S = unfoldedSpacings(E, deg=10)
    begin
        layout = @layout [grid(1, 3)]
        plot(layout=layout, size=(1800, 600))
        histogram!(E, subplot=2, nbins=Int(floor(sqrt(n))), title="Eigenvalues", legend=false, normed=true,
            opacity=0.5, color=:tomato)
        histogram!(S, subplot=1, nbins=100, title="Unfolded Spacings", normed=true, label="histogram",
            opacity=0.5, color=:royalblue)
        plot!(0:0.1:3, wignersurmise.(0:0.1:3), subplot=1, label="Wigner Surmise",
            lw=5)
        plot!(0:0.1:3, poissonPMF.(0:0.1:3), subplot=1, label="Exponential(1)",
            lw=5, color=:seagreen)

        numVar = [numberVariance(E, l) for l in 0:0.01:10]
        plot!(0:0.01:10, subplot=3, numVar, title="Number Variance", legend=false,
            lw=5, color=:forestgreen)
    end
end

end # module
