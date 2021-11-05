var documenterSearchIndex = {"docs":
[{"location":"ScalarValued/#Distributions","page":"Scarlar-Valued Distributions","title":"Distributions","text":"","category":"section"},{"location":"ScalarValued/#Marchenko-Pastur","page":"Scarlar-Valued Distributions","title":"Marchenko-Pastur","text":"","category":"section"},{"location":"ScalarValued/","page":"Scarlar-Valued Distributions","title":"Scarlar-Valued Distributions","text":"MarchenkoPastur","category":"page"},{"location":"ScalarValued/#RandomMatrix.MarchenkoPastur","page":"Scarlar-Valued Distributions","title":"RandomMatrix.MarchenkoPastur","text":"MarchenkoPastur <:ContinuousUnivariateDistribution\n\nMarchenkoPastur(λ,σ) \npdf(d::MarchenkoPastur,x::Real)  \n\nλ : default 0.5\nσ : default 1\n\nExamples\n\nGenerate a MP rv with ρ = 0.5, σ = 1\n\nrand(MarchenkoPastur())\n\n1.2656480923753979\n\nCompute the desity for the MP distribution with λ=1.6 at the point 0\n\npdf(MarchenkoPastur(1.6),0)\n\n0.375\n\nGenerate 100 MP rvs with ρ=0.1 and σ = 2\n\nrand(MarchenkoPastur(0.1,2),100)\n\n100-element Vector{Float64}:\n 5.31001367107032\n 2.3707745658317116\n 2.5378523986772343\n 2.585256212138476\n ⋮\n 3.66799182506567\n 6.517865226831382\n 3.7628212250212423\n 4.582520400697299\n\n\n\n\n\n","category":"type"},{"location":"ScalarValued/#Circular-Law","page":"Scarlar-Valued Distributions","title":"Circular Law","text":"","category":"section"},{"location":"ScalarValued/","page":"Scarlar-Valued Distributions","title":"Scarlar-Valued Distributions","text":"Circular","category":"page"},{"location":"ScalarValued/#RandomMatrix.Circular","page":"Scarlar-Valued Distributions","title":"RandomMatrix.Circular","text":"Circular(c=0,R=1)\n\nThe uniform distribution on the complex disk with center c and radius R\n\nrand(Circular(1,10),200)\n\nGenerates 200 uniform r.v.s on the complex disk of center 1 radius 10.\n\n\n\n\n\n","category":"type"},{"location":"ScalarValued/#Elliptic-Law","page":"Scarlar-Valued Distributions","title":"Elliptic Law","text":"","category":"section"},{"location":"ScalarValued/","page":"Scarlar-Valued Distributions","title":"Scarlar-Valued Distributions","text":"Elliptic","category":"page"},{"location":"ScalarValued/#RandomMatrix.Elliptic","page":"Scarlar-Valued Distributions","title":"RandomMatrix.Elliptic","text":"Elliptic <:ContinuousUnivariateDistribution\n\nElliptic(ρ;c,R)\n\nthe uniform rv on an ellipse of width 2(1+ρ)R, height 2(1-ρ)R centered at c\n\n\n\n\n\n","category":"type"},{"location":"Others/","page":"Under Construction1","title":"Under Construction1","text":"\nqresolvent\n\n\nComplexNormal\n\nGaussian\n\n±\n\npreNORTA\n\nnormview\n\nrandSampling","category":"page"},{"location":"Others/#RandomMatrix.qresolvent","page":"Under Construction1","title":"RandomMatrix.qresolvent","text":"qresolvent(A)\n\nreturns the quaternion resolvent function of A.\n\n\n\n\n\n","category":"function"},{"location":"Others/#RandomMatrix.ComplexNormal","page":"Under Construction1","title":"RandomMatrix.ComplexNormal","text":"ComplexNormal(μ,σ)\n\nμ : mean, μ = 0 by default\nσ : standard deviation, σ  =1 by default\n\n# Examples\n\n# Generates 10 iid standard Complex Gaussian r.v.s\nrand(ComplexNormal(), 10) \n\n# Generates complex normal with mean 1+1im, variance 4\nrand(ComplexNormal(1+1im,2)) \n\n\n\n\n\n","category":"type"},{"location":"Others/#RandomMatrix.Gaussian","page":"Under Construction1","title":"RandomMatrix.Gaussian","text":"Gaussian(beta,μ,σ)\n\nbeta : 1 (default) for Real Gaussian, 2 for Complex Gaussian \nμ : mean, μ = 0 by default\nσ : standard deviation, σ  =1 by default\n\n# Generates 10 iid standard normal r.v.s\nrand(Gaussian(), 10) \n\n# Generates a complex normal with mean 1+1im, variance 4\nrand(Gaussian(2,1+1im,2))\n\n\n\n\n\n","category":"type"},{"location":"Others/#RandomMatrix.:±","page":"Under Construction1","title":"RandomMatrix.:±","text":"±(a,b)\n\nreturns (a-b,a+b)\n\n# Examples\n\n1 ± 0.5  # returns (0.5,1.5)\n\n\n\n\n\n","category":"function"},{"location":"Others/#RandomMatrix.normview","page":"Under Construction1","title":"RandomMatrix.normview","text":"# Example\n\n# The resolvent of a random Hermitian matrix is approximately diagonal, \n# see Section 3 in https://arxiv.org/pdf/1903.10060.pdf\n normview(resolvent(randHermitian(1000,norm=true))(0.5+0.1im))\n\n\n\n\n\n","category":"function"},{"location":"Others/#RandomMatrix.randSampling","page":"Under Construction1","title":"RandomMatrix.randSampling","text":"randSampling(A::Matrix, B=I::Matrix; k=0::Int)\n\nGenerate a random sampling random Matrix S, E(SS')=I. ASS'B is typically used to approximate AB.\n\nExamples\n\nS = randSampling(rand(2,3),rand(3,2),k=2)\n\n3×2 Matrix{Float64}:\n 0.0      0.0\n 0.0      1.2439\n 1.15342  0.0\n\n\n\n\n\n","category":"function"},{"location":"IID Matrices/#Matrices-with-IID-Entries","page":"I.I.D. Matrices","title":"Matrices with IID Entries","text":"","category":"section"},{"location":"IID Matrices/#Diagonal-Matrices","page":"I.I.D. Matrices","title":"Diagonal Matrices","text":"","category":"section"},{"location":"IID Matrices/","page":"I.I.D. Matrices","title":"I.I.D. Matrices","text":"randDiagonal","category":"page"},{"location":"IID Matrices/#RandomMatrix.randDiagonal","page":"I.I.D. Matrices","title":"RandomMatrix.randDiagonal","text":"randDiagonal(d, n) \n\nrandDiagonal(n) \n\nd : default Normal(0，1), entry distribution\nn : dimension\n\nExamples\n\nGenerates a 3 by 3 diagonal matrix, with non-zero elements from Normal(0,1)\n\nrandDiagonal(3)\n\n3×3 Diagonal{Float64, Vector{Float64}}:\n 0.440359   ⋅         ⋅\n  ⋅        1.94832    ⋅\n  ⋅         ⋅       -0.52536\n\nGenerates a 5 by 5 diagonal matrix, with non-zero elements from Poisson(2)\n\nrandDiagonal(Poisson(2),5)\n\n5×5 Diagonal{Int64, Vector{Int64}}:\n 1  ⋅  ⋅  ⋅  ⋅\n ⋅  0  ⋅  ⋅  ⋅\n ⋅  ⋅  0  ⋅  ⋅\n ⋅  ⋅  ⋅  3  ⋅\n ⋅  ⋅  ⋅  ⋅  3\n\n\n\n\n\n","category":"function"},{"location":"IID Matrices/#Triangular-Matrices","page":"I.I.D. Matrices","title":"Triangular Matrices","text":"","category":"section"},{"location":"IID Matrices/","page":"I.I.D. Matrices","title":"I.I.D. Matrices","text":"randTriangular","category":"page"},{"location":"IID Matrices/#RandomMatrix.randTriangular","page":"I.I.D. Matrices","title":"RandomMatrix.randTriangular","text":"randTriangular(d , n ;  diag , Diag, upper ) \n\nrandTriangular(n;diag, upper)\n\nd : entry distribution\nn : dimension\ndiag : default diag = d, diagonal entry distribution\nDiag : default Diag = true, true includes diagonal, false with diagonal entries 0\nupper : default upper = true, true gives upper triangular, false gives lower triangular\n\nExamples\n\nGenerate an upper triangular matrix with entries Standard Normal\n\nrandTriangular(3)\n\n3×3 UpperTriangular{Float64, Matrix{Float64}}:\n -0.572757  -0.459518   -1.60622\n   ⋅         0.0216834  -0.416529\n   ⋅          ⋅         -1.00807\n\nGenerate a 3 by 3 strictly lower triangular matrix, with nonzero entries uniform from 123 \n\nrandTriangular(1:3,3,upper=false,Diag=false)\n\n3×3 LowerTriangular{Int64, Transpose{Int64, Matrix{Int64}}}: \n 0  ⋅  ⋅\n 3  0  ⋅\n 3  2  0\n\n\n\n\n\n","category":"function"},{"location":"IID Matrices/#Full-Matrices","page":"I.I.D. Matrices","title":"Full Matrices","text":"","category":"section"},{"location":"IID Matrices/","page":"I.I.D. Matrices","title":"I.I.D. Matrices","text":"randMatrix","category":"page"},{"location":"IID Matrices/#RandomMatrix.randMatrix","page":"I.I.D. Matrices","title":"RandomMatrix.randMatrix","text":"randMatrix(d::D, n::Int, m = n::Int; norm = false::Bool) where D<:S\n\nrandMatrix(n::Int, m = n::Int; norm = false::Bool)\n\nd : entry distribution\nn,m : default m = n , dimensions\nnorm : default false, if norm set to true, then the matrix will be normlaized with operatornamemin(nm)^-12.  \n\nExamples\n\nGenerates a 2 by 2 random  matrix with entries from the Standard  Gaussian.\n\nrandMatrix(2)\n\n2×2 Matrix{Float64}:\n 1.74043  -1.30317\n 0.72765   0.639943\n\nGenerates a 3 by 2 random  matrix with entries uniformly from {1,2,3,...,10}.\n\nrandMatrix(1:10,3,2)\n\n3×2 Matrix{Int64}:\n  1  3\n  6  4\n 10  1\n\nGenerate a normalized random 2 by 2  Matrix with entries  Poisson(2) rvs.  Need to import the Distributions package for Poisson(2)\n\nusing Distributions\nrandMatrix(Poisson(2),2,norm = true)\n\n2×2 Matrix{Float64}:\n 1.41421   0.0\n 0.707107  1.41421\n\n\n\n\n\n","category":"function"},{"location":"IID Matrices/#RMT:-Circular-Law","page":"I.I.D. Matrices","title":"RMT: Circular Law","text":"","category":"section"},{"location":"IID Matrices/","page":"I.I.D. Matrices","title":"I.I.D. Matrices","text":"Let left(X_nright)_n=1^infty be a sequence of n times n matrix ensembles whose entries are i.i.d. copies of a complex random variable x with mean 0 and variance 1. Let lambda_1 ldots lambda_n 1 leq j leq n denote the eigenvalues of frac1sqrtn X_n. Define the empirical spectral measure of frac1sqrtn X_n as","category":"page"},{"location":"IID Matrices/","page":"I.I.D. Matrices","title":"I.I.D. Matrices","text":"mu_frac1sqrtn X_n(A)=n^-1 leftj leq n lambda_j in Aright quad A in mathcalB(mathbbC)","category":"page"},{"location":"IID Matrices/","page":"I.I.D. Matrices","title":"I.I.D. Matrices","text":"The circular law asserts that almost surely (i.e. with probability one), the sequence of measures mu frac1sqrtn X_n converges in distribution to the uniform measure on the unit disk.  ","category":"page"},{"location":"IID Matrices/","page":"I.I.D. Matrices","title":"I.I.D. Matrices","text":"For reference, see for example the paper by Terence Tao and Van Vu: RANDOM MATRICES: THE CIRCULAR LAW","category":"page"},{"location":"IID Matrices/","page":"I.I.D. Matrices","title":"I.I.D. Matrices","text":"using Plots, RandomMatrix, LinearAlgebra\ngr() # hide\nPlots.reset_defaults() # hide\nN = 500\nM = randMatrix(N)\ncolors = [:red,:green,:blue,:purple]\nanim = @animate for n = (1:50...,51:10:N...,N:-10:51...,50:1...)\n     \n    M[1:n,1:n]/sqrt(n)|>eigvals|>x->scatter(x,ylims=(-1.25,1.25), xlims=(-1.25,1.25),ratio=1,label=\"n = $(n)\",size=(600,600))\n\n    plot!([exp(θ*im) for θ=0:0.01:2pi],label=\"\",lw=3,c=[rand(colors) for _=0:0.01:2pi])\n        \n    title!(\"Circular Law for IID Matrices\")\n    \nend \ngif(anim, \"IID1.gif\", fps = 10); nothing","category":"page"},{"location":"IID Matrices/","page":"I.I.D. Matrices","title":"I.I.D. Matrices","text":"(Image: )","category":"page"},{"location":"Correlated Matrices/#Under-construction","page":"Correlated Matrices","title":"Under construction","text":"","category":"section"},{"location":"Correlated Matrices/#Hermitian","page":"Correlated Matrices","title":"Hermitian","text":"","category":"section"},{"location":"Correlated Matrices/","page":"Correlated Matrices","title":"Correlated Matrices","text":"randHermitian","category":"page"},{"location":"Correlated Matrices/#RandomMatrix.randHermitian","page":"Correlated Matrices","title":"RandomMatrix.randHermitian","text":"randHermitian(d, n; diag, norm )\n\nrandHermitian(n; norm)\n\nd : entry distribution\nn  : dimensions \nnorm : default false, if norm set to true, then the matrix will be normalized with n^-12.  \ndiag : default diag = d, diagonal entry distribution.    To use a different distribution (say Circular(2)) for digonal elements, set diag = Circular(2).     The diagonal entries will always be forced to have imgainary part 0.\nSee also GUE\n\nExamples\n\nGenerates a 2 by 2 random Hermitian matrix with off-diagonal entries from the Standard Complex Gaussian, and Standard Normal on the diagonal.\n\nrandHermitian(2)\n\n2×2 Hermitian{ComplexF64, Matrix{ComplexF64}}:\n  0.382095+0.0im        -0.708469-0.0636734im\n -0.708469+0.0636734im   0.336952+0.0im\n\nGenerate a 3 by 3 Hermitian matrix, with off-diagonal entries Circular(1) and diagonal entries uniformly -1 or 1.\n\nrandHermitian(Circular(1),3,diag = (-1,1))\n\n3×3 Hermitian{ComplexF64, Matrix{ComplexF64}}:\n     1.0+0.0im       1.56259-0.676099im  1.39468-0.295073im\n 1.56259+0.676099im     -1.0+0.0im       1.53369+0.296583im\n 1.39468+0.295073im  1.53369-0.296583im     -1.0+0.0im\n\nGenerate a random 2 by 2 Symmetric Matrix with entries  Poisson(2) rvs.  This is also be done with randSymmetric(Poisson(2),3)\n\nrandHermitian(Poisson(2),3)\n\n3×3 Hermitian{Int64, Matrix{Int64}}:\n 3  1  0\n 1  1  2\n 0  2  1\n\nEntries uniformly from 12310\n\nrandHermitian(1:10,2)\n\n2×2 Hermitian{Int64, Matrix{Int64}}:\n 10  7\n  7  6\n\nEntries either -1 or pi with equal probability\n\nrandHermitian([-1,pi],3)\n\n3×3 Hermitian{Float64, Matrix{Float64}}:\n  3.14159  -1.0      -1.0\n -1.0      -1.0       3.14159\n -1.0       3.14159   3.14159\n\n\n\n\n\n","category":"function"},{"location":"Correlated Matrices/#Symmetric","page":"Correlated Matrices","title":"Symmetric","text":"","category":"section"},{"location":"Correlated Matrices/","page":"Correlated Matrices","title":"Correlated Matrices","text":"randSymmetric","category":"page"},{"location":"Correlated Matrices/#RandomMatrix.randSymmetric","page":"Correlated Matrices","title":"RandomMatrix.randSymmetric","text":"randSymmetric(d, n; Diag, norm) \n\nrandSymmetric(n; norm)\n\nd : entry distribution\nn : dimensions \nnorm : default  false, if norm set to true, then the matrix will be normalized with n^-12.   \ndiag : default diag = d, the distribution for diagonal entries.    To use a different distribution (say Binomial) for digonal elements, set diag = Binomial(1,0.5)\nSee also GOE\n\nExamples\n\nGenerates a 3 by 3 random Symmetric matrix with entries from the Standard Gaussian.\n\nrandSymmetric(3)\n\n3×3 Symmetric{Float64, Matrix{Float64}}:\n -0.230698  -1.72846     0.306362\n -1.72846    0.0845915  -0.0116108\n  0.306362  -0.0116108  -0.559046\n\n\n\n\n\n","category":"function"},{"location":"Correlated Matrices/#Elliptic-Matrices","page":"Correlated Matrices","title":"Elliptic Matrices","text":"","category":"section"},{"location":"Correlated Matrices/","page":"Correlated Matrices","title":"Correlated Matrices","text":"randEllipic","category":"page"},{"location":"Correlated Matrices/#RandomMatrix.randEllipic","page":"Correlated Matrices","title":"RandomMatrix.randEllipic","text":"randEllipic(d, n; r , diag, norm) \n\nrandEllipic(n; r, norm)\n\nd : default Normal(), entry distribution\nn  : dimensions \nr : default 0.5, the correlation of H_ijH_ji pairs\nnorm : default false, if norm set to true, then the matrix will be normalized with n^-12.  \ndiag : default diag=d, the distribution for diagonal entries.\n\nExamples\n\nGenerate a random elliptic matrix, with entries from mathscrN(01) and rho(H_ijH_ji)=05 \n\nrandEllipic(500)\n\n500×500 Matrix{Float64}:\n  2.03417    -0.424289    1.28267   …  -0.114754  -1.96059\n  0.44479    -1.45563     1.32828       1.00149    0.45786\n  1.56525     0.0832211  -0.186738     -1.3914     1.04151\n -0.11633    -0.483301   -1.81348      -1.57536    0.514818\n  ⋮                                 ⋱\n  1.24274    -0.411623   -1.04984      -0.812778  -1.84479\n -0.0817287  -0.254886    0.674914      0.756269  -0.0296209\n -1.48281     0.51675    -1.58041       0.156923   0.244599\n  0.852339    1.04593    -0.119082      1.43634    0.114493\n\nGenerate a normalized random elliptic matrix, with entries Poisson(10) and rho(H_ijH_ji)=01 \n\nusing Distributions\nrandEllipic(Poisson(10),500, r=0.1 , norm=true)\n\n500×500 Matrix{Float64}:\n  0.268328    -0.0413153  -0.0175096   …   0.0190835   0.0201304       \n -0.00599949   0.447214   -0.0175878       0.0112805   0.100704        \n -0.0258879    0.0219927   0.402492       -0.04749    -0.050853        \n  0.0219071    0.0119609  -0.00448502      0.0043233   0.0404757       \n  ⋮                                    ⋱\n -0.0145467    0.0800297   0.00247891      0.0189267   0.071565        \n  0.016412     0.0334019   0.0663348      -0.0180889   0.023773        \n -0.0485914   -0.0575288  -0.0409827       0.491935   -0.0969691       \n  0.0405447    0.0503843   0.00624668      0.0558304   0.402492        \n\n\n\n\n\n\n","category":"function"},{"location":"Correlated Matrices/#RMT:-Semicircle-Law","page":"Correlated Matrices","title":"RMT: Semicircle Law","text":"","category":"section"},{"location":"Correlated Matrices/","page":"Correlated Matrices","title":"Correlated Matrices","text":"The semi-circular law, lecture notes by Terence Tao","category":"page"},{"location":"Correlated Matrices/","page":"Correlated Matrices","title":"Correlated Matrices","text":"using Plots, RandomMatrix, LinearAlgebra, Distributions\nN = 2000\ncolors = [:red,:green,:blue,:purple]\nanim = @animate for n = (1:2:50...,51:20:N...,N:-20:51...,50:-2:1...)\n    M = randHermitian(Normal(0,1), n,norm=true) \n    histogram(eigvals(M), normed=true, nbins=100,ratio = 7, label=\"n=$n\")\n    plot!(x->1/2pi*sqrt(4-x^2), -2:0.001:2, linewidth = 5, label=\"Semicircle\",c=[rand(colors) for _=-2:0.001:2])\n    xlims!(-2.5,2.5)\n    ylims!(0,0.5)\n    title!(\"Semicircular Law for Random Hermitian Matrices\")\nend\n\ngif(anim,\"Semicircle.gif\",fps=20); nothing","category":"page"},{"location":"Correlated Matrices/","page":"Correlated Matrices","title":"Correlated Matrices","text":"(Image: )","category":"page"},{"location":"Correlated Matrices/#RMT-:-Elliptic-Law","page":"Correlated Matrices","title":"RMT : Elliptic Law","text":"","category":"section"},{"location":"Correlated Matrices/","page":"Correlated Matrices","title":"Correlated Matrices","text":"THE ELLIPTIC LAW by NGUYEN and O’ROURKE","category":"page"},{"location":"Correlated Matrices/","page":"Correlated Matrices","title":"Correlated Matrices","text":"\nusing Plots, RandomMatrix, LinearAlgebra, Distributions\ngr()  \nPlots.reset_defaults() \nN = 600\n\ncolors = [:red,:green,:blue,:purple]\n \nanim = @animate for n = (1:50...,51:10:N...,N:-10:51...,50:1...)\n    \nrandEllipic(500,r=((N-2*n)/N)+0.0001,norm=true)|>eigvals|>x->scatter(x,ylims=(-2,2), xlims=(-2,2),ratio=1,label=\"ρ = $(round(((N-2*n)/N)+0.0001,digits=2))\")\n\ntitle!(\"Elliptical Law\")\n\nend \ngif(anim, \"Elliptic.gif\", fps = 10); nothing","category":"page"},{"location":"Correlated Matrices/","page":"Correlated Matrices","title":"Correlated Matrices","text":"(Image: )","category":"page"},{"location":"Stochastic/#Stochastic-Matrices","page":"Stochastic Matrices","title":"Stochastic Matrices","text":"","category":"section"},{"location":"Stochastic/#Stochastic-Matrices-2","page":"Stochastic Matrices","title":"Stochastic Matrices","text":"","category":"section"},{"location":"Stochastic/","page":"Stochastic Matrices","title":"Stochastic Matrices","text":"randStochastic","category":"page"},{"location":"Stochastic/#RandomMatrix.randStochastic","page":"Stochastic Matrices","title":"RandomMatrix.randStochastic","text":"randStochastic(n; type, norm)\n\nn: dimension\ntype : default type = 3, 3 for doubly randStochastic, 1 for row, 2 for column stochastic \nnorm : default false, if set to true, the matrix will be normalized by sqrtn (not a typo)\n\nExamples\n\nGenerates a 3 by 3 random doubly stochastic  matrix\n\nrandStochastic(3)\n\n3×3 Matrix{Float64}:\n 0.132593  0.216041  0.651367\n 0.484097  0.320777  0.195126\n 0.261495  0.537825  0.20068\n\nGenerates a 3 by 3 normalized random row stochastic  matrix\n\nrandStochastic(3,type = 1)\n\n3×3 Matrix{Float64}:\n 0.220849   0.146942  0.632209\n 0.188052   0.26294   0.549008\n 0.0170714  0.524574  0.458355\n\nGenerates a 3 by 3 normalized random column stochastic  matrix\n\nrandStochastic(3,type=2,norm=true)\n\n3×3 Matrix{Float64}:\n 0.583396  0.608739  0.732921\n 0.672821  0.078786  0.302657\n 0.475834  1.04453   0.696473\n\n\n\n\n\n","category":"function"},{"location":"Stochastic/#RMT:-Circular-Law-for-Doubly-Stochastic-Random-Matrices","page":"Stochastic Matrices","title":"RMT: Circular Law for Doubly Stochastic Random Matrices","text":"","category":"section"},{"location":"Stochastic/","page":"Stochastic Matrices","title":"Stochastic Matrices","text":"Let X be a matrix sampled uniformly from the set of doubly stochastic matrices of size n times n. The empirical spectral distribution of the normalized matrix sqrtn(X-mathbfE X) converges almost surely to the circular law. ","category":"page"},{"location":"Stochastic/","page":"Stochastic Matrices","title":"Stochastic Matrices","text":"For reference, see the paper by Hoi H. Nguyen Random doubly stochastic matrices: The circular law","category":"page"},{"location":"Stochastic/","page":"Stochastic Matrices","title":"Stochastic Matrices","text":"using Plots, RandomMatrix, LinearAlgebra\ngr() # hide\nPlots.reset_defaults() # hide\nN = 600\ncolors = [:red,:green,:blue,:purple]\nani = @animate for n = (1:50...,51:10:N...,N:-10:51...,50:1...)\n     \n    randStochastic(n,norm=true)|>eigvals|>x->scatter(x,ratio=1,xlims=(-1.25,1.25),title=\"Circular Law for Doubly Stochastic Matrices\",size=(600,600),label = \"n = $(n)\")\n\n    plot!([exp(θ*im) for θ=0:0.01:2pi],label=\"\",lw=3,c=[rand(colors) for _=0:0.01:2pi])\n        \nend \ngif(ani, \"St1.gif\", fps = 10); nothing","category":"page"},{"location":"Stochastic/","page":"Stochastic Matrices","title":"Stochastic Matrices","text":"(Image: )","category":"page"},{"location":"MatrixValued/#Matrix-valued-Distributions","page":"Matrix-Valued Distributions","title":"Matrix-valued Distributions","text":"","category":"section"},{"location":"MatrixValued/#GUE","page":"Matrix-Valued Distributions","title":"GUE","text":"","category":"section"},{"location":"MatrixValued/","page":"Matrix-Valued Distributions","title":"Matrix-Valued Distributions","text":"GUE","category":"page"},{"location":"MatrixValued/#RandomMatrix.GUE","page":"Matrix-Valued Distributions","title":"RandomMatrix.GUE","text":"GUE <: ContinuousMatrixDistribution\nGUE(n)\n\nn : dimension\nThe Gaussian Unitary Ensemble (GUE) is an ensemble of random n times n Hermitian matrices    M_n in which the upper-triangular entries are iid with distribution N(01)_mathbfC,    and the diagonal entries are iid with distribution N(01)_mathbfR, and independent of the upper-triangular ones\n\nrand(M::GUE, norm::bool) \n\nnorm : default false,  if norm set to true, then the matrix will be normlaized with operatornamemin(nm)^-12.  \n\nExamples\n\nGenerate a 3 by 3 random matrix from GUE(3)\n\nrand(GUE(3))\n\n3×3 Hermitian{ComplexF64, Matrix{ComplexF64}}:\n -0.883413+0.0im         1.09872+0.874884im     -0.1985-1.04778im\n   1.09872-0.874884im    1.55483+0.0im        -0.488532+1.18694im\n   -0.1985+1.04778im   -0.488532-1.18694im   -0.0823873+0.0im\n\nrand(GUE(2),norm=true)\n2×2 Hermitian{ComplexF64, Matrix{ComplexF64}}:\n -0.457089+0.0im       -0.672713-0.102234im\n -0.672713+0.102234im   0.380126+0.0im\n\n\n\n\n\n","category":"type"},{"location":"MatrixValued/#GOE","page":"Matrix-Valued Distributions","title":"GOE","text":"","category":"section"},{"location":"MatrixValued/","page":"Matrix-Valued Distributions","title":"Matrix-Valued Distributions","text":"GOE","category":"page"},{"location":"MatrixValued/#RandomMatrix.GOE","page":"Matrix-Valued Distributions","title":"RandomMatrix.GOE","text":"GOE <: ContinuousMatrixDistribution\nGOE(n) \n\nn : dimension\nThe Gaussian Orthogonal Ensemble (GOE) is an ensemble of random n times n Symmetric matrices    M_n in which the upper-triangular entries are iid with distribution N(01)_mathbfR,    and the diagonal entries are iid with distribution N(02)_mathbfR, and independent of the upper-triangular ones\n\nrand(M::GOE, norm::bool)\n\nnorm : default false,  if norm set to true, then the matrix will be normlaized with operatornamemin(nm)^-12.  \n\nExamples\n\nGenerate a 3 by 3 random matrix from GOE(3)\n\nrand(GOE(3))\n\n3×3 Symmetric{Float64, Matrix{Float64}}:\n -1.62391   -0.451433    0.863883\n -0.451433   0.0271799  -0.524854\n  0.863883  -0.524854   -0.00930624\n\nrand(GOE(3),norm=true)\n\n3×3 Symmetric{Float64, Matrix{Float64}}:\n  0.302141   0.152634   -0.711679\n  0.152634  -0.0629327   0.103075\n -0.711679   0.103075    1.51861\n\n\n\n\n\n","category":"type"},{"location":"MatrixValued/#Haar","page":"Matrix-Valued Distributions","title":"Haar","text":"","category":"section"},{"location":"MatrixValued/","page":"Matrix-Valued Distributions","title":"Matrix-Valued Distributions","text":"Haar","category":"page"},{"location":"MatrixValued/#RandomMatrix.Haar","page":"Matrix-Valued Distributions","title":"RandomMatrix.Haar","text":"Haar(beta,n)\n\nUniform distribution on O(n) (beta = 1), U(n) (beta = 2)\nbeta: 1 for Orthogonal, 2 for Unitary\nn: dimension\n\n# Examples\n\n# Generate a 100 by 100 random Unitary Matrix uniformly from U(n)\nrand(Haar(2,100))\n\n\n# Generate a 100 by 100 random Orthogonal Matrix uniformly from O(n)\nrand(Haar(1,100))\n\n\n\n\n\n","category":"type"},{"location":"MatrixValued/#COE","page":"Matrix-Valued Distributions","title":"COE","text":"","category":"section"},{"location":"MatrixValued/","page":"Matrix-Valued Distributions","title":"Matrix-Valued Distributions","text":"COE","category":"page"},{"location":"MatrixValued/#RandomMatrix.COE","page":"Matrix-Valued Distributions","title":"RandomMatrix.COE","text":"COE\n\nn : Dimension \nEquivalent to Haar(1,n)\n\n\n\n\n\n","category":"function"},{"location":"MatrixValued/#CUE","page":"Matrix-Valued Distributions","title":"CUE","text":"","category":"section"},{"location":"MatrixValued/","page":"Matrix-Valued Distributions","title":"Matrix-Valued Distributions","text":"CUE","category":"page"},{"location":"MatrixValued/#RandomMatrix.CUE","page":"Matrix-Valued Distributions","title":"RandomMatrix.CUE","text":"CUE\n\nn : Dimension \nEquivalent to Haar(2,n)\n\n\n\n\n\n","category":"function"},{"location":"Gallery/","page":"Under Construction2","title":"Under Construction2","text":"\nusing Plots, RandomMatrix, LinearAlgebra, Distributions\ngr()  \nPlots.reset_defaults() \nN = 600\nM = randMatrix(N)\nU = randUnitary(N)\nl=@layout[c c; c c]\ncolors = [:red,:green,:blue,:purple]\n \nanim = @animate for n = (1:50...,51:10:N...,N:-10:51...,50:1...)\n     \n  p1 = M[1:n,1:n]/sqrt(n)|>eigvals|>x->scatter(x,ylims=(-1.25,1.25), xlims=(-1.25,1.25),ratio=1,label=\"n = $(n)\")\n        plot!([exp(θ*im) for θ=0:0.01:2pi],label=\"\",lw=3,c=[rand(colors) for _=0:0.01:2pi])\n        title!(\"Circular Law for IID Matrices\")\n    \n    p2 = U[1:N-n÷2,1:N-n÷2]|>eigvals|>x->scatter(x,ylims=(-1.25,1.25), xlims=(-1.25,1.25),ratio=1,label=\"ratio = $(round((N-n÷3)/N,digits=2))\")\n        plot!([exp(θ*im) for θ=0:0.01:2pi],label=\"\",lw=3,c=[rand(colors) for _=0:0.01:2pi])\n        plot!(sqrt((N-n÷2)/N)*[exp(θ*im) for θ=0:0.01:2pi],label=\"\",lw=3,c=[rand(colors) for _=0:0.01:2pi],alpha=0.1)\n        title!(\"Circular Law for Truncated Unitary\")\n\n    p3 = randStochastic(n,norm=true)|>eigvals|>x->scatter(x,ratio=1,xlims=(-1.25,1.25),title=\"Circular Law for Doubly Stochastic Matrices\",size=(600,600),label = \"n = $(n)\")\n\n    plot!([exp(θ*im) for θ=0:0.01:2pi],label=\"\",lw=3,c=[rand(colors) for _=0:0.01:2pi])\n    \n    p4 = randEllipic(500,r=((N-2*n)/N)+0.0001,norm=true)|>eigvals|>x->scatter(x,ylims=(-2,2), xlims=(-2,2),ratio=1,label=\"ρ = $(round(((N-2*n)/N)+0.0001,digits=2))\")\n   title!(\"Elliptical Law\")\n  plot(p1,p2,p3,p4,size = (1000, 1000),axis=false)\nend \ngif(anim, \"testing1.gif\", fps = 10)\n; nothing","category":"page"},{"location":"Gallery/","page":"Under Construction2","title":"Under Construction2","text":"(Image: )","category":"page"},{"location":"Unitary/#under-construction","page":"Unitary Matrices","title":"under construction","text":"","category":"section"},{"location":"Unitary/#Unitary-Matrices","page":"Unitary Matrices","title":"Unitary Matrices","text":"","category":"section"},{"location":"Unitary/","page":"Unitary Matrices","title":"Unitary Matrices","text":"randUnitary","category":"page"},{"location":"Unitary/#RandomMatrix.randUnitary","page":"Unitary Matrices","title":"RandomMatrix.randUnitary","text":"randUnitary(n::Int)\n\nGenerates a n by n random Unitary matrix\nEquivalent to  rand(Haar(2,n)), see Haar and CUE\nFor orthogonal matrices, use randOrthogonal or rand(Haar(1,n)) instead\nThe algorithm is disccused in How to Generate Random Matrices from the Classical Compact Groups\n\nExamples\n\nGenerate a 3 by 3 random Unitary matrix \n\nrandUnitary(3)\n\n3×3 Matrix{ComplexF64}:\n -0.149398+0.0572715im  -0.0935861+0.629201im  -0.257255-0.709625im\n  0.337035-0.342606im     -0.36366+0.599236im  -0.100838+0.517231im\n  -0.17097+0.845103im   -0.0767105+0.313259im   0.247081+0.3025im\n\n\n\n\n\n","category":"function"},{"location":"Unitary/#Orthogonal-Matrices","page":"Unitary Matrices","title":"Orthogonal Matrices","text":"","category":"section"},{"location":"Unitary/","page":"Unitary Matrices","title":"Unitary Matrices","text":"randOrthogonal","category":"page"},{"location":"Unitary/#RandomMatrix.randOrthogonal","page":"Unitary Matrices","title":"RandomMatrix.randOrthogonal","text":"randOrthogonal(n::Int)\n\nGenerates a n by n random Orthogonal matrix\nEquivalent to rand(Haar(1,n)), see Haar and COE\nFor unitary matrices, use randUnitary or rand(Haar(2,n)) instead\nThe algorithm is disccused in How to Generate Random Matrices from the Classical Compact Groups\n\nExamples\n\nGenerates a 3 by 3 random Orthogonal matrix \n\nrandOrthogonal(3)\n\n3×3 Matrix{Float64}:\n -0.875553  0.112448   0.469853\n -0.147915  0.863441  -0.482277\n  0.459921  0.491757   0.739356\n\n\n\n\n\n","category":"function"},{"location":"Unitary/#Permutation-Matrices","page":"Unitary Matrices","title":"Permutation Matrices","text":"","category":"section"},{"location":"Unitary/","page":"Unitary Matrices","title":"Unitary Matrices","text":"randPermutation","category":"page"},{"location":"Unitary/#RandomMatrix.randPermutation","page":"Unitary Matrices","title":"RandomMatrix.randPermutation","text":"randPermutation(n; fix) \n\nn : dimension\nfix : a keyword argument, default is set to fix = 0. If fix = x, randPermutation(n,x) will have atleast x fixed points\n\nExamples\n\nGenerates a random 5 by 5 permutation matrix\n\nrandPermutation(5)\n\n5×5 SparseArrays.SparseMatrixCSC{Int8, Int64} with 5 stored entries:\n ⋅  ⋅  ⋅  ⋅  1\n ⋅  1  ⋅  ⋅  ⋅\n 1  ⋅  ⋅  ⋅  ⋅\n ⋅  ⋅  1  ⋅  ⋅\n ⋅  ⋅  ⋅  1  ⋅\n\nGenerates  Generates a random 10 by 10 permutation matrix with atleast 7 fix points\n\nrandPermutation(10, fix = 7)\n\n10×10 SparseArrays.SparseMatrixCSC{Int8, Int64} with 10 stored entries: \n 1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅\n ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅\n ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅\n ⋮              ⋮\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1  ⋅  ⋅\n ⋅  ⋅  1  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅\n ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  ⋅  1\n\n\n\n\n\n\n","category":"function"},{"location":"Spectral Spacing/#Under-Construction","page":"Spectral Spacing","title":"Under Construction","text":"","category":"section"},{"location":"Spectral Spacing/","page":"Spectral Spacing","title":"Spectral Spacing","text":"wignersurmise","category":"page"},{"location":"Spectral Spacing/#RandomMatrix.wignersurmise","page":"Spectral Spacing","title":"RandomMatrix.wignersurmise","text":"wignersurmise(s ; beta)  \n\nbeta : default 1, \n1 for the spacing density corresponds to GOE(2) \np_1(s)=fracpi s2 e^-pi s^2  4\n2 for the spacing density corresponds to GUE(2) \np_2(s)=frac32 s^2pi^2 e^-4 s^2  pi\nFor more info see Wigner surmise on Wikipedia\n\nExamples\n\nEvaluate Wigner surmise for GOE at 2\n\nwignersurmise(2)\n\n0.1357605281502967\n\nEvaluate Wigner surmise for GUE at 2\n\nwignersurmise(2,beta=2)\n\n0.07962814352725955\n\n\n\n\n\n","category":"function"},{"location":"Spectral Spacing/","page":"Spectral Spacing","title":"Spectral Spacing","text":"unfoldedeigvals","category":"page"},{"location":"Spectral Spacing/#RandomMatrix.unfoldedeigvals","page":"Spectral Spacing","title":"RandomMatrix.unfoldedeigvals","text":"unfoldedeigvals(E; deg, goe)\n\nE : The array of eigenvalues\ndeg : default 10\ngoe : default false\n\n\n\n\n\n","category":"function"},{"location":"Spectral Spacing/","page":"Spectral Spacing","title":"Spectral Spacing","text":"unfoldedSpacings","category":"page"},{"location":"matrixtransformation/#Matrix-Transformations","page":"Matrix Transformations","title":"Matrix Transformations","text":"","category":"section"},{"location":"matrixtransformation/#Resolvent","page":"Matrix Transformations","title":"Resolvent","text":"","category":"section"},{"location":"matrixtransformation/","page":"Matrix Transformations","title":"Matrix Transformations","text":"Resolvent","category":"page"},{"location":"matrixtransformation/#RandomMatrix.Resolvent","page":"Matrix Transformations","title":"RandomMatrix.Resolvent","text":"Resolvent(A::Matrix)\n\nreturn the resolvent function of A, R(z)= (A-zI)^-1\n\nExamples\n\nGet the resolvent function of a normalized GOE(500).\n\nAnd evaluate the resolvent at 0\n\nR = Resolvent(rand(GOE(500),norm = true)) # this a function\n\nR(0)\n\n500×500 Matrix{Float64}:\n  0.460903   -0.384016   0.268931  …  -0.59029    -0.178288\n -0.384016    0.489355  -0.366855      0.452094   -0.190179\n  0.268931   -0.366855  -0.697169     -0.221433    0.535448\n  0.388846    0.946668  -0.354459     -0.796717    0.207156\n  ⋮                                ⋱\n -0.532533    0.950826  -0.366618      0.0926931  -0.288692\n -0.0874979  -0.553262  -0.388048     -0.0156897   0.177695\n -0.59029     0.452094  -0.221433      0.9136      0.343194\n -0.178288   -0.190179   0.535448      0.343194   -0.141701\n\n\n\n\n\n","category":"type"},{"location":"Patterned/#under-construction","page":"Patterned Matrices ","title":"under construction","text":"","category":"section"},{"location":"Patterned/","page":"Patterned Matrices ","title":"Patterned Matrices ","text":"randToeplitz","category":"page"},{"location":"Patterned/#RandomMatrix.randToeplitz","page":"Patterned Matrices ","title":"RandomMatrix.randToeplitz","text":"randToeplitz(d, n;  norm, hermitian)  \n\nd : entry distribution\nn : dimension\nnorm : default false;  if norm set to true, then the matrix will be normlaized with n^-12.  \nhermitian: default true; if true the matrix will be Hermitian\n\nExamples\n\nGenerate a 4 times 4 random Hermitian Toeplitz matrix with entries Standard Normal.\n\nrandToeplitz(4)\n\n4×4 Matrix{Float64}:\n  1.10207   -0.47292   -0.745498   1.06809\n -0.47292    1.10207   -0.47292   -0.745498\n -0.745498  -0.47292    1.10207   -0.47292\n  1.06809   -0.745498  -0.47292    1.10207\n\nGenerate a 4 times 4 normalized random Toeplitz matrix with entries Exponential(1).\n\nusing Distributions\nrandToeplitz(Exponential(1),4, norm = true, hermitian = false)\n\n4×4 Matrix{Float64}:\n 0.667888  0.260045  1.48812   0.477305\n 1.50374   0.667888  0.260045  1.48812\n 1.1475    1.50374   0.667888  0.260045\n 0.363966  1.1475    1.50374   0.667888\n\n\n\n\n\n","category":"function"},{"location":"Patterned/","page":"Patterned Matrices ","title":"Patterned Matrices ","text":"randHankel","category":"page"},{"location":"Patterned/#RandomMatrix.randHankel","page":"Patterned Matrices ","title":"RandomMatrix.randHankel","text":"randHankel(d, n;  norm )  \n\nrandHankel(n;  norm)\n\nd : entry distribution\nn : dimension\nnorm : default false,  if norm set to true, then the matrix will be normlaized with n^-12.  \n\nExamples\n\nGenerate a 5times 5 random Hankel matrix with entries uniformly distributed on 1 i pi \n\nrandHankel((1,im,pi),5)\n\n5×5 Matrix{Number}:\n  1   1  im  1   1\n  1  im   1  1   π\n im   1   1  π   π\n  1   1   π  π   π\n  1   π   π  π  im\n\n\n\n\n\n","category":"function"},{"location":"#RandomMatrix.jl-Documentation","page":"Home","title":"RandomMatrix.jl Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A package for Random Matrix Theory, and a project for me to learn Julia.  ","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"The package is registered, to use the package, you can add it by","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Pkg\nPkg.add(\"RandomMatrix\")","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: )","category":"page"}]
}
