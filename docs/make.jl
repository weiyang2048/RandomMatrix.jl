using Documenter, RandomMatrix, Plots, LinearAlgebra  

ENV["GKSwstype"] = "100"
gr()

const PAGES = Any["Home" =>"index.md",
      "Random Matrix Models" =>[
            "I.I.D. Matrices"=>"IID Matrices.md",
            "Hermitian Matrices"=>"Hermitian.md",
            "Stochastic Matrices"=>"Stochastic.md",
      ],
      "Distributions"=>"Distributions.md",
      "Under Construction" =>[ 
            "Under Construction1" =>"Others.md",
              "Under Construction2" =>"Gallery.md"
      ]

]

makedocs(modules = [RandomMatrix],
      sitename="RandomMatrix.jl",
      authors = "Wei Yang",
      pages = PAGES
      )

deploydocs(repo = "github.com/weiyang2048/RandomMatrix.jl.git",
devbranch = "main",
push_preview=true )