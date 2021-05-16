using Documenter, RandomMatrix, Plots, LinearAlgebra  

ENV["GKSwstype"] = "100"
gr()

const PAGES = Any["Home" =>"index.md",
      "Random Matrix Models" =>[
            "I.I.D. Matrices"=>"IID Matrices.md",
            "Hermitian and Elliptic Matrices"=>"Hermitian and Elliptic.md",
            "Unitary Matrices"=>"Unitary.md",
            "Stochastic Matrices"=>"Stochastic.md",
            "Patterned Matrices "=>"Patterned.md",
      ],
      "Distributions"=>[
            "Scarlar-Valued Distributions"=> "ScalarValued.md",
            "Matrix-Valued Distributions"=>"MatrixValued.md"   
            ],
      "Matrix and Spectral Transformations"=>[
            "Spectral Spacing"=>"Spectral Spacing.md"
            ],

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