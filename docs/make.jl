using Documenter, RandomMatrix, Plots, LinearAlgebra

ENV["GRDIR"] = ""
Pkg.build("GR")


const PAGES = Any["Home" =>"index.md",
      "Random Matrix Models" =>[
            "I.I.D. entries"=>"IID Matrices.md",
            "Under Construction" =>"Others.md"
      ] 

]

makedocs(modules = [RandomMatrix],
      sitename="RandomMatrix.jl",
      authors = "Wei Yang",
      pages = PAGES
      )

deploydocs(repo = "github.com/weiyang2048/RandomMatrix.jl.git",devbranch = "main")