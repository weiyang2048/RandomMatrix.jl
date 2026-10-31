# RandomMatrix.jl

[![CI](https://github.com/weiyang2048/RandomMatrix.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/weiyang2048/RandomMatrix.jl/actions/workflows/CI.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://weiyang2048.github.io/RandomMatrix.jl/dev/)

A package for Random Matrix Theory, providing functionalities to sample from matrix-valued distributions and more.

Since v0.4.0 it also covers the spiked covariance model and the BBP phase transition (`spikeLocation`, `spikeOverlap`, `estimateSpike`), Marčenko–Pastur covariance cleaning (`cleanCovariance` with the rotationally invariant estimator and eigenvalue clipping, plus Ledoit–Wolf and OAS linear shrinkage from data: `ledoitWolf`, `oas`), the Tracy–Widom distributions (`TracyWidom`, with Johnstone's largest-eigenvalue test `twTest`), and numerical free probability (`freeAdd`, `freeMul`, `rTransform`, `sTransform`).

The documentation for the package can be found at **[Documentation 🔗](https://weiyang2048.github.io/RandomMatrix.jl/dev/)**. Release history is in [CHANGELOG.md](CHANGELOG.md).

***
The package is registered, to use the package, you can add it by
```julia
using Pkg
Pkg.add("RandomMatrix")
```

## Quick example

```julia
using RandomMatrix

rand(GOE(3))          # Gaussian Orthogonal Ensemble
rand(GUE(3))          # Gaussian Unitary Ensemble
randUnitary(3)        # Haar-distributed unitary matrix
randToeplitz(4)       # random Hermitian Toeplitz matrix
pdf(MarchenkoPastur(0.5), 1.0)   # Marčenko–Pastur density
```
