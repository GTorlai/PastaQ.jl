using PastaQ
using ITensors
using Random

Random.seed!(1234)

N = 40
depth=16
gates = randomcircuit(N,depth)

# Evolution of pure state
ψ = runcircuit(N,gates,cutoff=1e-9)

# Noise
ρ = runcircuit(N,gates;noise="AD",γ=0.1, cutoff=1e-9)
@show maxlinkdim(ρ)

