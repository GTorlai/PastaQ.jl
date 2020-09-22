using PastaQ
using ITensors
using Random

N = 10
depth=4
gates = randomcircuit(N,depth)

# Evolution of pure state
ψ = runcircuit(N,gates)
@show maxlinkdim(ψ)

# Noise
ρ = runcircuit(N,gates;noise="AD",γ=0.1)
@show maxlinkdim(ρ)

