using PastaQ
using Random
using ITensors

Random.seed!(1234)

# Unitary circuit
Ψ,data = loaddata("../data/qst_circuit.h5")

N = length(Ψ)
χ = maxlinkdim(Ψ)

ψ0 = initializetomography(N;χ=χ,σ=0.1)
opt = SGD(ψ0;η = 0.005)

println("Training...")
ψ = tomography(ψ0,data,opt;
               batchsize=1000,
               epochs=10,
               target=Ψ,
               localnorm=true)
@show ψ

# Noisy circuit
ϱ,data = loaddata("../data/qst_circuit_noisy.h5")
N = length(ϱ)
χ = maxlinkdim(ϱ)
ξ = 2


ρ0 = initializetomography(N;χ=χ,ξ=ξ,σ=0.1)
opt = SGD(ρ0;η = 0.01)
println("Training...")
ρ = tomography(ρ0,data,opt;
               batchsize=1000,
               epochs=10,
               target=ϱ,
               localnorm=true)
@show ρ
