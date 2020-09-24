using PastaQ
using Random
using ITensors

Random.seed!(1234)

# 1. Quantum process tomography of a unitary circuit

# Load target state and measurements. Each samples is built out
# of a input state (`data_in`) to the quantum channel, and the
# measurement output (`data_out`) after a local basis rotation.
Φ,data_in,data_out = loaddata("../data/qpt_circuit.h5";process=true)

# Set parameters
N = length(Φ)     # Number of qubits
χ = maxlinkdim(Φ) # Bond dimension of variational MPS

# Initialize a variational MPS
Γ0 = initializetomography(N;χ=χ,σ=0.1)

# Initialize stochastic gradient descent optimizer
opt = SGD(Γ0;η = 0.1)

Γ = tomography(Γ0,data_in,data_out,opt;
               batchsize=500,
               epochs=20,
               target=Φ,
               localnorm=true)
@show Γ

# Noisy circuit
Random.seed!(1234)
ϱ,data_in,data_out = loaddata("../data/qpt_circuit_noisy.h5";process=true)
N = length(ϱ)
χ = 8
ξ = 2

Λ0 = initializetomography(N;χ=χ,ξ=ξ,σ=0.1)
opt = SGD(Λ0;η = 0.1)

println("Training...")
Λ = tomography(Λ0,data_in,data_out,opt;
               batchsize=500,
               epochs=20,
               target=ϱ,
               localnorm=true)
@show Λ
