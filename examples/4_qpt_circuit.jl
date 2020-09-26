using PastaQ
using Random
using ITensors

Random.seed!(1234)

# 1. Quantum process tomography of a unitary circuit

# Load target state and measurements. Each samples is built out
# of a input state (`data_in`) to the quantum channel, and the
# measurement output (`data_out`) after a local basis rotation.
U,data_in,data_out = loaddata("data/qpt_circuit.h5";process=true)
@show U
# Set parameters
N = length(U)     # Number of qubits
χ = maxlinkdim(U) # Bond dimension of variational MPS

# Initialize a variational MPS
V0 = randomprocess(N;χ=χ,σ=0.1)
# Initialize stochastic gradient descent optimizer
opt = SGD(V0;η = 0.1)

V = tomography(V0,data_in,data_out,opt;
               batchsize=500,
               epochs=5,
               target=U)

# Noisy circuit
Random.seed!(1234)
ϱ,data_in,data_out = loaddata("data/qpt_circuit_noisy.h5";process=true)
N = length(ϱ)
χ = 8
ξ = 2
Λ0 = randomprocess(N; mixed=true, χ=χ, ξ=ξ, σ=0.1)
opt = SGD(Λ0;η = 0.1)

Λ = tomography(Λ0,data_in,data_out,opt;
               batchsize=500,
               epochs=5,
               target=ϱ,
               localnorm=true)
@show Λ
