using PastaQ
using Random
using ITensors

Random.seed!(1234)

# 1. Quantum process tomography of a unitary circuit

# Load target state and measurements. Each samples is built out
# of a input state (`data_in`) to the quantum channel, and the
# measurement output (`data_out`) after a local basis rotation.
data_in, data_out, Û = loaddata("data/qpt_circuit.h5"; process = true)

# Set parameters
N = length(Û)     # Number of qubits
χ = maxlinkdim(Û) # Bond dimension of variational MPS

# Initialize the unitary MPO
U0 = randomprocess(N; χ = χ)
# TODO: temporary check
#ψ = randomstate(2*N;χ=8)

#U0 = unsplitunitary(ψ)
#ϕ = splitunitary(U0)

# Initialize stochastic gradient descent optimizer
@show maxlinkdim(U0)
opt = SGD(η = 0.1)

# Run process tomography
U = tomography(U0, data_in, data_out, opt;
               batchsize = 500,
               epochs = 5,
               target = Û)
@show U

# Noisy circuit
Random.seed!(1234)
# Load data and target Choi matrix
data_in, data_out, Φ = loaddata("data/qpt_circuit_noisy.h5"; process = true)
N = length(Φ)
χ = 8
ξ = 2

# Initialize the Choi LPDO
Λ0 = randomprocess(Φ; mixed = true, χ = χ, ξ = ξ)

# Initialize stochastic gradient descent optimizer
opt = SGD(η = 0.1)

# Run process tomography
Λ = tomography(Λ0, data_in, data_out, opt;
               mixed = true,
               batchsize = 500,
               epochs = 5,
               target = Φ)
@show Λ
