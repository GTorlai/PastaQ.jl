using PastaQ
using Random
using ITensors

Random.seed!(1234)

# 1. Quantum state tomography on measurements generated
# as the output of a unitary quantum circuit

# Load target state and measurement data
Ψ,data = loaddata("../data/qst_circuit.h5")

# Set parameters 
N = length(Ψ)     # Number of qubits
χ = maxlinkdim(Ψ) # Bond dimension of variational MPS

# Initialize the variational MPS
ψ0 = initializetomography(N;χ=χ,σ=0.1)

# Initialize stochastic gradient descent optimizer
opt = SGD(ψ0;η = 0.005)

# Run quantum state tomography, where a variational MPS `|ψ(θ)⟩`
# is optimized to mimimize the cross entropy between the data and 
# the tensor-network distribution `P(x) = |⟨x|ψ(θ)⟩|²`.
ψ = tomography(ψ0,data,opt;
               batchsize=1000,
               epochs=10,
               target=Ψ,
               localnorm=true)
@show ψ


# 2. Quantum state tomography on measurements generated
# as the output of a noisy quantum circuit

# Load target state and measurement data
ϱ,data = loaddata("../data/qst_circuit_noisy.h5")

# Set parameters
N = length(ϱ)     # Number of qubits
χ = maxlinkdim(ϱ) # Bond dimension of variational LPDO
ξ = 2             # Kraus dimension of variational LPDO

# Initialize the LPDO
ρ0 = initializetomography(N;χ=χ,ξ=ξ,σ=0.1)

# Initialize stochastic gradient descent optimizer
opt = SGD(ρ0;η = 0.01)

# Run quantum state tomography, where a variational LPDO `Γ(θ)`
# is optimized to mimimize the cross entropy between the data and 
# the tensor-network distribution `P(x) = ⟨x|ρ(θ)|x⟩`, where
# `ρ = ΓΓ†` is the corresponding density operator..
ρ = tomography(ρ0,data,opt;
               batchsize=1000,
               epochs=10,
               target=ϱ,
               localnorm=true)
@show ρ
