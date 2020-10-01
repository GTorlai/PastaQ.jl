using PastaQ
using Random
using ITensors

Random.seed!(1234)

# 1. Quantum state tomography on measurements generated
# as the output of a unitary quantum circuit

# Generate the sample data
N = 4
depth = 4
nshots = 10_000
gates = randomcircuit(N,depth)
data, ψ = getsamples(N, gates, nshots)
writesamples(data, ψ, "data/qst_circuit.h5")

# Load target state and measurement data
data, Ψ = readsamples("data/qst_circuit.h5")

# Set parameters 
N = length(Ψ)     # Number of qubits
χ = maxlinkdim(Ψ) # Bond dimension of variational MPS

# Initialize the variational MPS
ψ0 = randomstate(Ψ; χ=χ)

# Initialize stochastic gradient descent optimizer
opt = SGD(η = 0.01)

# Run quantum state tomography, where a variational MPS `|ψ(θ)⟩`
# is optimized to mimimize the cross entropy between the data and 
# the tensor-network distribution `P(x) = |⟨x|ψ(θ)⟩|²`.
println("Running tomography on to learn a pure state ψ:")
ψ = tomography(data, ψ0;
               optimizer = opt,
               batchsize = 1000,
               epochs = 5,
               target = Ψ)
@show maxlinkdim(ψ)
println()


# 2. Quantum state tomography on measurements generated
# as the output of a noisy quantum circuit

# Generate sample data
data, ρ = getsamples(N, gates, nshots;
                     noise = ("amplitude_damping", (γ = 0.01,)))
writesamples(data, ρ, "data/qst_circuit_noisy.h5")

# Load target state and measurement data
data, ϱ = readsamples("data/qst_circuit_noisy.h5")

# Set parameters
N = length(ϱ)     # Number of qubits
χ = maxlinkdim(ϱ) # Bond dimension of variational LPDO
ξ = 2             # Kraus dimension of variational LPDO

# Initialize the LPDO
ρ0 = randomstate(ϱ; mixed = true, χ = χ, ξ = ξ)

# Initialize stochastic gradient descent optimizer
opt = SGD(η = 0.1)
# Run quantum state tomography, where a variational LPDO `ρ(θ)`
# is optimized to mimimize the cross entropy between the data and 
# the tensor-network distribution `P(x) = ⟨x|ρ(θ)|x⟩`.
println("Running tomography on to learn a mixed state ρ:")
ρ = tomography(data, ρ0;
               optimizer = opt,
               batchsize = 1000,
               epochs = 5,
               target = ϱ)
@show maxlinkdim(ρ.X)
println()

