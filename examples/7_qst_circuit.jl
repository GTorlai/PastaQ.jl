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
circuit = randomcircuit(N, depth)
data, ψ = getsamples(circuit, nshots; local_basis=["X", "Y", "Z"])
writesamples(data, ψ, "data/qst_circuit.h5")

# Load target state and measurement data
data, Ψ = readsamples("data/qst_circuit.h5")

# Split data into a train and test dataset
train_data, test_data = split_dataset(data; train_ratio=0.9)

# Set parameters 
N = length(Ψ)     # Number of qubits
χ = maxlinkdim(Ψ) # Bond dimension of variational MPS

# Initialize the variational MPS
ψ0 = randomstate(Ψ; χ=χ)

# Initialize stochastic gradient descent optimizer
opt = SGD(; η=0.01)

# Initialize the observer for the fidelity
F(ψ::MPS) = fidelity(ψ, Ψ)
obs = Observer(F)

# Run quantum state tomography, where a variational MPS `|ψ(θ)⟩`
# is optimized to mimimize the cross entropy between the data and 
# the tensor-network distribution `P(x) = |⟨x|ψ(θ)⟩|²`.
println("Running tomography to learn a pure state ψ:")
ψ = tomography(
  train_data,
  ψ0;
  test_data=test_data,
  optimizer=opt,
  batchsize=1000,
  epochs=5,
  (observer!)=obs,
  print_metrics="F",
)
@show maxlinkdim(ψ)
println()

# 2. Quantum state tomography on measurements generated
# as the output of a noisy quantum circuit

# Generate sample data
data, ρ = getsamples(
  circuit, nshots; local_basis=["X", "Y", "Z"], noise=("amplitude_damping", (γ=0.01,))
)
writesamples(data, ρ, "data/qst_circuit_noisy.h5")

# Load target state and measurement data
data, ϱ = readsamples("data/qst_circuit_noisy.h5")

# Split data into a train and test dataset
train_data, test_data = split_dataset(data; train_ratio=0.9)

# Set parameters
N = length(ϱ)     # Number of qubits
χ = maxlinkdim(ϱ) # Bond dimension of variational LPDO
ξ = 2             # Kraus dimension of variational LPDO

# Initialize the LPDO
ρ0 = randomstate(ϱ; mixed=true, χ=χ, ξ=ξ)

# Initialize stochastic gradient descent optimizer
opt = SGD(; η=0.1)

# Initialize the observer
F(ρ::LPDO) = fidelity(ρ, ϱ; warnings=false)

obs = Observer(F)

# Run quantum state tomography, where a variational LPDO `ρ(θ)`
# is optimized to mimimize the cross entropy between the data and 
# the tensor-network distribution `P(x) = ⟨x|ρ(θ)|x⟩`.
println("Running tomography to learn a mixed state ρ:")
@disable_warn_order begin
  ρ = tomography(
    train_data,
    ρ0;
    test_data=test_data,
    optimizer=opt,
    batchsize=1000,
    epochs=5,
    (observer!)=obs,
    print_metrics="F",
  )
end
@show maxlinkdim(ρ.X)
println()
