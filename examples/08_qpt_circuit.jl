using PastaQ
using Random
using ITensors
using Observers
using Optimisers: Optimisers

Random.seed!(1234)

# 1. Quantum process tomography of a unitary circuit

# Make the random circuit
N = 4
depth = 4
nshots = 10_000
circuit = randomcircuit(N, depth)

# Generate samples
data, U = getsamples(circuit, nshots; local_basis=["X", "Y", "Z"], process=true)
writesamples(data, U, "data/qpt_circuit.h5")

# Load target state and measurements. Each samples is built out
# of an input state (`first.(data)`) to the quantum channel, and the
# measurement output (`last.(data)`) after a local basis rotation.
data, Û = readsamples("data/qpt_circuit.h5")

# Split data into a train and test dataset
train_data, test_data = split_dataset(data; train_ratio=0.9)

# Set parameters
N = length(Û)     # Number of qubits
χ = maxlinkdim(Û) # Bond dimension of variational MPS

# Initialize the unitary MPO
U0 = randomprocess(Û; χ=χ)

opt = Optimisers.Descent(0.01)

F(U::MPO; kwargs...) = fidelity(U, Û; process=true)
obs = Observer(["F" => F])

# Initialize stochastic gradient descent optimizer
@show maxlinkdim(U0)

# Run process tomography
println("Run process tomography to learn noiseless circuit U")
U = tomography(
  train_data,
  U0;
  test_data=test_data,
  optimizer=opt,
  batchsize=500,
  epochs=5,
  (observer!)=obs,
  print_metrics=["F"],
)

@show maxlinkdim(U)
println()

# Noisy circuit
# Generate samples
data, Λ = getsamples(
  circuit,
  nshots;
  local_basis=["X", "Y", "Z"],
  process=true,
  noise=("amplitude_damping", (γ=0.01,)),
)
writesamples(data, Λ, "data/qpt_circuit_noisy.h5")

# Load data and target Choi matrix
data, Φ = readsamples("data/qpt_circuit_noisy.h5")

# Split data into a train and test dataset
train_data, test_data = split_dataset(data; train_ratio=0.9)

# Set up
N = length(Φ)
χ = 8
ξ = 2

# Initialize the Choi LPDO
Λ0 = randomprocess(Φ; mixed=true, χ=χ, ξ=ξ)

# Initialize stochastic gradient descent optimizer
opt = Optimisers.ADAM()

F(Λ::LPDO; kwargs...) = fidelity(Λ, Φ; process=true)
obs = Observer(["F" => F])

# Run process tomography
println("Run process tomography to learn noisy process Λ")
@disable_warn_order begin
  Λ = tomography(
    train_data,
    Λ0;
    test_data=test_data,
    optimizer=opt,
    batchsize=500,
    epochs=5,
    (observer!)=obs,
    print_metrics=["F"],
  )
end
@show maxlinkdim(Λ.X)
println()
