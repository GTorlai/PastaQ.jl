using PastaQ
using ITensors
using Random

Random.seed!(1234)
N = 4   # Number of qubits
depth=4 # Depth of the quantum circuit
gates = randomcircuit(N,depth)

ψ = runcircuit(N, gates)
U = runcircuit(N, gates; process = true)
ρ = runcircuit(N, gates; noise = ("amplitude_damping", (γ = 0.01,)))
Λ = runcircuit(N, gates; process = true, noise = ("amplitude_damping", (γ = 0.01,)))

Random.seed!(1234)
nshots = 50

data, ψ = getsamples(N, gates, nshots; local_basis = ["X","Y","Z"])
writesamples(data, ψ, "../examples/data/qst_circuit_test.h5")

data, ρ = getsamples(N, gates, nshots; local_basis = ["X","Y","Z"],
                     noise = ("amplitude_damping", (γ = 0.01,)))
writesamples(data, ρ, "../examples/data/qst_circuit_noisy_test.h5")

data, U = getsamples(N, gates, nshots; local_basis = ["X","Y","Z"],
                     process = true)
writesamples(data, U, "../examples/data/qpt_circuit_test.h5")

data, Λ = getsamples(N,gates,nshots; local_basis = ["X","Y","Z"], 
                     process = true,
                     noise = ("amplitude_damping", (γ = 0.01,)))
writesamples(data, Λ, "../examples/data/qpt_circuit_noisy_test.h5")


Random.seed!(1234)
data, Ψ = readsamples("../examples/data/qst_circuit_test.h5")
N = length(Ψ)     # Number of qubits
χ = maxlinkdim(Ψ) # Bond dimension of variational MPS
ψ0 = randomstate(Ψ; χ = χ, σ = 0.1)
opt = SGD(η = 0.01)


obs = Observer([fidelity => Ψ])
print_metrics = ["fidelity"]
ψ = tomography(data, ψ0;
               optimizer = opt,
               epochs = 2,
               observer! = obs,
               print_metrics = print_metrics)


data, ϱ = readsamples("../examples/data/qst_circuit_noisy_test.h5")
N = length(ϱ)     # Number of qubits
χ = maxlinkdim(ϱ) # Bond dimension of variational LPDO
ξ = 2             # Kraus dimension of variational LPDO
ρ0 = randomstate(ϱ; mixed = true, χ = χ, ξ = ξ, σ = 0.1)
opt = SGD(η = 0.01)

obs = Observer([fidelity => ϱ, frobenius_distance => ϱ])
print_metrics = ["fidelity","frobenius_distance"]
ρ = tomography(data, ρ0;
               optimizer = opt,
               epochs = 2,
               observer! = obs,
               print_metrics = print_metrics)

Random.seed!(1234)
data, U = readsamples("../examples/data/qpt_circuit_test.h5")
N = length(U)     # Number of qubits
χ = maxlinkdim(U) # Bond dimension of variational MPS
V0 = randomprocess(U; χ = χ)
opt = SGD(η = 0.1)


obs = Observer(fidelity => U)
print_metrics = ["fidelity"]
V = tomography(data, V0;
               optimizer = opt,
               epochs = 2,
               trace_preserving_regularizer = 0.1,
               observer! = obs,
               print_metrics = print_metrics)

# Noisy circuit
Random.seed!(1234)
data, ϱ = readsamples("../examples/data/qpt_circuit_noisy_test.h5")
N = length(ϱ)
χ = 8
ξ = 2
Λ0 = randomprocess(ϱ; mixed = true, χ = χ, ξ = ξ, σ = 0.1)
opt = SGD(η = 0.1)
@disable_warn_order begin
  obs = Observer([fidelity => ϱ, fidelity_bound => ϱ])
  print_metrics = ["fidelity","fidelity_bound"]
  Λ = tomography(data, Λ0;
                 optimizer = opt,
                 epochs = 2,
                 trace_preserving_regularizer = 0.1,
                 observer! = obs,
                 print_metrics = print_metrics)
end
