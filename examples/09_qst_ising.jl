using PastaQ
using Random
using ITensors
using Observers
using Printf
using Optimisers: Descent

Random.seed!(1234)
N = 10  # Number of spins
B = 1.0 # Transverse magnetic field

# Build Ising Hamiltonian
sites = siteinds("Qubit", N)
ampo = AutoMPO()
for j in 1:(N - 1)
  # Ising ZZ interactions
  ampo .+= -1, "Z", j, "Z", j + 1
end
for j in 1:N
  # Transverse field X
  ampo .+= -B, "X", j
end
H = MPO(ampo, sites)

# Run DMRG
dmrg_iter = 5      # DMRG steps
dmrg_cutoff = 1E-10   # Cutoff
Ψ0 = randomMPS(sites) # Initial state
sweeps = Sweeps(dmrg_iter)
maxdim!(sweeps, 10, 20, 30, 40, 50, 100)
cutoff!(sweeps, dmrg_cutoff)
# Run DMRG
println("Running DMRG to get ground state of transverse field Ising model:")
E, Ψ = dmrg(H, Ψ0, sweeps)
@show maxlinkdim(Ψ)
println()

# Generate data
nshots = 10_000

# generate `nshots` bases from random local Pauli bases
bases = randombases(N, nshots; local_basis=["X", "Y", "Z"])
# this performs one measurement per basis
data = getsamples(Ψ, bases)
# can also run more than one measurement per basis as follows
# nbases = 100
# nshots_per_basis = 100
# bases = randombases(N, nbases; local_basis = ["X", "Y", "Z"])
# data = getsamples(Ψ, bases, nshots_per_basis) 

# Quantum state tomography
# Initialize variational state
χ = maxlinkdim(Ψ)
ψ0 = randomstate(Ψ0; χ=χ)

# Measurements 
Energy(ψ::MPS) = inner(ψ', H, ψ)
F(ψ::MPS) = fidelity(ψ, Ψ)
ZZ(ψ::MPS) = correlation_matrix(Ψ, "Z", "Z")

# Initialize observer
obs = Observer(["fidelity" => F, "energy" => Energy, "correlations" => ZZ])
#obs = Observer(["energy" => Energy])

@printf("⟨Ψ|Ĥ|Ψ⟩ =  %.5f   ", E)
# Run tomography
println("Running tomography to learn the Ising model ground state from sample data")
ψ = tomography(
  data,
  ψ0;
  optimizer=Descent(0.01),
  batchsize=500,
  epochs=10,
  (observer!)=obs,
  print_metrics=["energy", "fidelity"],
)

@show maxlinkdim(ψ)
println()
