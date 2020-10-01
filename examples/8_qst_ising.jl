using PastaQ
using Random
using ITensors

Random.seed!(1234)
N = 10  # Number of spins
B = 1.0 # Transverse magnetic field

# Build Ising Hamiltonian
sites = siteinds("Qubit", N)
ampo = AutoMPO()
for j in 1:N-1
  # Ising ZZ interactions
  ampo .+= -1, "Z", j, "Z", j+1
end
for j in 1:N
  # Transverse field X
  ampo .+= -B, "X", j
end
H = MPO(ampo,sites)

# Run DMRG
dmrg_iter   = 5      # DMRG steps
dmrg_cutoff = 1E-10   # Cutoff
Ψ0 = randomMPS(sites) # Initial state
sweeps = Sweeps(dmrg_iter)
maxdim!(sweeps, 10,20,30,40,50,100)
cutoff!(sweeps, dmrg_cutoff)
# Run DMRG
println("Running DMRG to get ground state of transverse field Ising model:")
E , Ψ = dmrg(H, Ψ0, sweeps)
@show maxlinkdim(Ψ)
println()

# Generate data
nshots = 10_000
bases = randombases(N, nshots)
data = getsamples(Ψ, bases)

# Quantum state tomography
# Initialize variational state
χ = maxlinkdim(Ψ)
ψ0 = randomstate(Ψ0; χ = χ)
# Run tomography
println("Running tomography to learn the Ising model ground state from sample data")
ψ = tomography(data, ψ0;
               optimizer = SGD(η = 0.01),
               batchsize = 500,
               epochs = 10,
               target = Ψ)
@show maxlinkdim(ψ)
println()

