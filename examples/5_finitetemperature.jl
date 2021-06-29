using PastaQ
using ITensors
using Printf

# 1. Prepation of a thermal state 
# 
# In this example, we show how to prepare the finite-temperature state
# of a many-body system:
#
# ρ̂(β) = exp(-β Ĥ)
#
# where Ĥ is the Hamiltonian and β is the inverse temperature.
# We specificallty consider the one-dimensional Ising model 
#
#   H = - ∑ᵢ σᶻ(i) σᶻ(i+1) - B ∑ᵢ σˣ(i)
#
# where B a the transverse magnetic field. 

# 1a. Custom gates 
#
# In order to build the thermal density operator, we implement the 
# simplest flavor of imaginary-time evolution, breaking the operator
# exp(-βĤ) into a set of two-qubit and single-qubit gates, corresponding
# to the Ising interactions and the transverse field respetively. The 
# time evolution to inverse temperature β is broken into elementary steps
# of size τ, where a gate is applied for each term appearing in the Hamiltonian.
#
# In this example, the quantum gates are not contained in the gate set of PastaQ. 
# In order to extend, it is ony required to define the gate matrices using a 
# format analogous to standard gates defined in gates.jl.

import PastaQ: gate

gate(::GateName"expτZZ"; τ::Float64) = exp(τ * kron(gate("Z"), gate("Z")))

gate(::GateName"expτX"; τ::Float64, B::Float64) = exp(τ * B * gate("X"))

# 1b. Generating the thermal state

N = 10    # Number of spins
B = 1.0   # Transverse magnetic field
β = 1.0   # Inverse temperature
τ = 0.005 # Trotter step

# Depth of the circuit
depth = β ÷ τ

# Ising interactions
zz_layer = [("expτZZ", (j, j + 1), (τ=τ,)) for j in 1:(N - 1)]
# Transverse field
x_layer = [("expτX", j, (τ=τ, B=B)) for j in 1:N]

# Build the gate structure
circuit = []
for d in 1:depth
  append!(circuit, zz_layer)
  append!(circuit, x_layer)
end

#
# 2. Run imaginary-time evolution towards the zero temperature
# ground state.
#

# 2a. Ground state energy with DMRG 
#
# We compute the ground state energy by running DMRG
# on the Hamiltonian MPO, whose algorithm is implemented in 
# ITensors.jl.

# In order to generate the MPO for the Hamiltonian, we leverage
# the ITensors.jl `AutoMPO()` function, which automatically
# generates the local MPO tensors from a set of pre-definend operators..
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
# Generate Hamilotnian MPO
H = MPO(ampo, sites)

# Density-matrix renormalization group
dmrg_iter = 5      # DMRG steps
dmrg_cutoff = 1E-10   # Cutoff
Ψ0 = randomMPS(sites) # Initial state
sweeps = Sweeps(dmrg_iter)
maxdim!(sweeps, 10, 20, 30, 40, 50, 100)
cutoff!(sweeps, dmrg_cutoff)
# Run 
println("Running DMRG to get ground state of transverse field Ising model:")
E, Ψ = dmrg(H, Ψ0, sweeps)
@printf("\nGround state energy:  %.8f  \n", E)
println("\n---------------------------------------\n")

#
# 2b. Run the imaginary-time circuit
#

β = 5.0 # Inverse temperature
Δ = 0.5  # Intermediate time-step
depth = Δ ÷ τ # Depth of the circuit
steps = β ÷ Δ # Total number of circuit application

# Initialize the density operator
ρ = PastaQ.identity_mpo(H)

println("Running imaginary time evolution to approximate the density matrix ρ = exp(-βH):")
for b in 1:steps
  # Run the circuit
  global ρ = runcircuit(ρ, circuit; cutoff=1E-12)

  # Normalize the density operatorr
  normalize!(ρ)

  # Measure the energy
  E_th = inner(ρ, H)

  @printf("β = %.1f : tr(ρH) = %.8f \n", (Δ * b), E_th)
end
