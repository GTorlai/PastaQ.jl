using PastaQ
using ITensors
using Printf
import PastaQ.gate

macro GateName_str(s)
  OpName{ITensors.SmallString(s)}
end

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
# exp(βĤ) into a set of two-qubit and single-qubit gates, corresponding
# to the Ising interactions and the transverse field respetively. The 
# time evolution to inverse temperature β is broken into elementary steps
# of size τ, where a gate is applied for each term appearing in the Hamiltonian.
#
# In this example, the quantum gates are not contained in the gate set of PastaQ. 
# In order to extend, it is ony required to define the gate matrices using a 
# format analogous to standard gates defined in gates.jl.

gate(::GateName"τZZ"; τ::Float64) = 
  exp(τ * kron(gate("Z"), gate("Z")))

gate(::GateName"τX"; τ::Float64, B::Float64) = 
  exp(τ * B * gate("X"))


# 1b. Generating the thermal state

N = 10    # Number of spins
B = 1.0   # Transverse magnetic field
β = 1.0   # Inverse temperature
τ = 0.005 # Trotter step

# Depth of the circuit
depth = β ÷ τ

gates = Tuple[]

# Ising interactions
zz_layer = Tuple[("τZZ", (j, j+1), (τ = τ,)) for j in 1:N-1]
# Transverse field
x_layer = gatelayer("τX",N; τ = τ, B = B)

# Build the gate structure
for d in 1:depth
  append!(gates,zz_layer)
  append!(gates,x_layer)
end 

# Initialize the density matrix
ρ0 = circuit(H)

ρ = runcircuit(ρ0,gates)
normalize!(ρ)

# Measure the energy
E_th = inner(ρ,H)
@printf("\nInverse temperature β = %.1f : Tr(ρ̂Ĥ) = %.8f  \n",β,E_th)
println("\n---------------------------------------\n")


# 2. Run imaginary-time evolution towards the zero temperature
# ground state.

# 2a. Ground state energy with DMRG 
#
# We compute the ground state energy by running DMRG
# on the Hamiltonian MPO, whose algorithm is implemented in 
# ITensors. 

# In order to generate the MPO for the Hamiltonian, we leverage
# the `AutoMPO()` function, which automatically generates
# the local MPO tensors from a set of pre-definend operators..
sites = siteinds("S=1/2",N)
ampo = AutoMPO()
for j in 1:N-1
  # Ising ZZ interactions
  ampo .+= -2.0, "Sz", j, "Sz", j+1
end
for j in 1:N
  # Transverse field X
  ampo .+= -B, "Sx", j
end
# Generate Hamilotnian MPO
H = MPO(ampo,sites)


# Density-matrix renormalization group
dmrg_iter   = 5      # DMRG steps
dmrg_cutoff = 1E-10   # Cutoff
Ψ0 = randomMPS(sites) # Initial state
sweeps = Sweeps(dmrg_iter)
maxdim!(sweeps, 10,20,30,40,50,100)
cutoff!(sweeps, dmrg_cutoff)
# Run 
println("Running DMRG to get ground state of transverse field Ising model:")
E , Ψ = dmrg(H, Ψ0, sweeps)
@printf("\nGround state energy:  %.8f  \n",E)
println("\n---------------------------------------\n")


# 2b. Run the imaginary-time circuit

β = 10.0 # Inverse temperature
Δ = 0.5  # Intermediate time-step
depth = Δ ÷ τ # Depth of the circuit
steps = β ÷ Δ # Total number of circuit application

# Initialize the density operator
ρ = circuit(H)

for b in 1:steps
  
  # Run the circuit
  global ρ = runcircuit(ρ,gates; cutoff = 1E-12)
  
  # Normalize the density operatorr
  normalize!(ρ)
  
  # Measure the energy
  E_th = inner(ρ, H)

  @printf("β = %.1f : Tr(ρ̂Ĥ) = %.8f \n", (Δ * b), E_th)
end

