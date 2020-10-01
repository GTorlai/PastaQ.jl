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

# Parameters

N = 10    # Number of spins
B = 1.0   # Transverse magnetic field
β = 1.0   # Inverse temperature
τ = 0.02  # Trotter step


# In order to generate the MPO for the Hamiltonian, we leverage
# the `AutoMPO()` function of ITensors, which automatically generates
# the local MPO tensor from a set of interactions.
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


# Depth of the circuit
depth = β ÷ τ

# Initialize the density matrix
ρ0 = circuit(H)

gate_tensors = ITensor[]
for d in 1:depth
  # ising interaction
  for j in 1:N-1;
    z1 = gate(ρ0,"Z",j)
    z2 = gate(ρ0,"Z",j+1)
    zz = z1 * z2
    g  = exp(τ * zz) 
    push!(gate_tensors,g)
  end 
  # transverse field
  for j in 1:N 
    x = gate(ρ0,"X",j)
    g = exp(τ * B * x)
    push!(gate_tensors,g)
  end 
end 

# Generate density matrix
ρ = runcircuit(ρ0,gate_tensors)
normalize!(ρ)

# Measure the energy
E_th = inner(ρ,H)
@printf("Inverse temperature β = %.1f : Tr(ρ̂Ĥ) = %.8f  ",β,E_th)
println("\n---------------------------------------\n")




# 2. Run imaginary-time evolution to the ground state

# Density-matrix renormalization group
#
# First, we compute the ground state energy by running DMRG
# on the Hamiltonian MPO, whose algoirthm is implemented in 
# ITensor. 

dmrg_iter   = 5      # DMRG steps
dmrg_cutoff = 1E-10   # Cutoff
Ψ0 = randomMPS(sites) # Initial state
sweeps = Sweeps(dmrg_iter)
maxdim!(sweeps, 10,20,30,40,50,100)
cutoff!(sweeps, dmrg_cutoff)
# Run 
println("Running DMRG to get ground state of transverse field Ising model:")
E , Ψ = dmrg(H, Ψ0, sweeps)
@printf("\nGround state energy:  %.8f  ",E)
println("\n---------------------------------------\n")

# Inverse temperature loop
for b in 1:20
  # Current inverse temperature
  β = 0.5 * b
  # Depth of the circuit 
  depth = β ÷ τ
  
  # Initialize the density matrix
  ρ0 = circuit(H)
  orthogonalize!(ρ0,1)

  gate_tensors = ITensor[]
  for d in 1:depth
    # ising interaction
    for j in 1:N-1;
      z1 = gate(ρ0,"Z",j)
      z2 = gate(ρ0,"Z",j+1)
      zz = z1 * z2
      g  = exp(τ * zz) 
      push!(gate_tensors,g)
    end 
    # transverse field
    for j in 1:N 
      x = gate(ρ0,"X",j)
      g = exp(τ * B * x)
      push!(gate_tensors,g)
    end 
  end 
  
  # Generate density matrix
  ρ = runcircuit(ρ0,gate_tensors; cutoff = 1E-12)
  normalize!(ρ)
  
  # Measure the energy
  E_th = inner(ρ,H)
  @printf("β = %.1f : Tr(ρ̂Ĥ) = %.8f ",β,E_th)
  println("   $(maxlinkdim(ρ))")
end

