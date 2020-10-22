using PastaQ
using ITensors
using Random

# Set the random seed so the results are the same
# each time it is run
Random.seed!(1234)

# Initialize the MPS state ψ = |0,0,0⟩
ψ = qubits(3)

# Apply the X gate on qubit 2
g = ("X",2)
ψ = runcircuit(ψ, g)

# Show samples from P(x) = |⟨x|ψ⟩|²
println("Sample from |ψ⟩ = X₂|0,0,0⟩:")
display(getsamples(ψ, 3; local_basis = nothing))
println()

# Custom gates
# 
# There might be often problems that require quantum gates not included
# in the standard PastaQ gate set. If that is the case, a new gate can be
# added by overloading the gate(::GateName"...") function, using the format
# defined in gates.jl.
#

# Import gate so it can be overloaded
import PastaQ: gate

gate(::GateName"iX") = 
  [ 0 im
   im  0]

# Set the state back to |0,0,0⟩
resetqubits!(ψ)

# Show samples from P(x) = |⟨x|ψ⟩|²
println("Sample from |ψ⟩ = iX₁|0,0,0⟩:")
g = ("iX", 1)
ψ = runcircuit(ψ, g)
display(getsamples(ψ, 3; local_basis = nothing))
println()

