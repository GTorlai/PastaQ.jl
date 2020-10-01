using PastaQ
using ITensors
using Random

import PastaQ.gate
macro GateName_str(s)
  OpName{ITensors.SmallString(s)}
end

Random.seed!(1234)

# Initialize the MPS state ψ = |0,0,0⟩
ψ = qubits(3)

@show ψ

# Apply the X gate on qubit 2
applygate!(ψ,"X",2)

# Show samples from P(x) = |⟨x|ψ⟩|²
println("Sample from |ψ⟩ = X₂|0,0,0⟩:")
display(getsamples(ψ, 3))
println()

# Custom gates
# 
# There might be often problems that require quantum gates not included
# in the standard PastaQ gate set. If that is the case, a new gate can be
# added by overloading the gate(::GateName"...") function, using the format
# defined in gates.jl.
#

gate(::GateName"myX") = 
  [0  1
   1  0]

resetqubits!(ψ)

# Show samples from P(x) = |⟨x|ψ⟩|²
println("Sample from |ψ⟩ = X̃₁|0,0,0⟩:")
applygate!(ψ,"myX",1)
display(getsamples(ψ, 3))
println()

