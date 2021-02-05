using PastaQ
using ITensors
using Random

Random.seed!(1234)

N = 4     # Number of qubits
depth = 10 # Depth of the quantum circuit

# Generate random quantum circuit built out of
# random 2-qubit unitaries 
println("Random circuit of depth $depth on $N qubits:")
circuit = randomcircuit(N, depth)

# Define custom function to measure an observable, in this
# case a Pauli operator on `site`
function measure_pauli(ψ::MPS, site::Int, pauli::String)
  ψ = orthogonalize!(copy(ψ), site)
  ϕ = ψ[site]
  obs_op = gate(pauli, firstsiteind(ψ, site))
  T = noprime(ϕ * obs_op)
  return real((dag(T) * ϕ)[])
end

# pauli X on site 2
σx2(ψ::MPS) = measure_pauli(ψ, 2, "X")
σz(ψ::MPS) = [measure_pauli(ψ, j, "Z") for j in 1:length(ψ)]

# define the Circuit observer
obs = Observer([
  "χs" => linkdims,      # bond dimension at each bond
  "χmax" => maxlinkdim,  # maximum bond dimension
  "σˣ(2)" => σx2,        # pauli X on site 2
  "σᶻ" => σz]            # pauli Z on each site
)

# run the circuit
ψ = runcircuit(circuit; observer! = obs)

# collect the measurements
println("Bond dimensions at each layer:")
display(results(obs,"χs"))
println()

println("Maximum bond dimension at each layer:")
display(results(obs,"χmax")')
println()

println("⟨ψ|σˣ(2)|ψ⟩ at each layer:")
display(results(obs,"σˣ(2)"))
println()

println("⟨ψ|σᶻ(n)|ψ⟩ at each layer:")
display(results(obs,"σᶻ"))
println()

