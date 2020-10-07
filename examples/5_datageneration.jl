using ITensors
using PastaQ
using Random
using HDF5

Random.seed!(1234)

N = 4
depth = 4
nshots = 5
gates = randomcircuit(N,depth)

# 1. Generation of measurement data on the quantum states
# at the output of a circuit. Each data-point is a projetive
# measurement in an arbitrary local basis. The default local basis 
# is `["X","Y","Z"]`.
# a) Unitary circuit
# Returns output state as MPS
println("Generate samples from random projective measurements of the state U|0,0,…>:")
data, ψ = getsamples(N, gates, nshots)
# Example of writing and reading
writesamples(data, ψ, "data/qst_circuit.h5")
data, ψ = readsamples("data/qst_circuit.h5")
@show maxlinkdim(ψ)
display(data)
println()

# Note: the above is equivalent to:
#> bases = randombases(N,nshots)
#> ψ = runcircuit(N,gates)
#> data = getsamples(ψ,nshots,bases)

# b) Noisy circuit
# Returns the mixed density operator as MPO
println("Generate samples from random projective measurements of the state ρ = ε(|0,0,…⟩⟨0,0,…|) generated from noisy gate evolution:")
data, ρ = getsamples(N, gates, nshots;
                     noise = ("amplitude_damping", (γ = 0.01,)))
# Example of writing and reading
writesamples(data, ρ, "data/qst_circuit_noisy.h5")
data, ρ = readsamples("data/qst_circuit_noisy.h5")
@show maxlinkdim(ρ)
display(data)
println()

# 2. Generation of measurerment data for quantum process
# tomography. Each measurement consist of a input product 
# state and an output projective measurement in a arbitrary
# local basis. By default, the single-qubit input states are 
# the 6 eigenstates of Pauli operators.
# Return the MPO for the unitary circuit
println("Generate samples from random input states and random project measurements of the circuit U:")
data, U = getsamples(N, gates, nshots;
                     process = true)
# Example of writing and reading
writesamples(data, U, "data/qpt_circuit.h5")
data, U = readsamples("data/qpt_circuit.h5")
if !isnothing(U)
  @show maxlinkdim(U)
end
display(data)
println()

println("Generate samples from random input states and random project measurements of the Choi matrix Λ generated from noisy gate evolution:")
# Returns the Choi matrix `Λ` as MPO wiith `2N` sites
data, Λ = getsamples(N, gates, nshots;
                     process = true,
                     noise = ("amplitude_damping", (γ = 0.01,)))
# Example of writing and reading
writesamples(data, Λ, "data/qpt_circuit_noisy.h5")
data, Λ = readsamples("data/qpt_circuit_noisy.h5")
if !isnothing(Λ)
  @show maxlinkdim(Λ.M)
end
display(data)
println()

