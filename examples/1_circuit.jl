using PastaQ
using ITensors
using Random

Random.seed!(1234)

N = 4   # Number of qubits
depth=4 # Depth of the quantum circuit

# Generate random quantum circuit built out of
# layers of single-qubit random rotations + `CX` 
# gates, alternating between even and of odd layers.
gates = randomcircuit(N,depth)

# 1. Unitary quantum circuit
# Returns the MPS at the output of the quantum circuit:
# `|ψ⟩ = Û|0,0,…,0⟩`
# where `Û` is the unitary circuit.
ψ = runcircuit(N,gates)
@show maxlinkdim(ψ)
@show ψ

# A representation of the unitary operation as a MPO
# is obtained using the flag `process=true`:
U = runcircuit(N,gates;process=true)
@show maxlinkdim(U)
@show U

# 2. Quantum circuit with noise
# Apply a noide model `noise` (specified by appropriate 
# parameters) to each quantum gate in `gates`.
# Returns a mixed density operator as MPO:
# `ρ = ε(|0,0,…⟩⟨0,0,̇|)`
# where `ε` is the quantum channel.
# Here, the noise is a single-qubit amplitude damping 
# channel with decay rate `γ=0.01`..
ρ = runcircuit(N,gates;noise="AD",γ=0.01)
@show maxlinkdim(ρ)
@show ρ

# A representation of the quantum channel as a MPO
# is obtained using the flag `process=true`, which 
# returns the Choi matrix `Λ` of the channel:`:
Λ = runcircuit(N,gates;process=true,noise="AD",γ=0.01)
@show maxlinkdim(Λ)
@show Λ


