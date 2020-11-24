using PastaQ
using ITensors
using ITensorsGPU
using Random

Random.seed!(1234)

using CUDA: allowscalar
allowscalar(false)

# Set to identity to run on CPU
gpu = cu

N = 40     # Number of qubits
depth = 18 # Depth of the quantum circuit

# Starting state on GPU
# `|ψ₀⟩ = |0,0,…,0⟩`
println("Make the starting state |ψ₀⟩ = |0,0,…,0⟩...")
ψ₀ = gpu(qubits(N))
@show maxlinkdim(ψ₀)
println()

# Generate random quantum circuit built out of
# layers of single-qubit random rotations + `CX` 
# gates, alternating between even and of odd layers.
println("Random circuit of depth $depth on $N qubits:")
gates = randomcircuit(N, depth)
display(gates)
println()

# Build the circuit as tensors and move them to GPU
gate_tensors = gpu.(complex.(buildcircuit(ψ₀, gates)))

# Obtain the approximate circuit evolution
println("Approximating random circuit evolution Û|0,0,…,0⟩...")
ψ = @time runcircuit(ψ₀, gate_tensors)
@show maxlinkdim(ψ)
println()

