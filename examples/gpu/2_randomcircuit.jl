using ITensors
using ITensorGPU
using PastaQ
using Random

Random.seed!(1234)

using CUDA: allowscalar
allowscalar(false)

using ITensors.NDTensors: tensor

function convert_eltype(ElType::Type, T::ITensor)
  if eltype(T) === ElType
    return T
  end
  return itensor(ElType.(tensor(T)))
end

# Set to `cu` to run on GPU, `identity` to run on CPU
device = cu

# In principle `ComplexF32` would be good to try,
# but SVD truncation is currently broken:
# https://github.com/ITensor/ITensors.jl/issues/890
device_eltype = ComplexF64

N = 10     # Number of qubits
depth = 6 # Depth of the quantum circuit

# Starting state on GPU
# `|ψ₀⟩ = |0,0,…,0⟩`
println("Make the starting state |ψ₀⟩ = |0,0,…,0⟩...")
ψ₀_cpu = productstate(N)
ψ₀ = device(convert_eltype.(device_eltype, ψ₀_cpu))
@show maxlinkdim(ψ₀)
println()

# Generate random quantum circuit built out of
# layers of single-qubit random rotations + `CX` 
# gates, alternating between even and of odd layers.
println("Random circuit of depth $depth on $N qubits:")
gates = randomcircuit(N; depth)
println()

# Build the circuit as tensors and move them to GPU
gate_tensors = map(
  gate_layer -> device.(convert_eltype.(device_eltype, gate_layer)), buildcircuit(ψ₀, gates)
)

# Obtain the approximate circuit evolution
println("Approximating random circuit evolution Û|0,0,…,0⟩...")
ψ = @time runcircuit(ψ₀, gate_tensors)
@show maxlinkdim(ψ)
println()
