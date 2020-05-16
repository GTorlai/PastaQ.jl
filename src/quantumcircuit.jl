struct QuantumCircuit
  N::Int
  seed::Int
end

function QuantumCircuit(;N::Int,seed::Int=1234)
  Random.seed!(seed)
  return QuantumCircuit(N,seed)
end
