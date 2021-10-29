using PastaQ
using ITensors
using Zygote
using Random

function Rylayer(N, θ⃗)
  return [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:N]
end

function CXlayer(N)
  return [("CX", (n, n + 1)) for n in 1:2:(N - 1)]
end

# The variational circuit we want to optimize
function variational_circuit(θ⃗)
  N = length(θ⃗)
  return vcat(Rylayer(N, θ⃗),CXlayer(N), Rylayer(N, θ⃗), CXlayer(N))
end

Random.seed!(1234)
N = 8
θ⃗ = 2π .* rand(N)
circuit = variational_circuit(θ⃗)

q = qubits(N)
ψ = productstate(q)
U = buildcircuit(ψ, circuit)
ϕ = runcircuit(q, randomcircuit(N; depth = 2))

function loss(θ⃗)
  circuit = variational_circuit(θ⃗)
  U = buildcircuit(ψ, circuit)
  return -abs2(PastaQ.inner_circuit(ϕ, U, ψ))
end
θ⃗ = randn!(θ⃗)
∇ad = loss'(θ⃗)

ϵ = 1e-5
for k in 1:length(θ⃗)
  θ⃗[k] += ϵ
  f₊ = loss(θ⃗)
  θ⃗[k] -= 2*ϵ
  f₋ = loss(θ⃗)
  ∇num = (f₊ - f₋)/(2ϵ)
  θ⃗[k] += ϵ
  println("∇ad = ",∇ad[k],"  ∇num = ",∇num)
  @show ∇ad[k] - ∇num
end
