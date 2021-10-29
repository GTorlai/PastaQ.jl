using PastaQ
using ITensors
using Zygote
using Random

function variational_circuit(θ⃗)
  N = 1
  H = OpSum()
  for n in 1:N
    H += θ⃗[1], "σˣ", 1
  end
  return trottercircuit(H; δt=0.1, t=0.1, layered=true)
end

Random.seed!(1234)
N = 1
θ⃗ = [π/3]
circuit = variational_circuit(θ⃗)

q = qubits(N)
ψ = productstate(q)
U = buildcircuit(ψ, circuit)
ϕ = runcircuit(q, U)

function loss(θ⃗)
  circuit = variational_circuit(θ⃗)
  U = buildcircuit(ψ, circuit)
  return -abs2(PastaQ.inner_circuit(ϕ, U, ψ))
end
θ⃗ = randn!(θ⃗)

@show loss(θ⃗)

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
