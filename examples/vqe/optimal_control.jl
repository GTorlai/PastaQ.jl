using PastaQ
using ITensors
using Zygote
using Random

N = 1

function variational_circuit(θ⃗)
  H = OpSum()
  for n in 1:N
    H += θ⃗[n], "σˣ", n
  end
  return trottercircuit(H; δt=0.1, t=0.1, order = 1)
end

Random.seed!(1234)
θ⃗ = rand(N)
circuit = variational_circuit(θ⃗)

q = qubits(N)
ψ = productstate(q)
U = buildcircuit(ψ, circuit)
ϕ = (N == 1) ? productstate(q, [1]) : runcircuit(q, randomcircuit(N; depth = 2))

function loss(θ⃗)
  circuit = variational_circuit(θ⃗)
  U = buildcircuit(ψ, circuit)
  return -abs2(PastaQ.inner_circuit(ϕ, U, ψ))
end

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
