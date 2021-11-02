using PastaQ
using ITensors
using Zygote
using Random

Random.seed!(1234)
N = 1
q = qubits(N)
θ = [π/3,π/5]
#ϕ = productstate(q,[1])
#ψ = productstate(q,[0])
#ψ = runcircuit(q, ("H",1))
ψ = randomstate(q; normalize = true)
ϕ = randomstate(q; normalize = true)
#debug_circuit(θ) = [("X",1,(f = x -> θ * x,))]
debug_circuit(θ) = [("X",1,(f = x -> exp(im * θ[1] * x),)),
                    ("X",1,(f = x -> exp(im * θ[1] * x),))]

function loss(θ)
  circuit = debug_circuit(θ)
  U = buildcircuit(ψ, circuit)
  return -abs2(PastaQ.inner_circuit(ϕ, U, ψ))
end
∇ad = loss'(θ)
#@show ∇ad 
#ϵ = 1e-5
#θ += ϵ
#f₊ = loss(θ)
#θ -= 2*ϵ
#f₋ = loss(θ)
#∇num = (f₊ - f₋)/(2ϵ)
#@show ∇num
#@show ∇ad - ∇num 
#@show ∇num
ϵ = 1e-8
for k in 1:length(θ)
  θ[k] += ϵ
  global  f₊ = loss(θ)
  θ[k] -= 2*ϵ
  global f₋ = loss(θ)
  global ∇num = (f₊ - f₋)/(2ϵ)
  θ[k] += ϵ
  println("∇ad = ",∇ad[k],"  ∇num = ",∇num)
  @show ∇ad[k] - ∇num
end
