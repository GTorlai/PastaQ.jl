using PastaQ
using ITensors
using Zygote
using Random

debug_circuit(θ) = ("X", 1, (f = x -> θ * x,))
Random.seed!(1234)
N = 1
q = qubits(N)
θ = π/3
ϕ = productstate(q,[1])
ψ = productstate(q,[0])

function loss(θ)
  circuit = debug_circuit(θ)
  U = buildcircuit(ψ, circuit)
  return -abs2(PastaQ.inner_circuit(ϕ, U, ψ))
end

@show -θ^2
@show loss(θ)
#  -(θ ^ 2) = -1.0966227112321507
#  loss(θ) = -1.0966227112321507

@show -2*θ
@show loss'(θ)
#  -2θ = -2.0943951023931953
#  (loss')(θ) = -2.2967612355777636
#
#
#
#
#
#
#N = 1
#
#function variational_circuit(θ⃗)
#  H = OpSum()
#  for n in 1:N
#    H += θ⃗[n], "σˣ", n
#  end
#  return Tuple[("X",1, (f = x -> θ⃗[1] * x,))]
#  #return trottercircuit(H; δt=0.1, t=0.1, order = 1)
#end
#
#Random.seed!(1234)
#θ⃗ = rand(N)
#circuit = variational_circuit(θ⃗)
#
#q = qubits(N)
#ψ = productstate(q)
#U = buildcircuit(ψ, circuit)
##ϕ = productstate(q, rand(0:1,N))
#ϕ = (N == 1) ? productstate(q, [1]) : runcircuit(q, randomcircuit(N; depth = 2))
#
#function loss(θ⃗)
#  circuit = variational_circuit(θ⃗)
#  U = buildcircuit(ψ, circuit)
#  return -abs2(PastaQ.inner_circuit(ϕ, U, ψ))
#end
#
##@show loss(θ⃗)
##∇ad = loss'(θ⃗)
##@show ∇ad
##u = exp(-im * τ * θ⃗[1] * gate("X"))
##M = -im * τ * gate("X") * u 
##
##g = -2 * real(conj(u[1,2]) * M[1,2])
##@show g
##
##θ⃗ = randn!(θ⃗)
#@show loss(θ⃗)
#∇ad = loss'(θ⃗)
#ϵ = 1e-5
#
#for k in 1:length(θ⃗)
#  θ⃗[k] += ϵ
#  f₊ = loss(θ⃗)
#  θ⃗[k] -= 2*ϵ
#  f₋ = loss(θ⃗)
#  ∇num = (f₊ - f₋)/(2ϵ)
#  θ⃗[k] += ϵ
#  println("∇ad = ",∇ad[k],"  ∇num = ",∇num)
#  @show ∇ad[k] - ∇num
#end
