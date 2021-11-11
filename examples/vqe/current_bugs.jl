using PastaQ
using ITensors
using Zygote
using Random
import PastaQ: _trottercircuit

function _trotterlayer(H::Vector{<:Tuple}, δτ::Number)
  layer = Tuple[]

  for Hk in H
    coupling = Hk[1]
    localop = Hk[2]
    support = Hk[3]

    # single-qubit gate
    if length(support) == 1
      g = (localop, support[1], (f = x -> exp(-0.5 * δτ * coupling * x),))
      layer = vcat(layer, [g])
    # multi-qubit gate
    else
      g = (localop, support, (f = x -> exp(-0.5 * δτ * coupling * x),))
      layer = vcat(layer, [g])
    end
  end
  return vcat(layer, reverse(layer))
end

function PastaQ._trottercircuit(H::Vector{<:Tuple}, τs::Vector; order::Int = 2, layered::Bool = true, kwargs...)
  nlayers = length(τs) - 1
  # XXX Zygote
  #Hs = repeat([H], nlayers)
  Hs = [H for _ in 1:nlayers]
  _trottercircuit(Hs, τs; kwargs...) 
end

function PastaQ._trottercircuit(H::Vector{<:Vector{<:Tuple}}, τs::Vector; order::Int = 2, layered::Bool = true, kwargs...)
  @assert length(H) == (length(τs) -1) || length(H) == length(τs)
  δτs = diff(τs)
  circuit = [_trotterlayer(H[t], δτs[t]) for t in 1:length(δτs)]
  layered && return circuit
  return reduce(vcat, circuit)
end


N = 2
q = qudits(N; dim = 4)
function variational_circuit(θ⃗)
  H = Tuple[] 
  for n in 1:N-1
    # is this related to each term being or not hermitian? the Trotter step?
    #H += θ⃗[n], "a† + a", n
    #H += θ⃗[n], "a†", n
    #H += θ⃗[n], "a", n
    H = vcat(H, [(θ⃗[n], "a†a", (n,n+1))])
  end
  return trottercircuit(H; δt=0.1, t=1.0, order = 2, layered = false)
end

#function variational_circuit(θ⃗)
#  H = OpSum()
#  for n in 1:N-1
#    # is this related to each term being or not hermitian? the Trotter step?
#    #H += θ⃗[n], "a† + a", n
#    #H += θ⃗[n], "a†", n
#    #H += θ⃗[n], "a", n
#    H += θ⃗[n], "a†a", (n,n+1)
#  end
#  return trottercircuit(H; δt=0.1, t=1.0, order = 2, layered = false)
#end

Random.seed!(1234)
θ⃗ = rand(N)
ψ = randomstate(q; normalize = true)
ϕ = randomstate(q; normalize = true)

function loss(θ⃗)
  circuit = variational_circuit(θ⃗)
  U = buildcircuit(ψ, circuit)
  return -abs2(PastaQ.inner_circuit(ϕ, U, ψ))
end

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

#
#N = 2
#q = qubits(N)
#
#function variational_circuit(θ⃗)
#  H = OpSum()
#  for n in 1:N
#    H += θ⃗[n], "X", n
#  end
#  H += θ⃗[1], "CX", 1,2
#  return trottercircuit(H; δt=0.1, t=1.0, order = 2, layered = true)
#end
#
#Random.seed!(1234)
#θ⃗ = rand(N)
#ψ = randomstate(q; normalize = true)
#ϕ = randomstate(q; normalize = true)
#
#function loss(θ⃗)
#  circuit = variational_circuit(θ⃗)
#  U = buildcircuit(ψ, circuit)
#  return -abs2(PastaQ.inner_circuit(ϕ, U, ψ))
#end
#
#∇ad = loss'(θ⃗)
#ϵ = 1e-5
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
#
#
#
#
#
