#using PastaQ
#using ITensors
#using Test
#using Random
#using Zygote
#
#using ChainRulesCore: rrule_via_ad
#using Zygote: ZygoteRuleConfig
#
#include("chainrulestestutils.jl")

@testset "fidelity optimization: 1-qubit & 2-qubit gates" begin 
  function Rylayer(N, θ⃗)
    return [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:N]
  end
  
  function RXXlayer(N, ϕ⃗)
    return [("RXX", (n, n + 1), (ϕ = ϕ⃗[n],)) for n in 1:(N÷2)]
  end
  
  # The variational circuit we want to optimize
  function variational_circuit(θ⃗,ϕ⃗)
    N = length(θ⃗)
    return vcat(Rylayer(N, θ⃗),RXXlayer(N,ϕ⃗), Rylayer(N, θ⃗), RXXlayer(N,ϕ⃗))
  end
  
  Random.seed!(1234)
  N = 3
  θ⃗ = 2π .* rand(N)
  ϕ⃗ = 2π .* rand(N÷2)
  
  # ITensor
  Random.seed!(1234)
  
  #q = qubits(N)
  #ψ = prod(productstate(q))
  #ϕ = prod(runcircuit(q, randomcircuit(N; depth = 2)))
  #
  #ts = 0:0.1:1.0
  #f = function (pars)
  #  θ⃗,ϕ⃗ = pars
  #  circuit = variational_circuit(θ⃗, ϕ⃗)
  #  U = buildcircuit(ψ, circuit)
  #  ψθ = runcircuit(ψ, U)
  #  return abs2(inner(ϕ, ψθ))
  #end
  # XXX: swap this in
  #test_rrule(ZygoteRuleConfig(), f, θ...; rrule_f=rrule_via_ad, check_inferred=false)
  #∇ad = f'([θ⃗,ϕ⃗])
  #ϵ = 1e-5
  #for k in 1:length(θ⃗)
  #  θ⃗[k] += ϵ
  #  f₊ = f([θ⃗,ϕ⃗]) 
  #  θ⃗[k] -= 2*ϵ
  #  f₋ = f([θ⃗,ϕ⃗]) 
  #  ∇num = (f₊ - f₋)/(2ϵ)
  #  θ⃗[k] += ϵ
  #  @test ∇ad[1][k] ≈ ∇num atol = 1e-8 
  #end 
  #for k in 1:length(ϕ⃗)
  #  ϕ⃗[k] += ϵ
  #  f₊ = f([θ⃗,ϕ⃗]) 
  #  ϕ⃗[k] -= 2*ϵ
  #  f₋ = f([θ⃗,ϕ⃗]) 
  #  ∇num = (f₊ - f₋)/(2ϵ)
  #  ϕ⃗[k] += ϵ
  #  @test ∇ad[2][k] ≈ ∇num atol = 1e-8 
  #end 
  #
  # MPS
  Random.seed!(1234)
  
  q = qubits(N)
  ψ = productstate(q)
  ϕ = (N == 1) ? productstate(q, [1]) : runcircuit(q, randomcircuit(N; depth = 2))
  
  ts = 0:0.1:1.0
  f = function (pars)
    θ⃗,ϕ⃗ = pars
    circuit = variational_circuit(θ⃗,ϕ⃗)
    U = buildcircuit(q, circuit)
    ψθ = runcircuit(ψ, U)
    return abs2(inner(ϕ, ψθ))
  end

  # XXX: swap this in
  #test_rrule(ZygoteRuleConfig(), f, θ...; rrule_f=rrule_via_ad, check_inferred=false)
  ∇ad = f'([θ⃗,ϕ⃗])
  ϵ = 1e-5
  for k in 1:length(θ⃗)
    θ⃗[k] += ϵ
    f₊ = f([θ⃗,ϕ⃗]) 
    θ⃗[k] -= 2*ϵ
    f₋ = f([θ⃗,ϕ⃗]) 
    ∇num = (f₊ - f₋)/(2ϵ)
    θ⃗[k] += ϵ
    @test ∇ad[1][k] ≈ ∇num atol = 1e-8 
  end 
  for k in 1:length(ϕ⃗)
    ϕ⃗[k] += ϵ
    f₊ = f([θ⃗,ϕ⃗]) 
    ϕ⃗[k] -= 2*ϵ
    f₋ = f([θ⃗,ϕ⃗]) 
    ∇num = (f₊ - f₋)/(2ϵ)
    ϕ⃗[k] += ϵ
    @test ∇ad[2][k] ≈ ∇num atol = 1e-8 
  end 
end

#@testset "fidelity optimization - Trotter circuit" begin
#  N = 4
#  
#  function hamiltonian(θ)
#    H = Tuple[]
#    for j in 1:N-1
#      H = vcat(H, [(1.0, "ZZ", (j,j+1))])
#    end
#    for j in 1:N
#      H = vcat(H, [(θ[j], "X", j)])
#    end
#    return H
#  end
#
#  # ITensor
#  Random.seed!(1234)
#  θ = rand(N)
#  
#  #q = qubits(N)
#  #ψ = prod(productstate(q))
#  #ϕ = (N == 1) ? prod(productstate(q, [1])) : prod(runcircuit(q, randomcircuit(N; depth = 2)))
#  #
#  #ts = 0:0.1:1.0
#  #f = function (θ)
#  #  H = hamiltonian(θ)
#  #  circuit = trottercircuit(H; ts = ts)
#  #  U = buildcircuit(q, circuit)
#  #  ψθ = runcircuit(ψ, U)
#  #  return abs2(inner(ϕ, ψθ))
#  #end
#
#  ## XXX: swap this in
#  ##test_rrule(ZygoteRuleConfig(), f, θ...; rrule_f=rrule_via_ad, check_inferred=false)
#  #∇ad = f'(θ)
#  #ϵ = 1e-5
#  #for k in 1:length(θ)
#  #  θ[k] += ϵ
#  #  f₊ = f(θ)
#  #  θ[k] -= 2*ϵ
#  #  f₋ = f(θ)
#  #  ∇num = (f₊ - f₋)/(2ϵ)
#  #  θ[k] += ϵ
#  #  @test ∇ad[k] ≈ ∇num atol = 1e-6 
#  #end
#  
#  q = qubits(N)
#  ψ = productstate(q)
#  ϕ = (N == 1) ? productstate(q, [1]) : runcircuit(q, randomcircuit(N; depth = 2))
#  
#  ts = 0:0.1:1.0
#  f = function (θ)
#    H = hamiltonian(θ)
#    circuit = trottercircuit(H; ts = ts)
#    U = buildcircuit(q, circuit)
#    ψθ = runcircuit(ψ, U)
#    return abs2(inner(ϕ, ψθ))
#  end
#
#  # XXX: swap this in
#  #test_rrule(ZygoteRuleConfig(), f, θ...; rrule_f=rrule_via_ad, check_inferred=false)
#  ∇ad = f'(θ)
#  ϵ = 1e-5
#  for k in 1:length(θ)
#    θ[k] += ϵ
#    f₊ = f(θ)
#    θ[k] -= 2*ϵ
#    f₋ = f(θ)
#    ∇num = (f₊ - f₋)/(2ϵ)
#    θ[k] += ϵ
#    @test ∇ad[k] ≈ ∇num atol = 1e-6 
#  end
#end
#
#
#
#@testset "rayleigh_quotient" begin 
#  function Rylayer(N, θ⃗)
#    return [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:N]
#  end
#  function Rxlayer(N, θ⃗)
#    return [("Rx", (n,), (θ=θ⃗[n],)) for n in 1:N]
#  end
#  function CXlayer(N)
#    return [("CX", (n, n + 1)) for n in 1:2:(N - 1)]
#  end
#  
#  # The variational circuit we want to optimize
#  function variational_circuit(θ⃗)
#    N = length(θ⃗)
#    return vcat(Rylayer(N, θ⃗),CXlayer(N), Rxlayer(N, θ⃗), Rylayer(N, θ⃗), CXlayer(N), Rxlayer(N, θ⃗))
#  end
#  
#  Random.seed!(1234)
#  N = 4
#  
#  q = qubits(N)
#  
#  os = OpSum()
#  for k in 1:N-1
#    os += 1.0, "Z",k,"Z",k+1
#    os += 1.0,"X",k
#  end
#  O = MPO(os,q)
#  
#  ## ITensor
#  #ψ = productstate(q)
#  #f = function (θ⃗)
#  #  circuit = variational_circuit(θ⃗)
#  #  U = buildcircuit(ψ, circuit)
#  #  ψθ = runcircuit(ψ, U)
#  #  return real(inner(ψθ', O, ψθ))
#  #  #return rayleigh_quotient(O, U, ψ)
#  #end
#  #θ⃗ = 2π .* rand(N)
#  #
#  #∇ad = f'(θ⃗)
#  #
#  #ϵ = 1e-5
#  #for k in 1:length(θ⃗)
#  #  θ⃗[k] += ϵ
#  #  f₊ = f(θ⃗) 
#  #  θ⃗[k] -= 2*ϵ
#  #  f₋ = f(θ⃗) 
#  #  ∇num = (f₊ - f₋)/(2ϵ)
#  #  θ⃗[k] += ϵ
#  #  @test ∇ad[k] ≈ ∇num atol = 1e-8 
#  #end 
#  # MPS
#  ψ = productstate(q)
#  f = function (θ⃗)
#    circuit = variational_circuit(θ⃗)
#    U = buildcircuit(ψ, circuit)
#    ψθ = runcircuit(ψ, U)
#    return real(inner(ψθ', O, ψθ))
#  end
#  θ⃗ = 2π .* rand(N)
#  
#  ∇ad = f'(θ⃗)
#  
#  ϵ = 1e-5
#  for k in 1:length(θ⃗)
#    θ⃗[k] += ϵ
#    f₊ = f(θ⃗) 
#    θ⃗[k] -= 2*ϵ
#    f₋ = f(θ⃗) 
#    ∇num = (f₊ - f₋)/(2ϵ)
#    θ⃗[k] += ϵ
#    @test ∇ad[k] ≈ ∇num atol = 1e-8 
#  end 
#end
#
