using PastaQ
using ITensors
using Test
using Random
using Zygote

using ChainRulesCore: rrule_via_ad
using Zygote: ZygoteRuleConfig

include("chainrulestestutils.jl")

function finite_difference(f, A, B, pars)
  g(pars) = f(A, B, pars)
  ∇ad = g'(pars)
  ϵ = 1e-8
  for k in 1:length(pars)
    if pars[k] isa AbstractArray
      for j in 1:length(pars[k])
        g₀ = g(pars)
        pars[k][j] += ϵ
        gϵ = g(pars)
        pars[k][j] -= ϵ
        ∇num = (gϵ - g₀)/ϵ
        @test ∇ad[k][j] ≈  ∇num atol = 1e-6
      end
    else
      g₀ = g(pars)
      pars[k] += ϵ
      gϵ = g(pars)
      pars[k] -= ϵ
      ∇num = (gϵ - g₀)/ϵ
      @test ∇ad[k] ≈  ∇num atol = 1e-6
    end
  end
end
#
#@testset "fidelity optimization: 1-qubit & 2-qubit gates" begin 
#  Random.seed!(1234)
#  N = 4
#  
#  function Rylayer(θ⃗)
#    return [("Rx", (n,), (θ=θ⃗[n],)) for n in 1:N]
#  end
#  
#  function RXXlayer(ϕ⃗)
#    return [("RXX", (n, n + 1), (ϕ = ϕ⃗[n],)) for n in 1:(N÷2)]
#  end
#  
#  # The variational circuit we want to optimize
#  function variational_circuit(pars)
#    θ⃗, ϕ⃗ = pars
#    return vcat(Rylayer(θ⃗),RXXlayer(ϕ⃗), Rylayer(θ⃗), RXXlayer(ϕ⃗))
#  end
#  
#  function f(ψ, ϕ, pars)
#    circuit = variational_circuit(pars)
#    U = buildcircuit(ψ, circuit)
#    ψθ = runcircuit(ψ, U)
#    return abs2(inner(ϕ, ψθ))
#  end
#  
#  θ⃗ = 2π .* rand(N)
#  ϕ⃗ = 2π .* rand(N÷2)
#  pars = [θ⃗, ϕ⃗]  
#  # ITensor
#  q = qubits(N)
#  ψ = productstate(q)
#  ϕ = randomstate(q; χ = 10, normalize = true)
#  
#  finite_difference(f, prod(ψ), prod(ϕ), pars)
#  # MPS
#  finite_difference(f, ψ, ϕ, pars)
#end
#
#@testset "fidelity optimization w MPO & apply_dag = true: 1-qubit & 2-qubit gates" begin 
#  Random.seed!(1234)
#  N = 4
#  q = qubits(N)
#  
#  Rylayer(θ⃗) = [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:N]
#  Rxlayer(θ⃗) = [("Rx", (n,), (θ=θ⃗[n],)) for n in 1:N]
#  RYYlayer(ϕ⃗) = [("RYY", (n, n + 1), (ϕ = ϕ⃗[n],)) for n in 1:(N÷2)]
#  RXXlayer(ϕ⃗) = [("RXX", (n, n + 1), (ϕ = ϕ⃗[n],)) for n in 1:(N÷2)]
#  
#  
#  # The variational circuit we want to optimize
#  function variational_circuit(pars)
#    θ⃗, ϕ⃗ = pars
#    return [Rylayer(θ⃗);
#            Rxlayer(ϕ⃗); 
#            Rylayer(ϕ⃗);
#            Rxlayer(θ⃗);
#            RXXlayer(θ⃗[1:N÷2]);
#            RYYlayer(ϕ⃗[1:N÷2])]
#  end
#  function f(ρ, ϕ, pars)
#    circuit = variational_circuit(pars)
#    U = buildcircuit(q, circuit)
#    ρθ = runcircuit(ρ, U)
#    return real(inner(ϕ', ρθ, ϕ))
#  end
#
#  ψ = randomstate(q; χ = 10, normalize = true)
#  ρ = outer(ψ, ψ)
#  ϕ = randomstate(q; χ = 1, normalize = true)
#  
#  θ⃗ = 2π .* rand(N)
#  ϕ⃗ = 2π .* rand(N)
#  pars = [θ⃗, ϕ⃗]
#  
#  finite_difference(f, prod(ρ), prod(ϕ), pars)
#  finite_difference(f, ρ, ϕ, pars)
#  
#end
#
#
#
@testset "rayleigh_quotient" begin 
  function Rylayer(N, θ⃗)
    return [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:N]
  end
  function Rxlayer(N, θ⃗)
    return [("Rx", (n,), (θ=θ⃗[n],)) for n in 1:N]
  end
  function CXlayer(N)
    return [("CX", (n, n + 1)) for n in 1:2:(N - 1)]
  end
  
  # The variational circuit we want to optimize
  function variational_circuit(θ⃗)
    N = length(θ⃗)
    return vcat(Rylayer(N, θ⃗),CXlayer(N), Rxlayer(N, θ⃗), Rylayer(N, θ⃗), CXlayer(N), Rxlayer(N, θ⃗))
  end
  
  Random.seed!(1234)
  N = 4
  
  q = qubits(N)
  
  os = OpSum()
  for k in 1:N-1
    os += 1.0, "Z",k,"Z",k+1
    os += 1.0,"X",k
  end
  O = MPO(os,q)
  
  function f(ψ, O, pars)
    circuit = variational_circuit(pars)
    ψθ = runcircuit(ψ, circuit)
    return real(inner(ψθ', O, ψθ))
  end
  pars = 2π .* rand(N)
  ψ = productstate(q)
  finite_difference(f, prod(ψ), prod(O), pars)
  finite_difference(f, ψ, O, pars)
end




@testset "rayleigh_quotient with noise" begin 
  function Rylayer(N, θ⃗)
    return [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:N]
  end
  function Rxlayer(N, θ⃗)
    return [("Rx", (n,), (θ=θ⃗[n],)) for n in 1:N]
  end
  function CXlayer(N)
    return [("CX", (n, n + 1)) for n in 1:2:(N - 1)]
  end
  
  # The variational circuit we want to optimize
  function variational_circuit(θ⃗)
    N = length(θ⃗)
    return vcat(Rylayer(N, θ⃗),CXlayer(N), Rxlayer(N, θ⃗), Rylayer(N, θ⃗), CXlayer(N), Rxlayer(N, θ⃗))
  end
  
  Random.seed!(1234)
  N = 4
  noisemodel = ("depolarizing", (p = 0.01,)) 
  q = qubits(N)
  
  os = OpSum()
  for k in 1:N-1
    os += 1.0, "Z",k,"Z",k+1
    os += 1.0,"X",k
  end
  O = MPO(os,q)
  
  function f(ψ, O, pars)
    circuit = variational_circuit(pars)
    ρθ = runcircuit(ψ, circuit; noise = noisemodel)
    ρθH = replaceprime(ρθ' * H, 2 => 1)
    return tr(ρθH)
  end
  pars = 2π .* rand(N)
  ψ = productstate(q)
  finite_difference(f, prod(ψ), prod(O), pars)
  finite_difference(f, ψ, O, pars)
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
