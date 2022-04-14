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

@testset "fidelity optimization with MPS" begin 
  Random.seed!(1234)
  N = 4
  
  function Rylayer(θ⃗)
    return [("Rx", (n,), (θ=θ⃗[n],)) for n in 1:N]
  end
  
  function RXXlayer(ϕ⃗)
    return [("RXX", (n, n + 1), (ϕ = ϕ⃗[n],)) for n in 1:(N÷2)]
  end
  
  # The variational circuit we want to optimize
  function variational_circuit(pars)
    θ⃗, ϕ⃗ = pars
    return vcat(Rylayer(θ⃗),RXXlayer(ϕ⃗), Rylayer(θ⃗), RXXlayer(ϕ⃗))
  end
  
  function f(ψ, ϕ, pars)
    circuit = variational_circuit(pars)
    U = buildcircuit(ψ, circuit)
    ψθ = runcircuit(ψ, U)
    return abs2(inner(ϕ, ψθ))
  end
  
  θ⃗ = 2π .* rand(N)
  ϕ⃗ = 2π .* rand(N÷2)
  pars = [θ⃗, ϕ⃗]  
  # ITensor
  q = qubits(N)
  ψ = productstate(q)
  ϕ = randomstate(q; χ = 10, normalize = true)
  
  finite_difference(f, prod(ψ), prod(ϕ), pars)
  # MPS
  finite_difference(f, ψ, ϕ, pars)
end

@testset "fidelity optimization w MPO & apply_dag = true" begin 
  Random.seed!(1234)
  N = 4
  q = qubits(N)
  
  Rylayer(θ⃗) = [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:N]
  Rxlayer(θ⃗) = [("Rx", (n,), (θ=θ⃗[n],)) for n in 1:N]
  RYYlayer(ϕ⃗) = [("RYY", (n, n + 1), (ϕ = ϕ⃗[n],)) for n in 1:(N÷2)]
  RXXlayer(ϕ⃗) = [("RXX", (n, n + 1), (ϕ = ϕ⃗[n],)) for n in 1:(N÷2)]
  
  
  # The variational circuit we want to optimize
  function variational_circuit(pars)
    θ⃗, ϕ⃗ = pars
    return [Rylayer(θ⃗);
            Rxlayer(ϕ⃗); 
            Rylayer(ϕ⃗);
            Rxlayer(θ⃗);
            RXXlayer(θ⃗[1:N÷2]);
            RYYlayer(ϕ⃗[1:N÷2])]
  end
  function f(ρ, ϕ, pars)
    circuit = variational_circuit(pars)
    U = buildcircuit(q, circuit)
    ρθ = runcircuit(ρ, U)
    return real(inner(ϕ', ρθ, ϕ))
  end

  ψ = randomstate(q; χ = 10, normalize = true)
  ρ = outer(ψ', ψ)
  ϕ = randomstate(q; χ = 1, normalize = true)
  
  θ⃗ = 2π .* rand(N)
  ϕ⃗ = 2π .* rand(N)
  pars = [θ⃗, ϕ⃗]
  
  finite_difference(f, prod(ρ), prod(ϕ), pars)
  finite_difference(f, ρ, ϕ, pars)
  
end

@testset "fidelity optimization w MPO & apply_dag = false" begin 
  Random.seed!(1234)
  N = 4
  q = qubits(N)
  
  Rylayer(θ⃗) = [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:N]
  Rxlayer(θ⃗) = [("Rx", (n,), (θ=θ⃗[n],)) for n in 1:N]
  RYYlayer(ϕ⃗) = [("RYY", (n, n + 1), (ϕ = ϕ⃗[n],)) for n in 1:(N÷2)]
  RXXlayer(ϕ⃗) = [("RXX", (n, n + 1), (ϕ = ϕ⃗[n],)) for n in 1:(N÷2)]
  
  
  # The variational circuit we want to optimize
  function variational_circuit(pars)
    θ⃗, ϕ⃗ = pars
    return [Rylayer(θ⃗);
            Rxlayer(ϕ⃗); 
            Rylayer(ϕ⃗);
            Rxlayer(θ⃗);
            RXXlayer(θ⃗[1:N÷2]);
            RYYlayer(ϕ⃗[1:N÷2])]
  end
  
  function f(ρ, ϕ, pars)
    circuit = variational_circuit(pars)
    U = buildcircuit(q, circuit)
    ρθ = runcircuit(ρ, U; apply_dag = false)
    return real(inner(ϕ', ρθ, ϕ))   
  end

  ψ = randomstate(q; χ = 10, normalize = true)
  ρ = outer(ψ', ψ)
  ϕ = randomstate(q; χ = 10, normalize = true)
  
  θ⃗ = 2π .* rand(N)
  ϕ⃗ = 2π .* rand(N)
  pars = [θ⃗, ϕ⃗]
  finite_difference(f, prod(ρ), prod(ϕ), pars)
  finite_difference(f, ρ, ϕ, pars)
  
end


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
    return [("Ry", n, (θ=θ⃗[n],)) for n in 1:N]
  end
  function Rxlayer(N, θ⃗)
    return [("Rx", n, (θ=θ⃗[n],)) for n in 1:N]
  end
  function CXlayer(N)
    return [("CX", (n, n + 1)) for n in 1:2:(N - 1)]
  end
  
  # The variational circuit we want to optimize
  function variational_circuit(θ⃗)
    N = length(θ⃗)
    return vcat(Rylayer(N, θ⃗),CXlayer(N), Rxlayer(N, θ⃗), CXlayer(N))
  end
  
  Random.seed!(1234)
  N = 2
  q = qubits(N)
  
  os = OpSum()
  for k in 1:N-1
    os += -1.0, "Z",k,"Z",k+1
    os += -1.0,"X",k
  end
  os += -1.0,"X",N
  O = MPO(os,q)
 
  pars = 2π .* rand(N)
  ψ₀ = productstate(q)
  ρ₀ = outer(ψ₀', ψ₀)
  
  function f1(pars)
    circuit = variational_circuit(pars)
    U = buildcircuit(q, circuit)
    ψθ = runcircuit(ψ₀, U)
    return real(inner(ψθ', O, ψθ))
  end
  f1_eval = f1(pars)
  f1_grad = f1'(pars)
  
  function f2(pars)
    circuit = variational_circuit(pars)
    U = buildcircuit(q, circuit)
    U = [swapprime(dag(u), 0 => 1) for u in reverse(U)]
    Oθ = runcircuit(O, U; apply_dag = true)
    return real(inner(ψ₀', Oθ, ψ₀))
  end
  f2_eval = f2(pars)
  f2_grad = f2'(pars)

  function f3(pars)
    circuit = variational_circuit(pars)
    U = buildcircuit(q, circuit)
    ρθ = runcircuit(ρ₀, U; apply_dag = true)
    X = replaceprime(ρθ' * O, 2 => 1)
    return real(tr(ρθ' * O; plev = 2 => 0))
  end
  f3_eval = f3(pars)
  f3_grad = f3'(pars)
  @test f1_eval ≈ f2_eval
  @test f2_eval ≈ f3_eval
  @test f1_grad ≈ f2_grad
  @test f2_grad ≈ f3_grad
  noisemodel = ("depolarizing", (p = 0.1,))
  function f(ρ₀, O, pars)
    circuit = variational_circuit(pars)
    circuit = insertnoise(circuit, noisemodel)
    U = buildcircuit(q, circuit)
    ρθ = runcircuit(ρ₀, U; apply_dag = true)
    return real(tr(ρθ' * O; plev = 2 => 0))
  end
  #finite_difference(f, prod(ρ₀), prod(O), pars)
  finite_difference(f, ρ₀, O, pars)
end


@testset "fidelity optimization - Trotter circuit" begin
  N = 4
  
  import PastaQ:gate 
  @eval gate(::GateName"ZZ") = kron(gate("Z", SiteType("Qubit")), gate("Z", SiteType("Qubit")))
  function hamiltonian(θ)
    H = Tuple[]
    for j in 1:N-1
      H = vcat(H, [(1.0, "ZZ", (j,j+1))])
    end
    for j in 1:N
      H = vcat(H, [(θ[j], "X", j)])
    end
    return H
  end

  Random.seed!(1234)
  pars = rand(N)
  ts = 0:0.1:1.0
  function f(ψ, ϕ, θ)
    H = hamiltonian(θ)
    circuit = trottercircuit(H; ts = ts)
    ψθ = runcircuit(ψ, circuit)
    return abs2(inner(ϕ, ψθ))
  end
  q = qubits(N)
  ψ = productstate(q)
  ϕ = randomstate(q; χ = 10, normalize = true)
  H = hamiltonian(pars)
  circuit = trottercircuit(H; ts = ts) 
  finite_difference(f, ψ, ϕ, pars)
end
