using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random
#
#@testset "fidelity optimization" begin 
#  function Rylayer(N, θ⃗)
#    return [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:N]
#  end
#  
#  function CXlayer(N)
#    return [("CX", (n, n + 1)) for n in 1:2:(N - 1)]
#  end
#  
#  # The variational circuit we want to optimize
#  function variational_circuit(θ⃗)
#    N = length(θ⃗)
#    return vcat(Rylayer(N, θ⃗),CXlayer(N), Rylayer(N, θ⃗), CXlayer(N))
#  end
#  
#  Random.seed!(1234)
#  N = 8
#  θ⃗ = 2π .* rand(N)
#  circuit = variational_circuit(θ⃗)
#  
#  q = qubits(N)
#  ψ = productstate(q)
#  ϕ = runcircuit(q, randomcircuit(N; depth = 2))
#  
#  function loss(θ⃗)
#    circuit = variational_circuit(θ⃗)
#    U = buildcircuit(ψ, circuit)
#    return -abs2(PastaQ.inner_circuit(ϕ, U, ψ))
#  end
#  
#  θ⃗ = randn!(θ⃗)
#  ∇ad = loss'(θ⃗)
#  
#  ϵ = 1e-5
#  for k in 1:length(θ⃗)
#    θ⃗[k] += ϵ
#    f₊ = loss(θ⃗) 
#    θ⃗[k] -= 2*ϵ
#    f₋ = loss(θ⃗) 
#    ∇num = (f₊ - f₋)/(2ϵ)
#    θ⃗[k] += ϵ
#    @test ∇ad[k] ≈ ∇num atol = 1e-8 
#  end 
#end
#
#
#@testset "optimal control - 1-qubit gate::ITensors.SiteOp{2}" begin
#  N = 4
#  function variational_circuit(θ⃗)
#    H = OpSum()
#    for n in 1:N
#      H += θ⃗[n], "σˣ", n
#    end
#    return trottercircuit(H; δt=0.1, t=1.0, order = 2, layered = true)
#  end
#  
#  Random.seed!(1234)
#  θ⃗ = rand(N)
#  circuit = variational_circuit(θ⃗)
#  
#  q = qubits(N)
#  ψ = productstate(q)
#  U = buildcircuit(ψ, circuit)
#  ϕ = (N == 1) ? productstate(q, [1]) : runcircuit(q, randomcircuit(N; depth = 2))
#  
#  function loss(θ⃗)
#    circuit = variational_circuit(θ⃗)
#    U = buildcircuit(ψ, circuit)
#    return -abs2(PastaQ.inner_circuit(ϕ, U, ψ))
#  end
#  
#  ∇ad = loss'(θ⃗)
#  ϵ = 1e-5
#  for k in 1:length(θ⃗)
#    θ⃗[k] += ϵ
#    f₊ = loss(θ⃗)
#    θ⃗[k] -= 2*ϵ
#    f₋ = loss(θ⃗)
#    ∇num = (f₊ - f₋)/(2ϵ)
#    θ⃗[k] += ϵ
#    @test ∇ad[k] ≈ ∇num atol = 1e-8 
#  end
#
#end
#
#
@testset "optimal control - 1-qubit with gate combination" begin
  N = 4
  function variational_circuit(θ⃗)
    H = OpSum()
    for n in 1:N
      H += θ⃗[n], "σˣ * σᶻ", n
      H += θ⃗[n], "H + S", n
    end
    return trottercircuit(H; δt=0.1, t=0.1, order = 1, layered = true)
  end
  
  Random.seed!(1234)
  θ⃗ = rand(N)
  circuit = variational_circuit(θ⃗)
  
  q = qubits(N)
  ψ = productstate(q)
  ϕ = (N == 1) ? productstate(q, [1]) : runcircuit(q, randomcircuit(N; depth = 2))
  
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
    @test ∇ad[k] ≈ ∇num atol = 1e-7 
  end

end


@testset "optimal control - 1-qudit gates" begin
  N = 4
  function variational_circuit(θ⃗)
    H = OpSum()
    for n in 1:N
      H += θ⃗[n], "a† * a", n
    end
    return trottercircuit(H; δt=0.1, t=1.0, order = 2, layered = true)
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
  
  ∇ad = loss'(θ⃗)
  ϵ = 1e-5
  for k in 1:length(θ⃗)
    θ⃗[k] += ϵ
    f₊ = loss(θ⃗)
    θ⃗[k] -= 2*ϵ
    f₋ = loss(θ⃗)
    ∇num = (f₊ - f₋)/(2ϵ)
    θ⃗[k] += ϵ
    @test ∇ad[k] ≈ ∇num atol = 1e-8 
  end

end

