using PastaQ
using ITensors
using Test
using Random
using Zygote

using ChainRulesCore: rrule_via_ad
using Zygote: ZygoteRuleConfig

include("chainrulestestutils.jl")

@testset "one-qubit gate gradients" begin 
  Random.seed!(1234)
  
  q = qubits(2)

  args = (π/3,)
  for σ′ in [1,2]
    for σ in [1,2]
      f = function (x)
        u = gate(q, "Rx", 1, (θ = x,))
        return abs(u[σ,σ′])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end

  args = (π/3,)
  for σ′ in [1,2]
    for σ in [1,2]
      f = function (x)
        u = gate(q, "Ry", 1, (θ = x,))
        return abs(u[σ,σ′])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end
  
  args = (π/3,)
  for σ′ in [1,2]
    for σ in [1,2]
      f = function (x)
        u = gate(q, "Rz", 1, (ϕ = x,))
        return abs(u[σ,σ′])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end
 
  args = (π/3,π/5,π/7)
  for σ′ in [1,2]
    for σ in [1,2]
      f = function (x,y,z)
        u = gate(q, "Rn", 1, (ϕ = x, θ = y, λ= z))
        return abs(u[σ,σ′])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end
 
end



@testset "two-qubit gate gradients" begin 
  Random.seed!(1234)
  
  q = qubits(2)

  args = (π/3,)
  

  basis = [[1,1],[1,2],[2,1],[2,2]]
  
  args = (π/3,)
  for σ′ in basis 
    for σ in basis 
      f = function (x)
        u = gate(q, "CRx", (1,2), (θ = x,))
        i = [σ[1],σ[2],σ′[1],σ′[2]]
        return abs(u[i...])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end
  
  args = (π/3,)
  for σ′ in basis 
    for σ in basis 
      f = function (x)
        u = gate(q, "CRy", (1,2), (θ = x,))
        i = [σ[1],σ[2],σ′[1],σ′[2]]
        return abs(u[i...])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end
  
  args = (π/3,)
  for σ′ in basis 
    for σ in basis 
      f = function (x)
        u = gate(q, "CRz", (1,2), (ϕ = x,))
        i = [σ[1],σ[2],σ′[1],σ′[2]]
        return abs(u[i...])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end
  
  args = (π/3,)
  for σ′ in basis 
    for σ in basis 
      f = function (x)
        u = gate(q, "CRx", (2,1), (θ = x,))
        i = [σ[1],σ[2],σ′[1],σ′[2]]
        return abs(u[i...])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end
  
  args = (π/3,)
  for σ′ in basis 
    for σ in basis 
      f = function (x)
        u = gate(q, "CRy", (2,1), (θ = x,))
        i = [σ[1],σ[2],σ′[1],σ′[2]]
        return abs(u[i...])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end
  
  args = (π/3,)
  for σ′ in basis 
    for σ in basis 
      f = function (x)
        u = gate(q, "CRz", (2,1), (ϕ = x,))
        i = [σ[1],σ[2],σ′[1],σ′[2]]
        return abs(u[i...])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end
 
  args = (π/3,π/5,π/7)
  for σ′ in basis 
    for σ in basis 
      f = function (x,y,z)
        u = gate(q, "CRn", (1,2), (ϕ = x, θ = y, λ= z))
        i = [σ[1],σ[2],σ′[1],σ′[2]]
        return abs(u[i...])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end
 
  args = (π/3,π/5,π/7)
  for σ′ in basis 
    for σ in basis 
      f = function (x,y,z)
        u = gate(q, "CRn", (2,1), (ϕ = x, θ = y, λ= z))
        i = [σ[1],σ[2],σ′[1],σ′[2]]
        return abs(u[i...])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end
 
  args = (π/3,)
  for σ′ in basis 
    for σ in basis 
      f = function (x)
        u = gate(q, "RXX", (1,2), (ϕ = x,))
        i = [σ[1],σ[2],σ′[1],σ′[2]]
        return abs(u[i...])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end

end


@testset "functions applied to qudit gates + gate combination" begin 
  Random.seed!(1234)
 
  N = 2
  q = qudits(N; dim = 5)
  basis = [1,2,3,4]
  
  args = (0.3,)
  for σ′ in basis 
    for σ in basis 
      f = function (x)
        u = gate(q, "a", 1, (f = x ->  exp(im * x),))
        return abs(u[σ,σ′])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end

  for σ′ in basis 
    for σ in basis 
      f = function (x)
        u = gate(q, "a† * a", 1, (f = x ->  exp(im  * x),))
        return abs(u[σ,σ′])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end

  for σ′ in basis 
    for σ in basis 
      f = function (x)
        u = gate(q, "a† * a + a† * a", 1, (f = x -> exp(im *x),))
        return abs(u[σ,σ′])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end

  basis = vec(Iterators.product(fill([1,2,3,4],2)...)|>collect)
  for σ′ in basis 
    for σ in basis 
      f = function (x)
        u = gate(q,"aa†", (1,2), (f = x ->  exp(im *x),))
        i = [σ[1],σ[2],σ′[1],σ′[2]]
        return abs(u[i...])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end
  
  for σ′ in basis 
    for σ in basis 
      f = function (x)
        u = gate(q,"aa† + a†a", (1,2), (f = x ->  exp(im *x),))
        i = [σ[1],σ[2],σ′[1],σ′[2]]
        return abs(u[i...])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end
  
  basis = [1,2,3,4,5]
  g(x) = x^2 * sin(x)
  args = (0.3,)
  for σ′ in basis 
    for σ in basis 
      f = function (x)
        u = gate(q, "a", 1, (f = y ->  exp(im * g(x) * y),))
        return abs(u[σ,σ′])
      end
      test_rrule(ZygoteRuleConfig(), f, args...; rrule_f=rrule_via_ad, check_inferred=false)
    end
  end
end


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
  N = 8
  θ⃗ = 2π .* rand(N)
  ϕ⃗ = 2π .* rand(N÷2)
  
  # ITensor
  Random.seed!(1234)
  
  q = qubits(N)
  ψ = prod(productstate(q))
  ϕ = (N == 1) ? prod(productstate(q, [1])) : prod(runcircuit(q, randomcircuit(N; depth = 2)))
  
  ts = 0:0.1:1.0
  f = function (pars)
    θ⃗,ϕ⃗ = pars
    circuit = variational_circuit(θ⃗,ϕ⃗)
    U = buildcircuit(q, circuit)
    return abs2(inner(ϕ, U, ψ))
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
    return abs2(inner(ϕ, U, ψ))
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

@testset "fidelity optimization - Trotter circuit" begin
  N = 4
  
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

  # ITensor
  Random.seed!(1234)
  θ = rand(N)
  
  q = qubits(N)
  ψ = prod(productstate(q))
  ϕ = (N == 1) ? prod(productstate(q, [1])) : prod(runcircuit(q, randomcircuit(N; depth = 2)))
  
  ts = 0:0.1:1.0
  f = function (θ)
    H = hamiltonian(θ)
    circuit = trottercircuit(H; ts = ts)
    U = buildcircuit(q, circuit)
    return abs2(inner(ϕ, U, ψ))
  end

  # XXX: swap this in
  #test_rrule(ZygoteRuleConfig(), f, θ...; rrule_f=rrule_via_ad, check_inferred=false)
  ∇ad = f'(θ)
  ϵ = 1e-5
  for k in 1:length(θ)
    θ[k] += ϵ
    f₊ = f(θ)
    θ[k] -= 2*ϵ
    f₋ = f(θ)
    ∇num = (f₊ - f₋)/(2ϵ)
    θ[k] += ϵ
    @test ∇ad[k] ≈ ∇num atol = 1e-6 
  end
  
  q = qubits(N)
  ψ = productstate(q)
  ϕ = (N == 1) ? productstate(q, [1]) : runcircuit(q, randomcircuit(N; depth = 2))
  
  ts = 0:0.1:1.0
  f = function (θ)
    H = hamiltonian(θ)
    circuit = trottercircuit(H; ts = ts)
    U = buildcircuit(q, circuit)
    return abs2(inner(ϕ, U, ψ))
  end

  # XXX: swap this in
  #test_rrule(ZygoteRuleConfig(), f, θ...; rrule_f=rrule_via_ad, check_inferred=false)
  ∇ad = f'(θ)
  ϵ = 1e-5
  for k in 1:length(θ)
    θ[k] += ϵ
    f₊ = f(θ)
    θ[k] -= 2*ϵ
    f₋ = f(θ)
    ∇num = (f₊ - f₋)/(2ϵ)
    θ[k] += ϵ
    @test ∇ad[k] ≈ ∇num atol = 1e-6 
  end
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
  
  # ITensor
  ψ = productstate(q)
  f = function (θ⃗)
    circuit = variational_circuit(θ⃗)
    U = buildcircuit(ψ, circuit)
    return rayleigh_quotient(O, U, ψ)
  end
  θ⃗ = 2π .* rand(N)
  
  ∇ad = f'(θ⃗)
  
  ϵ = 1e-5
  for k in 1:length(θ⃗)
    θ⃗[k] += ϵ
    f₊ = f(θ⃗) 
    θ⃗[k] -= 2*ϵ
    f₋ = f(θ⃗) 
    ∇num = (f₊ - f₋)/(2ϵ)
    θ⃗[k] += ϵ
    @test ∇ad[k] ≈ ∇num atol = 1e-8 
  end 
  # MPS
  ψ = productstate(q)
  f = function (θ⃗)
    circuit = variational_circuit(θ⃗)
    U = buildcircuit(ψ, circuit)
    return rayleigh_quotient(O, U, ψ)
  end
  θ⃗ = 2π .* rand(N)
  
  ∇ad = f'(θ⃗)
  
  ϵ = 1e-5
  for k in 1:length(θ⃗)
    θ⃗[k] += ϵ
    f₊ = f(θ⃗) 
    θ⃗[k] -= 2*ϵ
    f₋ = f(θ⃗) 
    ∇num = (f₊ - f₋)/(2ϵ)
    θ⃗[k] += ϵ
    @test ∇ad[k] ≈ ∇num atol = 1e-8 
  end 
end


