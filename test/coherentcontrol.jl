using PastaQ
using ITensors
using Test
using Random
using Printf
using Zygote: Zygote

@testset "Control field gradients" begin
  
  f(t, θ) = PastaQ.control_fourierseries(t, θ; maxfrequency = 1000.0, amplitude = 100.0)
  θ = rand(11)
  t = 0.1
  gs = Zygote.gradient(Zygote.Params([θ])) do
    f(t, θ)
  end
  
  alg_grads = gs[θ]
  ϵ = 1e-5
  for i in 1:length(θ)
    θ[i] += ϵ
    fp = f(t, θ)
    θ[i] -= 2*ϵ
    fm = f(t, θ)
    θ[i] += ϵ
    numgrad = (fp-fm)/(2*ϵ)
    @test numgrad ≈ alg_grads[i] atol = 1e-8
  end

  g(t, θ) = PastaQ.control_sweep(t, θ)
  θ = rand(6)
  t = 0.1
  gs = Zygote.gradient(Zygote.Params([θ])) do
    g(t, θ)
  end
  
  alg_grads = gs[θ]
  ϵ = 1e-5
  for i in 1:length(θ)
    θ[i] += ϵ
    gp = g(t, θ)
    θ[i] -= 2*ϵ
    gm = g(t, θ)
    θ[i] += ϵ
    numgrad = (gp-gm)/(2*ϵ)
    @test numgrad ≈ alg_grads[i] atol = 1e-8
  end
end



@testset "OCC-parameters gradients: single state, single drive on single mode" begin
  Random.seed!(1234)
  N = 3
  sites = qubits(N) 
  
  T   = 1.0
  δt  = 0.1 

  f(t, θ) = θ[1] * cos(t) + θ[2] * cos(2.0 * t)

  ts = 0.0:δt:T
  B = 0.5
  
  function generate_hamiltonian_sequence(parameter)
    Ht = OpSum[]
    for t in ts
      H = OpSum()
      # loop over the pauli operators
      for j in 1:N-1
        H += -1.0,"ZZ",(j,j+1)
        H += -B,"X",j
      end 
      H += -B,"X",N
      
      (f,θ) = first(parameter)
      (localop, support) = last(parameter)
      H += f(t, θ), localop, support, (∇ = true,)
      
      push!(Ht, H)
    end
    return Ht
  end 
  
  θ = [1.011,1.155]
  parameter = (f,θ) => ("Z",2)
  
  
  Ht = generate_hamiltonian_sequence(parameter)
  circuit = trottercircuit(Ht; ts = ts)
  
  ψ = productstate(sites)
  ϕ = normalize!(randomstate(sites))
  
  cmap = PastaQ.circuitmap(circuit)
  F, ∇alg = PastaQ.gradients(copy(ψ), copy(ϕ), circuit, parameter, collect(ts), cmap)
  ψT = runcircuit(ψ, circuit)
  @test F ≈ fidelity(ψT, ϕ) atol = 1e-5

  ϵ = 1e-5
  for k in 1:length(θ)
    θ[k] += ϵ
    parameter = (f,θ) => ("Z",2)
    Ht = generate_hamiltonian_sequence(parameter)
    circuit = trottercircuit(Ht; ts = ts)
    ψT = runcircuit(ψ, circuit)
    F₊ = fidelity(ψT, ϕ)

    θ[k] -= 2 * ϵ
    parameter = (f,θ) => ("Z",2)
    Ht = generate_hamiltonian_sequence(parameter)
    circuit = trottercircuit(Ht; ts = ts)
    ψT = runcircuit(ψ, circuit)
    F₋ = fidelity(ψT, ϕ)

    θ[k] += ϵ
    ∇num = (F₊ - F₋) / (2*ϵ)
    @test ∇num ≈ ∇alg[1][k]
  end
end


#@testset "OCC-parameters gradients: single state, single drive on many modes" begin
#  Random.seed!(1234)
#  N = 3
#  sites = qubits(N) 
#  
#  B = 0.5
#  H = OpSum()
#  # loop over the pauli operators
#  for j in 1:N-1
#    H += -1.0,"ZZ",(j,j+1)
#    H += -B,"X",j
#  end 
#  H += -B,"X",N
#  
#  T   = 1.0
#  δt  = 0.1 
#
#  f(t, θ) = θ[1] * cos(t) + θ[2] * cos(2.0 * t)
#  θ₀ = [1.011,1.155]
#
#  ts = 0.0:δt:T
#  drives = (f,θ₀) => [("Z",j) for j in 1:N]
#  Hts = [PastaQ._drivinghamiltonian(H, drives, t) for t in ts] 
#  circuit = trottercircuit(Hts; ts =  ts) 
#  
#  ψ = productstate(sites)
#  ϕ = normalize!(randomstate(sites))
#
#  cmap = PastaQ.circuitmap(circuit)
#  F, ∇alg = PastaQ.gradients(ψ, ϕ, circuit, drives, collect(ts), cmap)
#
#  ψT = runcircuit(ψ, circuit)
#  @test F ≈ fidelity(ψT, ϕ) atol = 1e-5
#
#  ϵ = 1e-5
#  for k in 1:length(θ₀)
#    θ₀[k] += ϵ
#    drives = [f,θ₀] => [("Z",j) for j in 1:N]
#    Hts = [PastaQ._drivinghamiltonian(H, drives, t) for t in ts]
#    circuit = trottercircuit(Hts; ts =  ts)
#    ψT = runcircuit(ψ, circuit)
#    F₊ = fidelity(ψT, ϕ)
#
#    θ₀[k] -= 2 * ϵ
#    drives = [f,θ₀] => [("Z",j) for j in 1:N]
#    Hts = [PastaQ._drivinghamiltonian(H, drives, t) for t in ts]
#    circuit = trottercircuit(Hts; ts =  ts)
#    ψT = runcircuit(ψ, circuit)
#    F₋ = fidelity(ψT, ϕ)
#
#    θ₀[k] += ϵ
#    ∇num = (F₊ - F₋) / (2*ϵ)
#    @test ∇num ≈ ∇alg[1][k]
#  end
#end
#
#
#@testset "OCC-parameters gradients: single state, many drives on many modes" begin
#
#  Random.seed!(1234)
#  N = 3
#  sites = qubits(N) 
#  
#  B = 0.5
#  H = OpSum()
#  # loop over the pauli operators
#  for j in 1:N-1
#    H += -1.0,"ZZ",(j,j+1)
#    H += -B,"X",j
#  end 
#  H += -B,"X",N
#  
#  T   = 1.0
#  δt  = 0.1 
#
#  f1(t, θ) = θ[1] * cos(t)   + θ[2] * cos(2.0 * t)
#  f2(t, θ) = θ[1] * cos(t/2) + θ[2] * cos(3.0 * t)
#  f3(t, θ) = θ[1] * cos(t/3) + θ[2] * cos(4.0 * t)
#
#  θ1 = [1.011,1.155]
#  θ2 = [1.22,0.98]
#  θ3 = [0.88,1.3]
#  θ = [θ1,θ2,θ3]
#  
#
#  ts = 0.0:δt:T
#  drives = [[f1,θ[1]] => ("Z",1), [f2,θ[2]] => ("Z",2), [f3,θ[3]] => ("Z",3)]
#  
#  Hts = [PastaQ._drivinghamiltonian(H, drives, t) for t in ts] 
#  circuit = trottercircuit(Hts; ts =  ts) 
#  
#  ψ = productstate(sites)
#  ϕ = normalize!(randomstate(sites))
#
#  cmap = PastaQ.circuitmap(circuit)
#  F, ∇alg = PastaQ.gradients(ψ, ϕ, circuit, drives, collect(ts), cmap)
#
#  ψT = runcircuit(ψ, circuit)
#  @test F ≈ fidelity(ψT, ϕ) atol = 1e-5
#
#  ϵ = 1e-5
#  for i in 1:length(drives)
#    for k in 1:length(θ[i])
#      θ[i][k] += ϵ
#      drives = [[f1,θ[1]] => ("Z",1), [f2,θ[2]] => ("Z",2), [f3,θ[3]] => ("Z",3)]
#      Hts = [PastaQ._drivinghamiltonian(H, drives, t) for t in ts]
#      circuit = trottercircuit(Hts; ts =  ts)
#      ψT = runcircuit(ψ, circuit)
#      F₊ = fidelity(ψT, ϕ)
#
#      θ[i][k] -= 2 * ϵ
#      drives = [[f1,θ[1]] => ("Z",1), [f2,θ[2]] => ("Z",2), [f3,θ[3]] => ("Z",3)]
#      Hts = [PastaQ._drivinghamiltonian(H, drives, t) for t in ts]
#      circuit = trottercircuit(Hts; ts =  ts)
#      ψT = runcircuit(ψ, circuit)
#      F₋ = fidelity(ψT, ϕ)
#
#      θ[i][k] += ϵ
#      ∇num = (F₊ - F₋) / (2*ϵ)
#      @test ∇num ≈ ∇alg[i][k]
#    end
#  end
#end



@testset "OCC-parameters gradients: many states, many drives on many modes" begin

  Random.seed!(1234)
  N = 3
  sites = qubits(N) 
  
  T   = 1.0
  δt  = 0.1 
  ts = 0.0:δt:T

  f1(t, θ) = θ[1] * cos(t)   + θ[2] * cos(2.0 * t)
  f2(t, θ) = θ[1] * cos(t/2) + θ[2] * cos(3.0 * t)
  f3(t, θ) = θ[1] * cos(t/3) + θ[2] * cos(4.0 * t)

  
  function generate_hamiltonian_sequence(parameters)
    B = 0.5
    Ht = OpSum[]
    for t in ts
      H = OpSum()
      # loop over the pauli operators
      for j in 1:N-1
        H += -1.0,"ZZ",(j,j+1)
        H += -B,"X",j
      end 
      H += -B,"X",N

      for parameter in parameters
        (f,θ) = first(parameter)
        μs = last(parameter)
        μs = μs isa Tuple ? [μs] : μs
        for μ in μs
          H += f(t, θ), μ..., (∇ = true,)
        end
      end
      push!(Ht, H)     
    end
    return Ht
  end
  
  θ1 = [1.011,1.155]
  θ2 = [1.22,0.98]
  θ3 = [0.88,1.3]
  θ⃗ = [θ1,θ2,θ3]
  #(f,θ₀) => [("Z",j) for j in 1:N] 
  parameters = [(f1,θ1) => [("Z",j) for j in 1:N],(f2,θ2) => ("Y",2),(f3,θ3) => ("CX",(1,2))]
  Ht = generate_hamiltonian_sequence(parameters)
  circuit = trottercircuit(Ht; ts = ts)
  
  ψs      = [productstate(sites)]
  push!(ψs, runcircuit(ψs[1], ("X",1)))
  push!(ψs, runcircuit(ψs[1], ("X",2)))
  push!(ψs, runcircuit(ψs[1], ("X",3)))
  ϕs = [normalize!(randomstate(sites)) for j in 1:length(ψs)]

  cmap = PastaQ.circuitmap(circuit)
  F, ∇alg = PastaQ.gradients(ψs, ϕs, circuit, parameters, collect(ts), cmap)

  function _cost(ψs, ϕs, circuit)
    O = 0.0
    for j in 1:length(ψs)
      ψ = runcircuit(ψs[j], circuit)
      O += inner(ψ, ϕs[j]) / length(ψs)
    end
    return O * conj(O)
  end
  @test F ≈ _cost(ψs, ϕs, circuit)
  ϵ = 1e-5
  for i in 1:length(parameters)
    for k in 1:length(θ⃗[i])
      θ⃗[i][k] += ϵ
      parameters = [(f1,θ1) => [("Z",j) for j in 1:N], (f2,θ2) => ("Y",2), (f3,θ3) => ("CX",(1,2))]
      Ht = generate_hamiltonian_sequence(parameters)
      circuit = trottercircuit(Ht; ts = ts)
      
      F₊ = _cost(ψs, ϕs, circuit)

      θ⃗[i][k] -= 2 * ϵ
      parameters = [(f1,θ1) => [("Z",j) for j in 1:N],(f2,θ2) => ("Y",2),(f3,θ3) => ("CX",(1,2))]
      Ht = generate_hamiltonian_sequence(parameters)
      circuit = trottercircuit(Ht; ts = ts)
      F₋ = _cost(ψs, ϕs, circuit)

      θ⃗[i][k] += ϵ
      ∇num = (F₊ - F₋) / (2*ϵ)
      @test ∇num ≈ ∇alg[i][k] atol = 1e-5
    end
  end
end

