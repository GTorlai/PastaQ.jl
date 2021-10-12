using PastaQ
using ITensors
using Test
using Random
using Printf
using Zygote: Zygote


function isingmodel(N::Int)
  # generate the hamiltonian MPO
  sites = siteinds("Qubit",N)
  ampo = AutoMPO()
  
  # loop over the pauli operators
  for j in 1:N-1
    ampo .+= -1.0,"Z",j,"Z",j+1 
    ampo .+= -1.0,"X",j
  end
  ampo .+= -1.0,"X",N
  H = MPO(ampo,sites)
  
  # find ground state with DMRG
  mps = randomMPS(sites)
  sweeps = Sweeps(10)
  maxdim!(sweeps, 10,20,30,50,100)
  cutoff!(sweeps, 1E-10)
  E0, mps = dmrg(H, mps, sweeps, outputlevel = 0);
  return H,mps  
end


#@testset "OCC-parameters gradients: single state, single drive on single mode" begin
#  Random.seed!(1234)
#  N = 3
#  sites = siteinds("Qubit",N)
#  M   = 100
#  T   = 1.0
#  δt  = T / M
#  Jxx = 1.2
#  Jz  = 0.45
#
#  f(θ, t) = θ[1] * cos(t) + θ[2] * cos(2.0 * t)
#
#  θ₀ = [1.011,1.155]
#
#  function makecircuit(θ::Vector)
#    circuit = Vector{Tuple}[]
#    for m in 1:M
#      t = δt * m
#      layer = Tuple[]
#      push!(layer, ("trotter_XX", (1,2), (τ = δt/2, X = Jxx)))
#      push!(layer, ("trotter_XX", (2,3), (τ = δt/2, X = Jxx)))
#      push!(layer, ("trotter_Z" , 1, (τ = δt/2, X = Jz )))
#      push!(layer, ("trotter_Z" , 2, (τ = δt/2, X = f(θ,t), ∇ = true)))
#      push!(layer, ("trotter_Z" , 3, (τ = δt/2, X = Jz )))
#      append!(layer, reverse(layer))
#      push!(circuit, layer)
#    end
#    return circuit
#  end
#
#  circuit = makecircuit(θ₀)
#  ψ₀      = productstate(sites)
#  ψtarget = normalize!(randomstate(sites))
#
#  drives = [f,θ₀] => ("Z",2)
#  cmap = PastaQ.circuitmap(circuit)
#
#  F, ∇alg = PastaQ.gradients(ψ₀, ψtarget, circuit, drives, δt, cmap)
#
#  ψ = runcircuit(ψ₀, circuit)
#  @test F ≈ fidelity(ψ, ψtarget)
#
#  ϵ = 1e-5
#  for k in 1:length(θ₀)
#    θ₀[k] += ϵ
#    circuit = makecircuit(θ₀)
#    ψ = runcircuit(ψ₀, circuit)
#    F₊ = fidelity(ψ, ψtarget)
#
#    θ₀[k] -= 2 * ϵ
#    circuit = makecircuit(θ₀)
#    ψ = runcircuit(ψ₀, circuit)
#    F₋ = fidelity(ψ, ψtarget)
#
#    θ₀[k] += ϵ
#    ∇num = (F₊ - F₋) / (2*ϵ)
#    @test ∇num ≈ ∇alg[1][k]
#  end
#end
#
#
#
#@testset "OCC-parameters gradients: single state, single drive on many modes" begin
#  Random.seed!(1234)
#  N = 3
#  sites = siteinds("Qubit",N)
#  M   = 100
#  T   = 1.0
#  δt  = T / M
#  Jxx = 1.2
#  Jz  = 0.45
#
#  f(θ, t) = θ[1] * cos(t) + θ[2] * cos(2.0 * t)
#
#  θ₀ = [1.011,1.155]
#
#  function makecircuit(θ::Vector)
#    circuit = Vector{Tuple}[]
#    for m in 1:M
#      t = δt * m
#      layer = Tuple[]
#      push!(layer, ("trotter_XX", (1,2), (τ = δt/2, X = Jxx)))
#      push!(layer, ("trotter_XX", (2,3), (τ = δt/2, X = Jxx)))
#      push!(layer, ("trotter_Z" , 1, (τ = δt/2, X = f(θ,t), ∇ = true )))
#      push!(layer, ("trotter_Z" , 2, (τ = δt/2, X = f(θ,t), ∇ = true)))
#      push!(layer, ("trotter_Z" , 3, (τ = δt/2, X = f(θ,t), ∇ = true )))
#      append!(layer, reverse(layer))
#      push!(circuit, layer)
#    end
#    return circuit
#  end
#
#  circuit = makecircuit(θ₀)
#  ψ₀      = productstate(sites)
#  ψtarget = normalize!(randomstate(sites))
#
#  #drives = f => [("Z",j) for j in 1:N]
#  drives = [f,θ₀] => [("Z",j) for j in 1:N]
#  cmap = PastaQ.circuitmap(circuit)
#
#  F, ∇alg = PastaQ.gradients(ψ₀, ψtarget, circuit, drives, δt, cmap)
#
#  ψ = runcircuit(ψ₀, circuit)
#  @test F ≈ fidelity(ψ, ψtarget)
#
#  ϵ = 1e-5
#  for k in 1:length(θ₀)
#    θ₀[k] += ϵ
#    circuit = makecircuit(θ₀)
#    ψ = runcircuit(ψ₀, circuit)
#    F₊ = fidelity(ψ, ψtarget)
#
#    θ₀[k] -= 2 * ϵ
#    circuit = makecircuit(θ₀)
#    ψ = runcircuit(ψ₀, circuit)
#    F₋ = fidelity(ψ, ψtarget)
#
#    θ₀[k] += ϵ
#    ∇num = (F₊ - F₋) / (2*ϵ)
#    @test ∇num ≈ ∇alg[1][k]
#  end
#end
#
#
#
#@testset "OCC-parameters gradients: single state, many drives on many modes" begin
#  Random.seed!(1234)
#  N = 3
#  sites = siteinds("Qubit",N)
#  M   = 100
#  T   = 1.0
#  δt  = T / M
#  Jxx = 1.2
#  Jz  = 0.45
#
#  f1(θ, t) = θ[1] * cos(t) + θ[2] * cos(2.0 * t)
#  f2(θ, t) = θ[1] * cos(t/2) + θ[2] * cos(3.0 * t)
#  f3(θ, t) = θ[1] * cos(t/3) + θ[2] * cos(4.0 * t)
#
#  θ1 = [1.011,1.155]
#  θ2 = [1.22,0.98]
#  θ3 = [0.88,1.3]
#  θ = [θ1,θ2,θ3]
#
#  function makecircuit(θ::Vector)
#    circuit = Vector{Tuple}[]
#    for m in 1:M
#      t = δt * m
#      layer = Tuple[]
#      push!(layer, ("trotter_XX", (1,2), (τ = δt/2, X = Jxx)))
#      push!(layer, ("trotter_XX", (2,3), (τ = δt/2, X = Jxx)))
#      push!(layer, ("trotter_Z" , 1, (τ = δt/2, X = f1(θ[1],t), ∇ = true )))
#      push!(layer, ("trotter_Z" , 2, (τ = δt/2, X = f2(θ[2],t), ∇ = true)))
#      push!(layer, ("trotter_Z" , 3, (τ = δt/2, X = f3(θ[3],t), ∇ = true )))
#      append!(layer, reverse(layer))
#      push!(circuit, layer)
#    end
#    return circuit
#  end
#
#  circuit = makecircuit(θ)
#  ψ₀      = productstate(sites)
#  ψtarget = normalize!(randomstate(sites))
#
#  drives = [[f1,θ1] => ("Z",1), [f2,θ2] => ("Z",2), [f3,θ3] => ("Z",3)]
#  #drives = [f1 => ("Z",1), f2 => ("Z",2), f3 => ("Z",3)]
#  cmap = PastaQ.circuitmap(circuit)
#
#  F, ∇alg = PastaQ.gradients(ψ₀, ψtarget, circuit, drives, δt, cmap)
#
#  ψ = runcircuit(ψ₀, circuit)
#  @test F ≈ fidelity(ψ, ψtarget)
#
#  ϵ = 1e-5
#  for i in 1:length(drives)
#    for k in 1:length(θ[i])
#      θ[i][k] += ϵ
#      circuit = makecircuit(θ)
#      ψ = runcircuit(ψ₀, circuit)
#      F₊ = fidelity(ψ, ψtarget)
#
#      θ[i][k] -= 2 * ϵ
#      circuit = makecircuit(θ)
#      ψ = runcircuit(ψ₀, circuit)
#      F₋ = fidelity(ψ, ψtarget)
#
#      θ[i][k] += ϵ
#      ∇num = (F₊ - F₋) / (2*ϵ)
#      @test ∇num ≈ ∇alg[i][k]
#    end
#  end
#end
#
#
#@testset "OCC-parameters gradients: many states, many drives on many modes" begin
#  Random.seed!(1234)
#  N = 3
#  sites = siteinds("Qubit",N)
#  M   = 100
#  T   = 1.0
#  δt  = T / M
#  Jxx = 1.2
#  Jz  = 0.45
#
#  f1(θ, t) = θ[1] * cos(t) + θ[2] * cos(2.0 * t)
#  f2(θ, t) = θ[1] * cos(t/2) + θ[2] * cos(3.0 * t)
#  f3(θ, t) = θ[1] * cos(t/3) + θ[2] * cos(4.0 * t)
#
#  θ1 = [1.011,1.155]
#  θ2 = [1.22,0.98]
#  θ3 = [0.88,1.3]
#  θ = [θ1,θ2,θ3]
#
#  function makecircuit(θ::Vector)
#    circuit = Vector{Tuple}[]
#    for m in 1:M
#      t = δt * m
#      layer = Tuple[]
#      push!(layer, ("trotter_XX", (1,2), (τ = δt/2, X = Jxx)))
#      push!(layer, ("trotter_XX", (2,3), (τ = δt/2, X = Jxx)))
#      push!(layer, ("trotter_Z" , 1, (τ = δt/2, X = f1(θ[1],t), ∇ = true )))
#      push!(layer, ("trotter_Z" , 2, (τ = δt/2, X = f2(θ[2],t), ∇ = true)))
#      push!(layer, ("trotter_Z" , 3, (τ = δt/2, X = f3(θ[3],t), ∇ = true )))
#      append!(layer, reverse(layer))
#      push!(circuit, layer)
#    end
#    return circuit
#  end
#
#  circuit = makecircuit(θ)
#  ψ₀      = [productstate(sites)]
#  push!(ψ₀, runcircuit(ψ₀[1], ("X",1)))
#  push!(ψ₀, runcircuit(ψ₀[1], ("X",2)))
#  push!(ψ₀, runcircuit(ψ₀[1], ("X",3)))
#  ψtarget = [normalize!(randomstate(sites)) for j in 1:length(ψ₀)]
#
#  drives = [[f1,θ1] => ("Z",1), [f2,θ2] => ("Z",2), [f3,θ3] => ("Z",3)]
#
#
#  cmap = PastaQ.circuitmap(circuit)
#
#  F, ∇alg = PastaQ.gradients(ψ₀, ψtarget, circuit, drives, δt, cmap)
#
#
#  function _cost(ψ₀, ψtarget, circuit)
#    O = 0.0
#    for j in 1:length(ψ₀)
#      ψ = runcircuit(ψ₀[j], circuit)
#      O += inner(ψ, ψtarget[j]) / length(ψ₀)
#    end
#    return O * conj(O)
#  end
#  @test F ≈ _cost(ψ₀, ψtarget, circuit)
#
#  ϵ = 1e-5
#
#  for i in 1:length(drives)
#    for k in 1:length(θ[i])
#      θ[i][k] += ϵ
#      circuit = makecircuit(θ)
#      F₊ = _cost(ψ₀, ψtarget, circuit)
#
#      θ[i][k] -= 2 * ϵ
#      circuit = makecircuit(θ)
#      F₋ = _cost(ψ₀, ψtarget, circuit)
#
#      θ[i][k] += ϵ
#      ∇num = (F₊ - F₋) / (2*ϵ)
#      @test ∇num ≈ ∇alg[i][k]
#    end
#  end
#end
