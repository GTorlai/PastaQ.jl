using PastaQ
using ITensors
using Test
using Random
import Optimisers


@testset "Optimizer update" begin
  θ = rand(10)
  ∇ = rand(10)
  θtest = copy(θ)
  opt = Optimisers.Descent(0.01)
  st  = Optimisers.state(opt, θ)
  st, θ = Optimisers.update(opt, st, θ, ∇)
  @test θ ≈ (θtest - 0.01*∇)
end


@testset "get/set parameters" begin
  N = 3
  sites = siteinds("Qubit",N)
  
  # MPS WAVEFUNCTION
  ψ = randomstate(sites; χ = 4)
  θ = PastaQ.getparameters(LPDO(ψ))
  ϕ = LPDO(randomstate(sites; χ = 4))
  @test !(ψ ≈ ϕ.X)
  PastaQ.setparameters!(ϕ, θ)
  @test PastaQ.array(ψ) ≈ PastaQ.array(ϕ.X)

  # LPDO (state) 
  ρ = randomstate(sites; χ = 4, ξ = 3)
  θ = PastaQ.getparameters(ρ)
  σ = randomstate(sites; χ = 4, ξ = 3) 
  PastaQ.setparameters!(σ, θ)
  @test PastaQ.array(ρ) ≈ PastaQ.array(σ)
  
  # UNITARY 
  U = LPDO(PastaQ.unitary_mpo_to_choi_mps(randomprocess(sites; χ = 4)))
  θ = PastaQ.getparameters(U)
  V = LPDO(PastaQ.unitary_mpo_to_choi_mps(randomprocess(sites; χ = 4)))
  PastaQ.setparameters!(V, θ)
  @test PastaQ.array(U.X) ≈ PastaQ.array(V.X)

  # CHOI
  Λ = randomprocess(sites; χ = 4, ξ = 3)
  θ = PastaQ.getparameters(Λ)
  Γ = randomprocess(sites; χ = 4, ξ = 3)
  PastaQ.setparameters!(Γ, θ)
  @test PastaQ.array(Γ) ≈ PastaQ.array(Λ)
end

@testset "mps-qst: SGD update" begin
  N = 3
  χ = 10
  nsamples = 10
  data = PastaQ.convertdatapoints(randompreparations(N, nsamples))

  ψ = randomstate(N; χ=χ)
  opt = Optimisers.Descent(0.1) 
  st = PastaQ.state(opt, ψ)
  ∇, _ = PastaQ.gradients(LPDO(ψ), data)
   
  ϕ = LPDO(copy(ψ))
  ϕ = PastaQ.update!(ϕ, ∇, (opt,st)) 
  ψp = copy(ψ) 
  for j in 1:N
    ψp[j] = ψp[j] - 0.1*∇[j]
  end
  @test PastaQ.array(ϕ.X) ≈ PastaQ.array(ψp)
end

@testset "lpdo-qst: SGD update" begin
  N = 3
  χ = 10
  nsamples = 10
  data = PastaQ.convertdatapoints(randompreparations(N, nsamples))

  ρ = randomstate(N; χ=χ,ξ = 2)
  opt = Optimisers.Descent(0.1) 
  st = PastaQ.state(opt, ρ)
  ∇, _ = PastaQ.gradients(ρ, data)
  
  γ = copy(ρ)
  γ = PastaQ.update!(γ, ∇, (opt,st)) 
  ρp = copy(ρ) 
  for j in 1:N
    ρp.X[j] = ρp.X[j] - 0.1*∇[j]
  end
  @test PastaQ.array(γ) ≈ PastaQ.array(ρp)
end

@testset "mpo-qpt: SGD update" begin
  N = 3
  χ = 4
  nsamples = 10
  trace_preserving_regularizer = 0.1
  Random.seed!(1234)
  data_in = randompreparations(N, nsamples)
  data_out = PastaQ.convertdatapoints(randompreparations(N, nsamples))
  data = data_in .=> data_out

  U = randomprocess(N; χ=χ)
  Φ = LPDO(PastaQ.unitary_mpo_to_choi_mps(U))
  PastaQ.normalize!(Φ; localnorm=2)
  
  opt = Optimisers.Descent(0.1) 
  st = PastaQ.state(opt, Φ)
  ∇, _ = PastaQ.gradients(Φ, data)

  γ = copy(Φ)
  γ = PastaQ.update!(γ, ∇, (opt,st)) 
  Φp = copy(Φ) 
  for j in 1:N
    Φp.X[j] = Φp.X[j] - 0.1*∇[j]
  end
  @test PastaQ.array(γ.X) ≈ PastaQ.array(Φp.X)
end

@testset "lpdo-qpt: SGD update" begin
  N = 3
  χ = 4
  nsamples = 10
  trace_preserving_regularizer = 0.1
  Random.seed!(1234)
  data_in = randompreparations(N, nsamples)
  data_out = PastaQ.convertdatapoints(randompreparations(N, nsamples))
  data = data_in .=> data_out

  Λ = randomprocess(N; χ=χ, ξ = 3)
  PastaQ.normalize!(Λ; localnorm=2)
  
  opt = Optimisers.Descent(0.1) 
  st = PastaQ.state(opt, Λ)
  
  ∇, _ = PastaQ.gradients(Λ, data)

  γ = copy(Λ)
  γ = PastaQ.update!(γ, ∇, (opt,st)) 
  Λp = copy(Λ) 
  for j in 1:N
    Λp.X[j] = Λp.X[j] - 0.1*∇[j]
  end
  @test PastaQ.array(γ) ≈ PastaQ.array(Λp)
end

