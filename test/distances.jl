using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random


@testset "quantum state fidelity" begin
  N = 4
  circuit1 = randomcircuit(N,3)
  circuit2 = randomcircuit(N,3)
  # MPS wavefunction
  ψ1 = runcircuit(circuit1)
  ψ2 = runcircuit(trivialstate(ψ1),circuit2)
  # MPO density matrix
  ρ1 = runcircuit(trivialstate(ψ1), circuit1; noise = ("DEP",(p=0.01,)))
  ρ2 = runcircuit(trivialstate(ψ1), circuit2; noise = ("DEP",(p=0.01,)))
  # LPDO density matrix
  ϱ1 = normalize!(randomstate(ψ1; mixed = true))
  ϱ2 = normalize!(randomstate(ψ1; mixed = true))

  ψ1vec = PastaQ.array(ψ1)
  ρ1mat = PastaQ.array(ρ1)
  ϱ1mat = PastaQ.array(ϱ1)
  
  ψ2vec = PastaQ.array(ψ2)
  ρ2mat = PastaQ.array(ρ2)
  ϱ2mat = PastaQ.array(ϱ2)
  
  @test fidelity(ψ1,ψ2) ≈ abs2(ψ1vec' * ψ2vec)
  @test fidelity(ψ1,ρ2) ≈ ψ1vec' * ρ2mat * ψ1vec
  @test fidelity(ψ1,ϱ2) ≈ (ψ1vec' * ϱ2mat * ψ1vec) 
  
  @test fidelity(ρ1,ρ2) ≈ real(tr(sqrt(sqrt(ρ1mat)*ρ2mat*sqrt(ρ1mat))))^2 atol=1e-8 
  @test fidelity(ρ1,ϱ2) ≈ real(tr(sqrt(sqrt(ρ1mat)*ϱ2mat*sqrt(ρ1mat))))^2 atol=1e-8
  @test fidelity(ϱ1,ϱ2) ≈ real(tr(sqrt(sqrt(ϱ1mat)*ϱ2mat*sqrt(ϱ1mat))))^2 atol=1e-8 
end




@testset "quantum process fidelity" begin
  N = 3
  
  circuit1 = randomcircuit(N,3)
  circuit2 = randomcircuit(N,3)
  # MPO unitary 
  U1 = runcircuit(circuit1; process = true)
  U2 = randomprocess(U1)
  
  # MPO Choi matrix 
  ρ1 = PastaQ.choimatrix(PastaQ.hilbertspace(U1), circuit1; noise = ("DEP",(p=0.01,)))
  ρ2 = PastaQ.choimatrix(PastaQ.hilbertspace(U1), circuit2; noise = ("DEP",(p=0.01,)))
  # LPDO Choi matrix
  ϱ1 = normalize!(randomprocess(U1; mixed = true))
  ϱ2 = normalize!(randomprocess(U1; mixed = true))

  @disable_warn_order begin
    ϕ1 = PastaQ.unitary_mpo_to_choi_mps(U1)
    normalize!(ϕ1)
    ϕ1vec = PastaQ.array(ϕ1)
    ρ1mat = PastaQ.array(ρ1)
    ρ1mat = ρ1mat / tr(ρ1mat) 
    ϱ1mat = PastaQ.array(ϱ1)
    ϱ1mat = ϱ1mat / tr(ϱ1mat) 

    
    ϕ2 = PastaQ.unitary_mpo_to_choi_mps(U2)
    normalize!(ϕ2)
    ϕ2vec = PastaQ.array(ϕ2)
    ρ2mat = PastaQ.array(ρ2)
    ρ2mat = ρ2mat / tr(ρ2mat) 
    ϱ2mat = PastaQ.array(ϱ2)
    ϱ2mat = ϱ2mat / tr(ϱ2mat) 
    @test fidelity(U1,U2; process = true) ≈ abs2(ϕ1vec' * ϕ2vec)
    @test fidelity(U1,ρ2; process = true) ≈ ϕ1vec' * ρ2mat * ϕ1vec
    @test fidelity(U1,ϱ2; process = true) ≈ (ϕ1vec' * ϱ2mat * ϕ1vec) 
    
    @test fidelity(ρ1,ρ2; process = true) ≈ real(tr(sqrt(sqrt(ρ1mat)*ρ2mat*sqrt(ρ1mat))))^2 atol=1e-8 
    @test fidelity(ρ1,ϱ2; process = true) ≈ real(tr(sqrt(sqrt(ρ1mat)*ϱ2mat*sqrt(ρ1mat))))^2 atol=1e-8
    @test fidelity(ϱ1,ϱ2; process = true) ≈ real(tr(sqrt(sqrt(ϱ1mat)*ϱ2mat*sqrt(ϱ1mat))))^2 atol=1e-8
  end
end


@testset "frobenius distance" begin 

  N = 4
  Random.seed!(1111)
  ψ1 = randomstate(N;χ=2)
  Random.seed!(2222)
  ψ2 = randomstate(ψ1;χ=2)
  
  ρ_mpo = MPO(ψ1)
  σ_mpo = MPO(ψ2)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat) 
  Kσ = tr(σ_mat) 
  
  δ = ρ_mat/Kρ - σ_mat/Kσ

  T = sqrt(tr(conj(transpose(δ)) * δ))

  F = frobenius_distance(ρ_mpo,σ_mpo)
  @test T ≈ F
  @test F ≈ frobenius_distance(ψ1,σ_mpo)
  @test F ≈ frobenius_distance(ρ_mpo,ψ2)
  @test F ≈ frobenius_distance(ψ1,ψ2)
  
  Random.seed!(1111)
  ρ = randomstate(ψ1;mixed=true,χ=2,ξ=2)
  
  ρ_mpo = MPO(ρ)
  σ_mpo = MPO(ψ2)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat) 
  Kσ = tr(σ_mat) 
  
  δ = ρ_mat/Kρ - σ_mat/Kσ

  T = sqrt(tr(conj(transpose(δ)) * δ))

  F = frobenius_distance(ρ, σ_mpo)
  @test T ≈ F
  @test F ≈ frobenius_distance(ρ_mpo,ψ2)

  Random.seed!(1111)
  σ = randomstate(ψ1;mixed=true,χ=2,ξ=2)
  
  ρ_mpo = MPO(ψ1)
  σ_mpo = MPO(σ)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat) 
  Kσ = tr(σ_mat) 
  
  δ = ρ_mat/Kρ - σ_mat/Kσ

  T = sqrt(tr(conj(transpose(δ)) * δ))

  F = frobenius_distance(ρ_mpo,σ)
  @test T ≈ F
  @test F ≈ frobenius_distance(ψ1,σ_mpo)
  
  Random.seed!(1111)
  ρ = randomstate(N;mixed=true,χ=2,ξ=2)
  Random.seed!(1111)
  σ = randomstate(ρ;mixed=true,χ=2,ξ=2)
  
  ρ_mpo = MPO(ρ)
  σ_mpo = MPO(σ)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat) 
  Kσ = tr(σ_mat) 
  
  δ = ρ_mat/Kρ - σ_mat/Kσ

  T = sqrt(tr(conj(transpose(δ)) * δ))

  F = frobenius_distance(ρ,σ)
  @test T ≈ F
end

@testset "fidelity bound" begin 

  N = 4
  Random.seed!(1111)
  ψ1 = randomstate(N;χ=2)
  Random.seed!(2222)
  ψ2 = randomstate(ψ1;χ=2)
  
  ρ_mpo = MPO(ψ1)
  σ_mpo = MPO(ψ2)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat) 
  Kσ = tr(σ_mat) 
  
  f = tr(conj(transpose(ρ_mat/Kρ)) * (σ_mat/Kσ))
  F̃ = fidelity_bound(ρ_mpo,σ_mpo)
  @test f ≈ F̃
  @test F̃ ≈ fidelity(ψ1,σ_mpo)
  @test F̃ ≈ fidelity(ρ_mpo,ψ2)
  @test F̃ ≈ fidelity(ψ1,ψ2)
   
  Random.seed!(1111)
  ρ = randomstate(ψ1;mixed=true,χ=2,ξ=2)
  
  ρ_mpo = MPO(ρ)
  σ_mpo = MPO(ψ2)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat) 
  Kσ = tr(σ_mat) 

  f = tr(conj(transpose(ρ_mat/Kρ)) * (σ_mat/Kσ))
  F̃ = fidelity_bound(ρ,σ_mpo)
  @test f ≈ F̃
  @test F̃ ≈ fidelity(ρ_mpo,ψ2)

  Random.seed!(1111)
  σ = randomstate(ψ1;mixed=true,χ=2,ξ=2)
  
  ρ_mpo = MPO(ψ1)
  σ_mpo = MPO(σ)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat) 
  Kσ = tr(σ_mat) 
  f = tr(conj(transpose(ρ_mat/Kρ)) * (σ_mat/Kσ))
  F̃ = fidelity_bound(ρ_mpo,σ)
  @test f ≈ F̃
  @test F̃ ≈ fidelity(ψ1,σ_mpo)

  Random.seed!(1111)
  ρ = randomstate(N;mixed=true,χ=2,ξ=2)
  Random.seed!(1111)
  σ = randomstate(ρ;mixed=true,χ=2,ξ=2)
  
  ρ_mpo = MPO(ρ)
  σ_mpo = MPO(σ)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat) 
  Kσ = tr(σ_mat) 
  
  f = tr(conj(transpose(ρ_mat/Kρ)) * (σ_mat/Kσ))
  F̃ = fidelity_bound(ρ,σ)
  @test f ≈ F̃
end


