using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

@testset "fidelity" begin
  """ F = |<PSI1|PSI2>|^2 """
  
  N = 3
  χ = 4
  Random.seed!(1111)
  ψ1 = initializetomography(N;χ=χ)
  ψ2 = copy(ψ1)
  ψ2[1] = ITensor(ones(2,4),inds(ψ2[1])[1],inds(ψ2[1])[2])
  
  ψ1_vec = fullvector(ψ1)
  ψ2_vec = fullvector(ψ2)
 
  K1 = sum(ψ1_vec .* conj(ψ1_vec)) 
  ψ1_vec ./= sqrt(K1)
  K2 = sum(ψ2_vec .* conj(ψ2_vec)) 
  ψ2_vec ./= sqrt(K2)
  
  ex_F = abs2(dot(ψ1_vec ,ψ2_vec))
  F = fidelity(ψ1,ψ2)
  
  @test ex_F ≈ F
  
  gates = randomcircuit(N,2)
  Φ1 = choimatrix(N,gates)
  F = fidelity(Φ1,Φ1)
  @test F ≈ 1.0

  """ F = <PSI|RHO|PSI> """
  N = 3
  χ = 2
  ψ = initializetomography(N;χ=χ)
  ψ_vec = fullvector(ψ)   
  
  K = sum(ψ_vec .* conj(ψ_vec))
  ψ_vec ./= sqrt(K)
  
  ξ = 2
  ρ = initializetomography(ψ;χ=χ,ξ=ξ)
  
  ρ_mat = fullmatrix(MPO(ρ))
  J = tr(ρ_mat)
  ρ_mat ./= J

  ex_F = dot(ψ_vec, ρ_mat * ψ_vec)
  F = fidelity(ρ, ψ)
  @test F ≈ ex_F
end

@testset "frobenius distance" begin 

  N = 4
  Random.seed!(1111)
  ψ1 = initializetomography(N;χ=2)
  Random.seed!(2222)
  ψ2 = initializetomography(ψ1;χ=2)
  
  ρ_mpo = MPO(ψ1)
  σ_mpo = MPO(ψ2)

  ρ_mat = fullmatrix(ρ_mpo)
  σ_mat = fullmatrix(σ_mpo)
  Kρ = tr(ρ_mat) 
  Kσ = tr(σ_mat) 
  
  δ = ρ_mat/Kρ - σ_mat/Kσ

  T = sqrt(tr(conj(transpose(δ)) * δ))

  F = frobenius_distance(ρ_mpo,σ_mpo)
  @test T ≈ F
    
  Random.seed!(1111)
  ρ = initializetomography(ψ1;χ=2,ξ=2)
  
  ρ_mpo = MPO(ρ)
  σ_mpo = MPO(ψ2)

  ρ_mat = fullmatrix(ρ_mpo)
  σ_mat = fullmatrix(σ_mpo)
  Kρ = tr(ρ_mat) 
  Kσ = tr(σ_mat) 
  
  δ = ρ_mat/Kρ - σ_mat/Kσ

  T = sqrt(tr(conj(transpose(δ)) * δ))

  F = frobenius_distance(ρ, σ_mpo)
  @test T ≈ F


  Random.seed!(1111)
  σ = initializetomography(ψ1;χ=2,ξ=2)
  
  ρ_mpo = MPO(ψ1)
  σ_mpo = MPO(σ)

  ρ_mat = fullmatrix(ρ_mpo)
  σ_mat = fullmatrix(σ_mpo)
  Kρ = tr(ρ_mat) 
  Kσ = tr(σ_mat) 
  
  δ = ρ_mat/Kρ - σ_mat/Kσ

  T = sqrt(tr(conj(transpose(δ)) * δ))

  F = frobenius_distance(ρ_mpo,σ)
  @test T ≈ F
  
  Random.seed!(1111)
  ρ = initializetomography(N;χ=2,ξ=2)
  Random.seed!(1111)
  σ = initializetomography(ρ;χ=2,ξ=2)
  
  ρ_mpo = MPO(ρ)
  σ_mpo = MPO(σ)

  ρ_mat = fullmatrix(ρ_mpo)
  σ_mat = fullmatrix(σ_mpo)
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
  ψ1 = initializetomography(N;χ=2)
  Random.seed!(2222)
  ψ2 = initializetomography(ψ1;χ=2)
  
  ρ_mpo = MPO(ψ1)
  σ_mpo = MPO(ψ2)

  ρ_mat = fullmatrix(ρ_mpo)
  σ_mat = fullmatrix(σ_mpo)
  Kρ = tr(ρ_mat) 
  Kσ = tr(σ_mat) 
  
  f = tr(conj(transpose(ρ_mat/Kρ)) * (σ_mat/Kσ))
  F̃ = fidelity_bound(ρ_mpo,σ_mpo)
  @test f ≈ F̃
    
  Random.seed!(1111)
  ρ = initializetomography(ψ1;χ=2,ξ=2)
  
  ρ_mpo = MPO(ρ)
  σ_mpo = MPO(ψ2)

  ρ_mat = fullmatrix(ρ_mpo)
  σ_mat = fullmatrix(σ_mpo)
  Kρ = tr(ρ_mat) 
  Kσ = tr(σ_mat) 

  f = tr(conj(transpose(ρ_mat/Kρ)) * (σ_mat/Kσ))
  F̃ = fidelity_bound(ρ,σ_mpo)
  @test f ≈ F̃


  Random.seed!(1111)
  σ = initializetomography(ψ1;χ=2,ξ=2)
  
  ρ_mpo = MPO(ψ1)
  σ_mpo = MPO(σ)

  ρ_mat = fullmatrix(ρ_mpo)
  σ_mat = fullmatrix(σ_mpo)
  Kρ = tr(ρ_mat) 
  Kσ = tr(σ_mat) 
  f = tr(conj(transpose(ρ_mat/Kρ)) * (σ_mat/Kσ))
  F̃ = fidelity_bound(ρ_mpo,σ)
  @test f ≈ F̃
  
  Random.seed!(1111)
  ρ = initializetomography(N;χ=2,ξ=2)
  Random.seed!(1111)
  σ = initializetomography(ρ;χ=2,ξ=2)
  
  ρ_mpo = MPO(ρ)
  σ_mpo = MPO(σ)

  ρ_mat = fullmatrix(ρ_mpo)
  σ_mat = fullmatrix(σ_mpo)
  Kρ = tr(ρ_mat) 
  Kσ = tr(σ_mat) 
  
  f = tr(conj(transpose(ρ_mat/Kρ)) * (σ_mat/Kσ))
  F̃ = fidelity_bound(ρ,σ)
  @test f ≈ F̃
  
end


