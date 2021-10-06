using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

@testset "quantum state fidelity: normalized input states" begin
  N = 4
  circuit1 = randomcircuit(N, 3)
  circuit2 = randomcircuit(N, 3)
  
  sites = siteinds("Qubit", N)
  # MPS wavefunction
  ψ1 = runcircuit(sites, circuit1)
  ψ2 = runcircuit(sites, circuit2)
  # MPO density matrix
  ρ1 = runcircuit(sites, circuit1; noise=("DEP", (p=0.01,)))
  ρ2 = runcircuit(sites, circuit2; noise=("DEP", (p=0.01,)))
  # LPDO density matrix
  ϱ1 = normalize!(randomstate(sites; χ = 2, ξ = 3))
  ϱ2 = normalize!(randomstate(sites; χ = 2, ξ = 3))

  ψ1vec = PastaQ.array(ψ1)
  ρ1mat = PastaQ.array(ρ1)
  ϱ1mat = PastaQ.array(ϱ1)

  ψ2vec = PastaQ.array(ψ2)
  ρ2mat = PastaQ.array(ρ2)
  ϱ2mat = PastaQ.array(ϱ2)

  @test fidelity(ψ1, ψ2) ≈ abs2(ψ1vec' * ψ2vec)
  @test fidelity(ψ1, ρ2) ≈ ψ1vec' * ρ2mat * ψ1vec
  @test fidelity(ψ1, ϱ2) ≈ real((ψ1vec' * ϱ2mat * ψ1vec))

  @test fidelity(ρ1, ρ2) ≈ real(tr(sqrt(sqrt(ρ1mat) * ρ2mat * sqrt(ρ1mat))))^2 atol = 1e-7
  @test fidelity(ρ1, ϱ2) ≈ real(tr(sqrt(sqrt(ρ1mat) * ϱ2mat * sqrt(ρ1mat))))^2 atol = 1e-7
  @test fidelity(ϱ1, ϱ2) ≈ real(tr(sqrt(sqrt(ϱ1mat) * ϱ2mat * sqrt(ϱ1mat))))^2 atol = 1e-7

  # ITensors

  ψ1prod = prod(ψ1)
  ψ2prod = prod(ψ2)
  ρ1prod = prod(ρ1)
  ρ2prod = prod(ρ2)
  ϱ1prod = prod(ϱ1)
  ϱ2prod = prod(ϱ2)

  @test fidelity(ψ1prod, ψ2prod) ≈ fidelity(ψ1, ψ2)  
  @test fidelity(ψ1prod, ψ2) ≈ fidelity(ψ1, ψ2)
  @test fidelity(ψ1, ψ2prod) ≈ fidelity(ψ1, ψ2)

  @test fidelity(ψ1prod, ρ2prod) ≈ fidelity(ψ1, ρ2) 
  @test fidelity(ψ1prod, ρ2) ≈ fidelity(ψ1, ρ2) 
  @test fidelity(ψ1, ρ2prod) ≈ fidelity(ψ1, ρ2) 
  
  @test fidelity(ψ1prod, ϱ2prod) ≈ fidelity(ψ1, ϱ2) 
  @test fidelity(ψ1prod, ϱ2) ≈ fidelity(ψ1, ϱ2) 
  @test fidelity(ψ1, ϱ2prod) ≈ fidelity(ψ1, ϱ2) 
                                           
  @test fidelity(ρ1prod, ρ2prod) ≈ fidelity(ρ1, ρ2) 
  @test fidelity(ρ1, ρ2prod) ≈ fidelity(ρ1, ρ2) 
  @test fidelity(ρ1prod, ρ2) ≈ fidelity(ρ1, ρ2) 
  
  @test fidelity(ρ1prod, ϱ2prod) ≈ fidelity(ρ1, ϱ2) 
  @test fidelity(ρ1, ϱ2prod) ≈ fidelity(ρ1, ϱ2) 
  @test fidelity(ρ1prod, ϱ2) ≈ fidelity(ρ1, ϱ2) 

  @test fidelity(ϱ1prod, ϱ2prod) ≈ fidelity(ϱ1, ϱ2) 
  @test fidelity(ϱ1prod, ϱ2) ≈ fidelity(ϱ1, ϱ2) 
  @test fidelity(ϱ1, ϱ2prod) ≈ fidelity(ϱ1, ϱ2) 

end


@testset "quantum state fidelity: unnormalized input states" begin
  N = 4
  sites = siteinds("Qubit",N)
  # MPS wavefunction
  ψ1 = randomstate(sites; χ = 4) 
  ψ2 = randomstate(sites; χ = 5) 
  # LPDO density matrix
  ϱ1 = randomstate(sites; χ = 5, ξ = 2)
  ϱ2 = randomstate(sites; χ = 5, ξ = 3)
  
  # MPO density matrix
  ρ1 = MPO(ϱ1) 
  ρ2 = MPO(ϱ2)

  ψ1vec = PastaQ.array(normalize!(copy(ψ1)))
  ρ1mat = PastaQ.array(normalize!(copy(ρ1)))  
  ϱ1mat = PastaQ.array(normalize!(copy(ϱ1)))

  ψ2vec = PastaQ.array(normalize!(copy(ψ2)))
  ρ2mat = PastaQ.array(normalize!(copy(ρ2)))
  ϱ2mat = PastaQ.array(normalize!(copy(ϱ2)))

  @test fidelity(ψ1, ψ2) ≈ abs2(ψ1vec' * ψ2vec)
  @test fidelity(ψ1, ρ2) ≈ ψ1vec' * ρ2mat * ψ1vec
  @test fidelity(ψ1, ϱ2) ≈ real((ψ1vec' * ϱ2mat * ψ1vec))

  @test fidelity(ρ1, ρ2) ≈ real(tr(sqrt(sqrt(ρ1mat) * ρ2mat * sqrt(ρ1mat))))^2 atol = 1e-7
  @test fidelity(ρ1, ϱ2) ≈ real(tr(sqrt(sqrt(ρ1mat) * ϱ2mat * sqrt(ρ1mat))))^2 atol = 1e-7
  @test fidelity(ϱ1, ϱ2) ≈ real(tr(sqrt(sqrt(ϱ1mat) * ϱ2mat * sqrt(ϱ1mat))))^2 atol = 1e-7

  # ITensors

  ψ1prod = prod(ψ1)
  ψ2prod = prod(ψ2)
  ρ1prod = prod(ρ1)
  ρ2prod = prod(ρ2)
  ϱ1prod = prod(ϱ1)
  ϱ2prod = prod(ϱ2)

  @test fidelity(ψ1prod, ψ2prod) ≈ fidelity(ψ1, ψ2)  
  @test fidelity(ψ1prod, ψ2) ≈ fidelity(ψ1, ψ2)
  @test fidelity(ψ1, ψ2prod) ≈ fidelity(ψ1, ψ2)

  @test fidelity(ψ1prod, ρ2prod) ≈ fidelity(ψ1, ρ2) 
  @test fidelity(ψ1prod, ρ2) ≈ fidelity(ψ1, ρ2) 
  @test fidelity(ψ1, ρ2prod) ≈ fidelity(ψ1, ρ2) 
  
  @test fidelity(ψ1prod, ϱ2prod) ≈ fidelity(ψ1, ϱ2) 
  @test fidelity(ψ1prod, ϱ2) ≈ fidelity(ψ1, ϱ2) 
  @test fidelity(ψ1, ϱ2prod) ≈ fidelity(ψ1, ϱ2) 
                                           
  @test fidelity(ρ1prod, ρ2prod) ≈ fidelity(ρ1, ρ2) 
  @test fidelity(ρ1, ρ2prod) ≈ fidelity(ρ1, ρ2) 
  @test fidelity(ρ1prod, ρ2) ≈ fidelity(ρ1, ρ2) 
  
  @test fidelity(ρ1prod, ϱ2prod) ≈ fidelity(ρ1, ϱ2) 
  @test fidelity(ρ1, ϱ2prod) ≈ fidelity(ρ1, ϱ2) 
  @test fidelity(ρ1prod, ϱ2) ≈ fidelity(ρ1, ϱ2) 

  @test fidelity(ϱ1prod, ϱ2prod) ≈ fidelity(ϱ1, ϱ2) 
  @test fidelity(ϱ1prod, ϱ2) ≈ fidelity(ϱ1, ϱ2) 
  @test fidelity(ϱ1, ϱ2prod) ≈ fidelity(ϱ1, ϱ2) 

end

@testset "quantum process fidelity" begin
  N = 3
  sites = siteinds("Qubit",N)

  circuit1 = randomcircuit(N, 3)
  circuit2 = randomcircuit(N, 3)
  # MPO unitary 
  U1 = runcircuit(sites, circuit1; process=true)
  U2 = randomprocess(sites)

  # MPO Choi matrix 
  ρ1 = PastaQ.choimatrix(sites, circuit1; noise=("DEP", (p=0.01,)))
  ρ2 = PastaQ.choimatrix(sites, circuit2; noise=("DEP", (p=0.01,)))
  # LPDO Choi matrix
  ϱ1 = normalize!(randomprocess(sites; mixed=true))
  ϱ2 = normalize!(randomprocess(sites; mixed=true))

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
    
    
    @test fidelity(U1, U2; process=true) ≈ abs2(ϕ1vec' * ϕ2vec)
    @test fidelity(U1, ρ2; process=true) ≈ ϕ1vec' * ρ2mat * ϕ1vec
    @test fidelity(U1, ϱ2; process=true) ≈ (ϕ1vec' * ϱ2mat * ϕ1vec)

    @test fidelity(ρ1, ρ2; process=true) ≈
          real(tr(sqrt(sqrt(ρ1mat) * ρ2mat * sqrt(ρ1mat))))^2 atol = 1e-7
    @test fidelity(ρ1, ϱ2; process=true) ≈
          real(tr(sqrt(sqrt(ρ1mat) * ϱ2mat * sqrt(ρ1mat))))^2 atol = 1e-7
    @test fidelity(ϱ1, ϱ2; process=true, cutoff = 1e-14) ≈
          real(tr(sqrt(sqrt(ϱ1mat) * ϱ2mat * sqrt(ϱ1mat))))^2 atol = 1e-7

    # ITensors
    U1prod = prod(U1) 
    U2prod = prod(U2) 
    ρ1prod = prod(ρ1)
    ρ2prod = prod(ρ2)
    ϱ1prod = prod(ϱ1)
    ϱ2prod = prod(ϱ2)

    @test fidelity(U1prod, U2prod; process=true) ≈ fidelity(U1, U2; process=true)  
    @test fidelity(U1, U2prod; process=true) ≈ fidelity(U1, U2; process=true)  
    @test fidelity(U1prod, U2; process=true) ≈ fidelity(U1, U2; process=true)  
    
    @test fidelity(U1prod, ρ2prod; process=true) ≈ fidelity(U1, ρ2; process=true) 
    @test fidelity(U1, ρ2prod; process=true) ≈ fidelity(U1, ρ2; process=true) 
    @test fidelity(U1prod, ρ2; process=true) ≈ fidelity(U1, ρ2; process=true) 
    
    @test fidelity(U1prod, ϱ2prod; process=true) ≈ fidelity(U1, ϱ2; process=true) 
    @test fidelity(U1prod, ϱ2; process=true) ≈ fidelity(U1, ϱ2; process=true) 
    @test fidelity(U1, ϱ2prod; process=true) ≈ fidelity(U1, ϱ2; process=true) 
  end
end

@testset "quantum process fidelity: unnormalized states" begin
  N = 3
  sites = siteinds("Qubit",N)

  circuit1 = randomcircuit(N, 3)
  circuit2 = randomcircuit(N, 3)
  # MPO unitary 
  U1 = randomprocess(sites; χ = 3) 
  U2 = randomprocess(sites; χ = 4)

  # LPDO Choi matrix
  ϱ1 = randomprocess(sites; ξ = 3, χ = 2)
  ϱ2 = randomprocess(sites; ξ = 3, χ = 3)

  @disable_warn_order begin
    ϕ1 = PastaQ.unitary_mpo_to_choi_mps(U1)
    normalize!(ϕ1)
    ϕ1vec = PastaQ.array(ϕ1)
    ϱ1mat = PastaQ.array(ϱ1)
    ϱ1mat = ϱ1mat / tr(ϱ1mat)
    
    ϕ2 = PastaQ.unitary_mpo_to_choi_mps(U2)
    normalize!(ϕ2)
    ϕ2vec = PastaQ.array(ϕ2)
    ϱ2mat = PastaQ.array(ϱ2)
    ϱ2mat = ϱ2mat / tr(ϱ2mat)
    
    @test fidelity(U1, U2; process=true) ≈ abs2(ϕ1vec' * ϕ2vec)
    @test fidelity(U1, ϱ2; process=true) ≈ (ϕ1vec' * ϱ2mat * ϕ1vec)
    @test fidelity(ϱ1, ϱ2; process=true) ≈
          real(tr(sqrt(sqrt(ϱ1mat) * ϱ2mat * sqrt(ϱ1mat))))^2 atol = 1e-7

    ## ITensors
    U1prod = prod(U1) 
    U2prod = prod(U2) 
    ϱ1prod = prod(ϱ1)
    ϱ2prod = prod(ϱ2)

    @test fidelity(U1prod, U2prod; process=true) ≈ fidelity(U1, U2; process=true)  
    @test fidelity(U1, U2prod; process=true) ≈ fidelity(U1, U2; process=true)  
    @test fidelity(U1prod, U2; process=true) ≈ fidelity(U1, U2; process=true)  
    
    @test fidelity(U1prod, ϱ2prod; process=true) ≈ fidelity(U1, ϱ2; process=true) 
    @test fidelity(U1, ϱ2prod; process=true) ≈ fidelity(U1, ϱ2; process=true) 
  end
end

@testset "frobenius distance" begin
  N = 4
  Random.seed!(1111)
  ψ1 = randomstate(N; χ=2)
  Random.seed!(2222)
  ψ2 = randomstate(ψ1; χ=2)

  ρ_mpo = MPO(ψ1)
  σ_mpo = MPO(ψ2)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat)
  Kσ = tr(σ_mat)

  δ = ρ_mat / Kρ - σ_mat / Kσ

  T = sqrt(tr(conj(transpose(δ)) * δ))

  F = frobenius_distance(ρ_mpo, σ_mpo)
  @test T ≈ F
  @test F ≈ frobenius_distance(ψ1, σ_mpo)
  @test F ≈ frobenius_distance(ρ_mpo, ψ2)
  @test F ≈ frobenius_distance(ψ1, ψ2)

  Random.seed!(1111)
  ρ = randomstate(ψ1; mixed=true, χ=2, ξ=2)

  ρ_mpo = MPO(ρ)
  σ_mpo = MPO(ψ2)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat)
  Kσ = tr(σ_mat)

  δ = ρ_mat / Kρ - σ_mat / Kσ

  T = sqrt(tr(conj(transpose(δ)) * δ))

  F = frobenius_distance(ρ, σ_mpo)
  @test T ≈ F
  @test F ≈ frobenius_distance(ρ_mpo, ψ2)

  Random.seed!(1111)
  σ = randomstate(ψ1; mixed=true, χ=2, ξ=2)

  ρ_mpo = MPO(ψ1)
  σ_mpo = MPO(σ)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat)
  Kσ = tr(σ_mat)

  δ = ρ_mat / Kρ - σ_mat / Kσ

  T = sqrt(tr(conj(transpose(δ)) * δ))

  F = frobenius_distance(ρ_mpo, σ)
  @test T ≈ F
  @test F ≈ frobenius_distance(ψ1, σ_mpo)

  Random.seed!(1111)
  ρ = randomstate(N; mixed=true, χ=2, ξ=2)
  Random.seed!(1111)
  σ = randomstate(ρ; mixed=true, χ=2, ξ=2)

  ρ_mpo = MPO(ρ)
  σ_mpo = MPO(σ)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat)
  Kσ = tr(σ_mat)

  δ = ρ_mat / Kρ - σ_mat / Kσ

  T = sqrt(tr(conj(transpose(δ)) * δ))

  F = frobenius_distance(ρ, σ)
  @test T ≈ F
end

@testset "fidelity bound" begin
  N = 4
  Random.seed!(1111)
  ψ1 = randomstate(N; χ=2)
  Random.seed!(2222)
  ψ2 = randomstate(ψ1; χ=2)

  ρ_mpo = MPO(ψ1)
  σ_mpo = MPO(ψ2)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat)
  Kσ = tr(σ_mat)

  f = tr(conj(transpose(ρ_mat / Kρ)) * (σ_mat / Kσ))
  F̃ = fidelity_bound(ρ_mpo, σ_mpo)
  @test f ≈ F̃
  @test F̃ ≈ fidelity(ψ1, σ_mpo)
  @test F̃ ≈ fidelity(ρ_mpo, ψ2)
  @test F̃ ≈ fidelity(ψ1, ψ2)

  Random.seed!(1111)
  ρ = randomstate(ψ1; mixed=true, χ=2, ξ=2)

  ρ_mpo = MPO(ρ)
  σ_mpo = MPO(ψ2)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat)
  Kσ = tr(σ_mat)

  f = tr(conj(transpose(ρ_mat / Kρ)) * (σ_mat / Kσ))
  F̃ = fidelity_bound(ρ, σ_mpo)
  @test f ≈ F̃
  @test F̃ ≈ fidelity(ρ_mpo, ψ2)

  Random.seed!(1111)
  σ = randomstate(ψ1; mixed=true, χ=2, ξ=2)

  ρ_mpo = MPO(ψ1)
  σ_mpo = MPO(σ)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat)
  Kσ = tr(σ_mat)
  f = tr(conj(transpose(ρ_mat / Kρ)) * (σ_mat / Kσ))
  F̃ = fidelity_bound(ρ_mpo, σ)
  @test f ≈ F̃
  @test F̃ ≈ fidelity(ψ1, σ_mpo)

  Random.seed!(1111)
  ρ = randomstate(N; mixed=true, χ=2, ξ=2)
  Random.seed!(1111)
  σ = randomstate(ρ; mixed=true, χ=2, ξ=2)

  ρ_mpo = MPO(ρ)
  σ_mpo = MPO(σ)

  ρ_mat = PastaQ.array(ρ_mpo)
  σ_mat = PastaQ.array(σ_mpo)
  Kρ = tr(ρ_mat)
  Kσ = tr(σ_mat)

  f = tr(conj(transpose(ρ_mat / Kρ)) * (σ_mat / Kσ))
  F̃ = fidelity_bound(ρ, σ)
  @test f ≈ F̃
end
