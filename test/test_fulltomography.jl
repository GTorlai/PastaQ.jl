using ITensors
using PastaQ
using Test
using LinearAlgebra
using Random

using SCS
using Convex

@testset " counts and frequencies" begin
  N = 3
  d = 1 << N
  gates = randomcircuit(N; depth=4)
  ψ = runcircuit(N, gates)

  bases = fullbases(N)
  samples = getsamples(ψ, bases, 100)

  C = PastaQ.measurement_counts(samples)
  @test length(keys(C)) == 3^N
  probs1 = PastaQ.empirical_probabilities(samples)
  @test length(keys(probs1)) == 3^N
  probs2 = PastaQ.empirical_probabilities(C)
  @test length(keys(probs2)) == 3^N
  @test probs1 == probs2

  for (basis, counts_in_basis) in C
    tot_counts = sum(values(counts_in_basis))
    for (proj, counts) in counts_in_basis
      freq = counts / tot_counts
      @test freq ≈ probs1[basis][proj]
    end
  end
end

@testset "POVM matrix" begin
  N = 3
  d = 1 << N
  gates = randomcircuit(N, ; depth=4)
  ψ = runcircuit(N, gates)

  bases = fullbases(N)
  samples = getsamples(ψ, bases, 100)

  ρ = outer(ψ', ψ)
  ρmat = PastaQ.array(ρ)
  #ρ = projector(toarray(ψ))
  ρ_vec = vec(ρmat)

  probs = PastaQ.empirical_probabilities(samples)
  A = PastaQ.design_matrix(probs; return_probs=false)
  real_probs = A * ρ_vec
  ρ̂_vec = pinv(A) * real_probs
  ρ̂ = reshape(ρ̂_vec, (d, d))
  @test ρmat ≈ ρ̂
end

@testset "make PSD" begin
  ρ = zeros(5, 5)
  ρ[1, 1] = 3 / 5
  ρ[2, 2] = 1 / 2
  ρ[3, 3] = 7 / 20
  ρ[4, 4] = 1 / 10
  ρ[5, 5] = -11 / 20

  ρ̂ = PastaQ.make_PSD(ρ)
  λ = reverse(first(eigen(ρ̂)))
  @test λ[1] ≈ 9 / 20
  @test λ[2] ≈ 7 / 20
  @test λ[3] ≈ 1 / 5
  @test λ[4] ≈ 0.0
  @test λ[5] ≈ 0.0
end

@testset "PSD constraint in QST" begin
  N = 2
  d = 1 << N
  gates = randomcircuit(N; depth=4)
  ψ = runcircuit(N, gates)

  bases = fullbases(N)
  samples = getsamples(ψ, bases, 100)

  ϱ = PastaQ.array(outer(ψ', ψ))

  ρ = PastaQ.array(tomography(samples; method="LI"))
  λ = first(eigen(ρ))
  @test all(λ .≥ -1e-4)

  ρ = PastaQ.array(tomography(samples; method="LS"))
  λ = first(eigen(ρ))
  @test all(real(λ) .≥ -1e-3)

  ρ = PastaQ.array(tomography(samples; method="ML")) # Ensured method keyword consistency
  λ = first(eigen(ρ))
  @test all(real(λ) .≥ -1e-2)
end

@testset "arbitrary trace in QST" begin
  N = 2
  d = 1 << N
  gates = randomcircuit(N; depth=4)
  ψ = runcircuit(N, gates)

  bases = fullbases(N)
  samples = getsamples(ψ, bases, 100)

  ϱ = PastaQ.array(outer(ψ', ψ))

  ρ = PastaQ.array(tomography(samples; method="LI", trρ=2.0))
  @test tr(ρ) ≈ 2.0
  ρ = PastaQ.array(tomography(samples; method="LS", trρ=2.0))
  @test tr(ρ) ≈ 2.0 atol = 1e-4
  ρ = PastaQ.array(tomography(samples; method="ML", trρ=2.0)) # Ensured method keyword consistency
  @test tr(ρ) ≈ 2.0 atol = 1e-4
end

@testset "Choi POVM matrix" begin
  N = 2
  d = 2^(2 * N)
  nshots = 3
  gates = randomcircuit(N; depth=2)

  Λ = runcircuit(gates; process=true, noise=("DEP", (p=0.001,)))
  preps = fullpreparations(N)
  bases = fullbases(N)
  data = getsamples(Λ, preps, bases, nshots)

  Λmat = PastaQ.array(Λ)
  Λvec = vec(Λmat)

  probs = PastaQ.empirical_probabilities(data)
  A = PastaQ.design_matrix(probs; return_probs=false, process=true)
  real_probs = A * Λvec
  Λ̂vec = pinv(A) * real_probs
  Λ̂ = reshape(Λ̂vec, (d, d))
  @test Λmat ≈ Λ̂
end

@testset "PSD constraint in QPT" begin
  N = 2
  d = 1 << N
  nshots = 3
  gates = randomcircuit(N; depth=4)
  Λ = runcircuit(gates; process=true, noise=("DEP", (p=0.001,)))
  preps = fullpreparations(N)
  bases = fullbases(N)
  data = getsamples(Λ, preps, bases, nshots)

  ρ = PastaQ.array(tomography(data; method="LI"))
  λ = first(eigen(ρ))
  @test all(λ .≥ -1e-4)

  ρ = PastaQ.array(tomography(data; method="LS"))
  λ = first(eigen(ρ))
  @test all(real(λ) .≥ -1e-4)
end

@testset "Trace preserving condition in QPT" begin
  Random.seed!(1234)
  N = 2
  d = 1 << N
  nshots = 3
  gates = randomcircuit(N; depth=4)
  Λ = runcircuit(gates; process=true, noise=("DEP", (p=0.001,)))
  preps = fullpreparations(N)
  bases = fullbases(N)
  data = getsamples(Λ, preps, bases, nshots)

  ρ = tomography(data; method="LS")
  for j in 1:N
    s = firstind(ρ; tags="Output,n=$(j)", plev=0)
    ρ = ρ * δ(s, s')
  end

  ρmat = PastaQ.array(ρ)
  @test ρmat ≈ Matrix{Float64}(I, d, d) atol = 1e-5
end
