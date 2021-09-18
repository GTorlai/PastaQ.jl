using ITensors
using PastaQ
using Test
using LinearAlgebra
using Random

@testset " counts and frequencies" begin
  N = 3
  d = 1<<N
  gates = randomcircuit(N,4)
  ψ = runcircuit(N,gates)

  samples = getsamples(ψ, 100; local_basis = ["X","Y","Z"], informationally_complete = true)

  C = PastaQ.measurement_counts(samples)
  @test length(keys(C)) == 3^N
  probs1 = PastaQ.empirical_probabilities(samples)
  @test length(keys(probs1)) == 3^N
  probs2 = PastaQ.empirical_probabilities(C)
  @test length(keys(probs2)) == 3^N
  @test probs1 == probs2

  for (basis,counts_in_basis) in C
    tot_counts = sum(values(counts_in_basis))
    for (proj,counts) in counts_in_basis
      freq = counts / tot_counts
      @test freq ≈ probs1[basis][proj]
    end
  end
end

@testset "POVM matrix" begin
  N = 3
  d = 1<<N
  gates = randomcircuit(N,4)
  ψ = runcircuit(N,gates)

  samples = getsamples(ψ,100; local_basis = ["X","Y","Z"], informationally_complete = true)

  ρ = MPO(ψ)
  ρmat = PastaQ.array(ρ)
  #ρ = projector(toarray(ψ))
  ρ_vec = vec(ρmat)

  probs = PastaQ.empirical_probabilities(samples)
  A = PastaQ.projector_matrix(probs; return_probs = false)
  real_probs = A * ρ_vec
  ρ̂_vec = pinv(A) * real_probs
  ρ̂ = reshape(ρ̂_vec,(d,d))
  @test ρmat ≈ ρ̂
end


@testset "make PSD" begin
  ρ = zeros(5,5)
  ρ[1,1] = 3/5; ρ[2,2] = 1/2; ρ[3,3] = 7/20;
  ρ[4,4] = 1/10; ρ[5,5] = -11/20;

  ρ̂ = PastaQ.make_PSD(ρ)
  λ = reverse(first(eigen(ρ̂)))
  @test λ[1] ≈ 9/20
  @test λ[2] ≈ 7/20
  @test λ[3] ≈ 1/5
  @test λ[4] ≈ 0.0
  @test λ[5] ≈ 0.0
end

@testset "PSD constraint in QST" begin
  N = 2
  d = 1<<N
  gates = randomcircuit(N,4)
  ψ = runcircuit(N,gates)

  samples = getsamples(ψ,100; local_basis = ["X","Y","Z"], informationally_complete=true)

  ϱ = PastaQ.array(MPO(ψ))

  ρ = tomography(samples; method = "LI")
  λ = first(eigen(ρ))
  @test all(λ .≥ -1e-4)

  ρ = tomography(samples; method = "LS")
  λ = first(eigen(ρ))
  @test all(real(λ) .≥ -1e-4)

  ρ = tomography(samples; method = "MLE")
  λ = first(eigen(ρ))
  @test all(real(λ) .≥-1e-4)
end

@testset "arbitrary trace in QST" begin
  N = 2
  d = 1<<N
  gates = randomcircuit(N,4)
  ψ = runcircuit(N,gates)

  samples = getsamples(ψ,100; local_basis = ["X","Y","Z"], informationally_complete=true)

  ϱ = PastaQ.array(MPO(ψ))

  ρ = tomography(samples; method = "LI", trρ = 2.0)
  @test tr(ρ) ≈ 2.0
  ρ = tomography(samples; method = "LS", trρ = 2.0)
  @test tr(ρ) ≈ 2.0 atol = 1e-5
  ρ = tomography(samples; method = "MLE", trρ = 2.0)
  @test tr(ρ) ≈ 2.0 atol = 1e-5

end

#@testset "state tomography circuits" begin
#  N = 3
#  d = 1<<N
#  gates = randomcircuit(N,4)
#  ngates = length(gates)
#
#  bases, qst_circs = tomography_circuits(N, gates)
#  @test length(qst_circs) == 3^N
#
#  @show typeof(bases)
#  for b in 1:size(bases,1)
#    basis = bases[b,:]
#    mgates = 0
#    for j in 1:N
#      if basis[j] ≠ "Z"
#        mgates += 1
#      end
#    end
#    @test length(qst_circs[b]) == ngates + mgates
#  end
#
#  _, qst_circs = tomography_circuits(N, gates; nbases = 10)
#  @test length(qst_circs) == 10
#
#end
