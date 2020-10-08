using PastaQ
using Random
using Test

@testset "tr Choi" begin
  Random.seed!(1234)
  N = 4
  depth = 4
  nshots = 100
  gates = randomcircuit(N, depth)
  data, Λ = getsamples(N, gates, nshots;
                       process = true,
                       noise = ("amplitude_damping", (γ = 0.01,)))
  @test tr(Λ.M) ≈ 16
  @test tr(Λ) ≈ 16
end

@testset "normalize! Choi" begin
  Random.seed!(1234)
  N = 4
  depth = 4
  nshots = 10_000
  gates = randomcircuit(N, depth)
  data, Φ = getsamples(N, gates, nshots;
                       process = true,
                       noise = ("amplitude_damping", (γ = 0.01,)))
  N = length(Φ)
  χ = 8
  ξ = 2
  # Initialize the Choi LPDO
  Λ0 = randomprocess(Φ; mixed = true, χ = χ, ξ = ξ)
  # Initialize stochastic gradient descent optimizer
  opt = SGD(η = 0.1)
  # Run process tomography
  println("Run process tomography to learn noisy process Λ")
  Λ = tomography(data, Λ0;
                 optimizer = opt,
                 batchsize = 500,
                 epochs = 5,
                 target = Φ)
  @test fidelity_bound(Φ.M, Λ.M) < 1
  @test fidelity_bound(Φ, Λ) < 1
end
