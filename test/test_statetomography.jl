using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

""" HELPER FUNCTIONS """
function numgradslogZ(L::LPDO; accuracy=1e-8)
  M = L.X
  N = length(M)
  grad_r = []
  grad_i = []
  for j in 1:N
    push!(grad_r, zeros(ComplexF64, size(M[j])))
    push!(grad_i, zeros(ComplexF64, size(M[j])))
  end

  epsilon = zeros(ComplexF64, size(M[1]))
  # Site 1
  for i in 1:length(epsilon)
    epsilon[i] = accuracy
    eps = ITensor(epsilon, inds(M[1]))
    M[1] += eps
    loss_p = 2.0 * lognorm(M)
    M[1] -= eps
    loss_m = 2.0 * lognorm(M)
    grad_r[1][i] = (loss_p - loss_m) / (accuracy)

    epsilon[i] = im * accuracy
    eps = ITensor(epsilon, inds(M[1]))
    M[1] += eps
    loss_p = 2.0 * lognorm(M)
    M[1] -= eps
    loss_m = 2.0 * lognorm(M)
    grad_i[1][i] = (loss_p - loss_m) / (im * accuracy)

    epsilon[i] = 0.0
  end

  for j in 2:(N - 1)
    epsilon = zeros(ComplexF64, size(M[j]))
    for i in 1:length(epsilon)
      epsilon[i] = accuracy
      eps = ITensor(epsilon, inds(M[j]))
      M[j] += eps
      loss_p = 2.0 * lognorm(M)
      M[j] -= eps
      loss_m = 2.0 * lognorm(M)
      grad_r[j][i] = (loss_p - loss_m) / (accuracy)

      epsilon[i] = im * accuracy
      eps = ITensor(epsilon, inds(M[j]))
      M[j] += eps
      loss_p = 2.0 * lognorm(M)
      M[j] -= eps
      loss_m = 2.0 * lognorm(M)
      grad_i[j][i] = (loss_p - loss_m) / (im * accuracy)

      epsilon[i] = 0.0
    end
  end
  # Site N
  epsilon = zeros(ComplexF64, size(M[N]))
  for i in 1:length(epsilon)
    epsilon[i] = accuracy
    eps = ITensor(epsilon, inds(M[N]))
    M[N] += eps
    loss_p = 2.0 * lognorm(M)
    M[N] -= eps
    loss_m = 2.0 * lognorm(M)
    grad_r[N][i] = (loss_p - loss_m) / (accuracy)

    epsilon[i] = im * accuracy
    eps = ITensor(epsilon, inds(M[N]))
    M[N] += eps
    loss_p = 2.0 * lognorm(M)
    M[N] -= eps
    loss_m = 2.0 * lognorm(M)
    grad_i[N][i] = (loss_p - loss_m) / (im * accuracy)

    epsilon[i] = 0.0
  end

  return grad_r - grad_i
end

numgradslogZ(M::MPS; kwargs...) = numgradslogZ(LPDO(M); kwargs...)

function numgradsnll(L::LPDO, data::Array; accuracy=1e-8)
  M = L.X
  N = length(M)
  grad_r = []
  grad_i = []
  for j in 1:N
    push!(grad_r, zeros(ComplexF64, size(M[j])))
    push!(grad_i, zeros(ComplexF64, size(M[j])))
  end

  epsilon = zeros(ComplexF64, size(M[1]))
  # Site 1
  for i in 1:length(epsilon)
    epsilon[i] = accuracy
    eps = ITensor(epsilon, inds(M[1]))
    M[1] += eps
    loss_p = PastaQ.nll(L, data)
    M[1] -= eps
    loss_m = PastaQ.nll(L, data)
    grad_r[1][i] = (loss_p - loss_m) / (accuracy)

    epsilon[i] = im * accuracy
    eps = ITensor(epsilon, inds(M[1]))
    M[1] += eps
    loss_p = PastaQ.nll(L, data)
    M[1] -= eps
    loss_m = PastaQ.nll(L, data)
    grad_i[1][i] = (loss_p - loss_m) / (im * accuracy)

    epsilon[i] = 0.0
  end

  for j in 2:(N - 1)
    epsilon = zeros(ComplexF64, size(M[j]))
    for i in 1:length(epsilon)
      epsilon[i] = accuracy
      eps = ITensor(epsilon, inds(M[j]))
      M[j] += eps
      loss_p = PastaQ.nll(L, data)
      M[j] -= eps
      loss_m = PastaQ.nll(L, data)
      grad_r[j][i] = (loss_p - loss_m) / (accuracy)

      epsilon[i] = im * accuracy
      eps = ITensor(epsilon, inds(M[j]))
      M[j] += eps
      loss_p = PastaQ.nll(L, data)
      M[j] -= eps
      loss_m = PastaQ.nll(L, data)
      grad_i[j][i] = (loss_p - loss_m) / (im * accuracy)

      epsilon[i] = 0.0
    end
  end

  # Site N
  epsilon = zeros(ComplexF64, size(M[N]))
  for i in 1:length(epsilon)
    epsilon[i] = accuracy
    eps = ITensor(epsilon, inds(M[N]))
    M[N] += eps
    loss_p = PastaQ.nll(L, data)
    M[N] -= eps
    loss_m = PastaQ.nll(L, data)
    grad_r[N][i] = (loss_p - loss_m) / (accuracy)

    epsilon[i] = im * accuracy
    eps = ITensor(epsilon, inds(M[N]))
    M[N] += eps
    loss_p = PastaQ.nll(L, data)
    M[N] -= eps
    loss_m = PastaQ.nll(L, data)
    grad_i[N][i] = (loss_p - loss_m) / (im * accuracy)

    epsilon[i] = 0.0
  end

  return grad_r - grad_i
end

numgradsnll(M::MPS, args...; kwargs...) = numgradsnll(LPDO(M), args...; kwargs...)

""" MPS TESTS """

@testset "mps-qst: normalization" begin
  N = 10
  χ = 4
  ψ = randomstate(N; χ=χ)
  @test length(ψ) == N
  logZ = lognorm(ψ)
  localZ = []
  PastaQ.normalize!(ψ; (localnorms!)=localZ)
  @test logZ ≈ sum(log.(localZ))
  @test norm(ψ) ≈ 1
end

@testset "mps-qst: grad logZ" begin
  N = 5
  χ = 4

  # 1. Unnormalized
  ψ = randomstate(N; χ=χ)
  alg_grad, _ = PastaQ.gradlogZ(ψ)
  num_grad = numgradslogZ(ψ)
  for j in 1:N
    @test ITensors.array(alg_grad[j]) ≈ num_grad[j] rtol = 1e-3
  end

  # 2. Globally normalized
  ψ = randomstate(N; χ=χ)
  PastaQ.normalize!(ψ)
  @test norm(ψ)^2 ≈ 1
  alg_grad, _ = PastaQ.gradlogZ(ψ)
  num_grad = numgradslogZ(ψ)
  for j in 1:N
    @test ITensors.array(alg_grad[j]) ≈ num_grad[j] rtol = 1e-3
  end

  # 3. Locally normalized
  ψ = randomstate(N; χ=χ)
  num_grad = numgradslogZ(ψ)

  localnorms = []
  PastaQ.normalize!(ψ; (localnorms!)=localnorms)
  @test norm(ψ) ≈ 1
  alg_grad, _ = PastaQ.gradlogZ(ψ; localnorms=localnorms)
  for j in 1:N
    @test ITensors.array(alg_grad[j]) ≈ num_grad[j] rtol = 1e-3
  end
end

@testset "mps-qst: grad nll" begin
  N = 5
  χ = 4
  nsamples = 10
  Random.seed!(1234)
  data = PastaQ.convertdatapoints(randompreparations(N, nsamples))

  # 1. Unnormalized
  ψ = randomstate(N; χ=χ)
  num_grad = numgradsnll(ψ, data)
  alg_grad, loss = PastaQ.gradnll(ψ, data)
  for j in 1:N
    @test ITensors.array(alg_grad[j]) ≈ num_grad[j] rtol = 1e-3
  end

  # 2. Globally normalized
  ψ = randomstate(N; χ=χ)
  PastaQ.normalize!(ψ)
  num_grad = numgradsnll(ψ, data)
  alg_grad, loss = PastaQ.gradnll(ψ, data)
  for j in 1:N
    @test ITensors.array(alg_grad[j]) ≈ num_grad[j] rtol = 1e-3
  end

  # 3. Locally normalized
  ψ = randomstate(N; χ=χ)
  num_grad = numgradsnll(ψ, data)
  localnorms = []
  PastaQ.normalize!(ψ; (localnorms!)=localnorms)
  @test norm(ψ) ≈ 1
  alg_grad_localnorm, loss = PastaQ.gradnll(ψ, data; localnorms=localnorms)
  for j in 1:N
    @test ITensors.array(alg_grad_localnorm[j]) ≈ num_grad[j] rtol = 1e-3
  end
end

@testset "mps-qst: full gradients" begin
  N = 5
  χ = 4
  nsamples = 10
  data = PastaQ.convertdatapoints(randompreparations(N, nsamples))

  # 1. Unnormalized
  ψ = randomstate(N; χ=χ)
  logZ = 2.0 * log(norm(ψ))
  NLL = PastaQ.nll(ψ, data)
  ex_loss = logZ + NLL
  num_gradZ = numgradslogZ(ψ)
  num_gradNLL = numgradsnll(ψ, data)
  num_grads = num_gradZ + num_gradNLL

  alg_grads, loss = PastaQ.gradients(ψ, data)
  @test ex_loss ≈ loss
  for j in 1:N
    @test ITensors.array(alg_grads[j]) ≈ num_grads[j] rtol = 1e-3
  end

  # 2. Globally normalized
  ψ = randomstate(N; χ=χ)
  PastaQ.normalize!(ψ)
  num_gradZ = numgradslogZ(ψ)
  num_gradNLL = numgradsnll(ψ, data)
  num_grads = num_gradZ + num_gradNLL
  NLL = PastaQ.nll(ψ, data)
  ex_loss = NLL
  @test norm(ψ)^2 ≈ 1

  alg_grads, loss = PastaQ.gradients(ψ, data)
  @test ex_loss ≈ loss
  for j in 1:N
    @test ITensors.array(alg_grads[j]) ≈ num_grads[j] rtol = 1e-3
  end

  # 3. Locally normalized
  ψ = randomstate(N; χ=χ)
  num_gradZ = numgradslogZ(ψ)
  num_gradNLL = numgradsnll(ψ, data)
  num_grads = num_gradZ + num_gradNLL

  localnorms = []
  PastaQ.normalize!(ψ; (localnorms!)=localnorms)
  NLL = PastaQ.nll(ψ, data)
  ex_loss = NLL
  @test norm(ψ)^2 ≈ 1

  alg_grads, loss = PastaQ.gradients(ψ, data; localnorms=localnorms)
  @test ex_loss ≈ loss
  for j in 1:N
    @test ITensors.array(alg_grads[j]) ≈ num_grads[j] rtol = 1e-3
  end
end

""" LPDO TESTS """

@testset "lpdo-qst: normalization" begin
  N = 10
  χ = 4
  ξ = 3
  ρ = randomstate(N; mixed=true, ξ=ξ, χ=χ)
  @test length(ρ) == N
  logZ = logtr(ρ)
  localZ = []
  PastaQ.normalize!(ρ; (sqrt_localnorms!)=localZ)
  @test logZ ≈ 2.0 * sum(log.(localZ))
  @test tr(ρ) ≈ 1
end

@testset "lpdo-qst: grad logZ" begin
  N = 5
  χ = 4
  ξ = 3

  # 1. Unnormalized
  ρ = randomstate(N; mixed=true, χ=χ, ξ=ξ)
  alg_grad, logZ = PastaQ.gradlogZ(ρ)
  @test logZ ≈ logtr(ρ)
  num_grad = numgradslogZ(ρ)
  alg_gradient = permutedims(ITensors.array(alg_grad[1]), [1, 3, 2])
  @test alg_gradient ≈ num_grad[1] rtol = 1e-3
  for j in 2:(N - 1)
    alg_gradient = permutedims(ITensors.array(alg_grad[j]), [2, 1, 3, 4])
    @test alg_gradient ≈ num_grad[j] rtol = 1e-3
  end
  alg_gradient = permutedims(ITensors.array(alg_grad[N]), [2, 1, 3])
  @test alg_gradient ≈ num_grad[N] rtol = 1e-3

  # 2. Globally normalized
  ρ = randomstate(N; mixed=true, χ=χ, ξ=ξ)
  PastaQ.normalize!(ρ)
  @test tr(ρ) ≈ 1
  alg_grad, _ = PastaQ.gradlogZ(ρ)
  num_grad = numgradslogZ(ρ)

  alg_gradient = permutedims(ITensors.array(alg_grad[1]), [1, 3, 2])
  @test alg_gradient ≈ num_grad[1] rtol = 1e-3
  for j in 2:(N - 1)
    alg_gradient = permutedims(ITensors.array(alg_grad[j]), [2, 1, 3, 4])
    @test alg_gradient ≈ num_grad[j] rtol = 1e-3
  end
  alg_gradient = permutedims(ITensors.array(alg_grad[N]), [2, 1, 3])
  @test alg_gradient ≈ num_grad[N] rtol = 1e-3

  # 3. Locally normalized
  ρ = randomstate(N; mixed=true, χ=χ, ξ=ξ)
  num_grad = numgradslogZ(ρ)

  sqrt_localnorms = []
  PastaQ.normalize!(ρ; (sqrt_localnorms!)=sqrt_localnorms)
  @test tr(ρ) ≈ 1
  alg_grad, _ = PastaQ.gradlogZ(ρ; sqrt_localnorms=sqrt_localnorms)

  alg_gradient = permutedims(ITensors.array(alg_grad[1]), [1, 3, 2])
  @test alg_gradient ≈ num_grad[1] rtol = 1e-3
  for j in 2:(N - 1)
    alg_gradient = permutedims(ITensors.array(alg_grad[j]), [2, 1, 3, 4])
    @test alg_gradient ≈ num_grad[j] rtol = 1e-3
  end
  alg_gradient = permutedims(ITensors.array(alg_grad[N]), [2, 1, 3])
  @test alg_gradient ≈ num_grad[N] rtol = 1e-3
end

@testset "lpdo-qst: grad nll" begin
  N = 5
  χ = 4
  ξ = 3

  nsamples = 10
  Random.seed!(1234)
  data = PastaQ.convertdatapoints(randompreparations(N, nsamples))

  # 1. Unnormalized
  ρ = randomstate(N; mixed=true, χ=χ, ξ=ξ)
  num_grad = numgradsnll(ρ, data)
  alg_grad, loss = PastaQ.gradnll(ρ, data)
  @test loss ≈ PastaQ.nll(ρ, data)

  alg_gradient = permutedims(ITensors.array(alg_grad[1]), [3, 1, 2])
  @test alg_gradient ≈ num_grad[1] rtol = 1e-3
  for j in 2:(N - 1)
    alg_gradient = permutedims(ITensors.array(alg_grad[j]), [4, 2, 3, 1])
    @test alg_gradient ≈ num_grad[j] rtol = 1e-3
  end
  alg_gradient = permutedims(ITensors.array(alg_grad[N]), [3, 1, 2])
  @test alg_gradient ≈ num_grad[N] rtol = 1e-3

  ## 2. Globally normalized
  ρ = randomstate(N; mixed=true, χ=χ, ξ=ξ)
  PastaQ.normalize!(ρ)
  num_grad = numgradsnll(ρ, data)
  alg_grad, loss = PastaQ.gradnll(ρ, data)

  alg_gradient = permutedims(ITensors.array(alg_grad[1]), [3, 1, 2])
  @test alg_gradient ≈ num_grad[1] rtol = 1e-3
  for j in 2:(N - 1)
    alg_gradient = permutedims(ITensors.array(alg_grad[j]), [4, 2, 3, 1])
    @test alg_gradient ≈ num_grad[j] rtol = 1e-3
  end
  alg_gradient = permutedims(ITensors.array(alg_grad[N]), [3, 1, 2])
  @test alg_gradient ≈ num_grad[N] rtol = 1e-3

  # 3. Locally normalized
  ρ = randomstate(N; mixed=true, χ=χ, ξ=ξ)
  num_grad = numgradsnll(ρ, data)
  sqrt_localnorms = []
  PastaQ.normalize!(ρ; (sqrt_localnorms!)=sqrt_localnorms)
  @test tr(ρ) ≈ 1
  alg_grad, loss = PastaQ.gradnll(ρ, data; sqrt_localnorms=sqrt_localnorms)
  alg_gradient = permutedims(ITensors.array(alg_grad[1]), [3, 1, 2])
  @test alg_gradient ≈ num_grad[1] rtol = 1e-3
  for j in 2:(N - 1)
    alg_gradient = permutedims(ITensors.array(alg_grad[j]), [4, 2, 3, 1])
    @test alg_gradient ≈ num_grad[j] rtol = 1e-3
  end
  alg_gradient = permutedims(ITensors.array(alg_grad[N]), [3, 1, 2])
  @test alg_gradient ≈ num_grad[N] rtol = 1e-3
end
