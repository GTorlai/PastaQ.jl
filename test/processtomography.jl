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

#numgradslogZ(M::MPS; kwargs...) = numgradslogZ(LPDO(M); kwargs...)

function numgradsnll(L::LPDO, data::Matrix{Pair{String,Pair{String,Int}}}, accuracy=1e-8)
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

function numgradsTP(L::LPDO; accuracy=1e-8)
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
    loss_p = PastaQ.TP(L)
    M[1] -= eps
    loss_m = PastaQ.TP(L)
    grad_r[1][i] = (loss_p - loss_m) / (accuracy)

    epsilon[i] = im * accuracy
    eps = ITensor(epsilon, inds(M[1]))
    M[1] += eps
    loss_p = PastaQ.TP(L)
    M[1] -= eps
    loss_m = PastaQ.TP(L)
    grad_i[1][i] = (loss_p - loss_m) / (im * accuracy)

    epsilon[i] = 0.0
  end

  for j in 2:(N - 1)
    epsilon = zeros(ComplexF64, size(M[j]))
    for i in 1:length(epsilon)
      epsilon[i] = accuracy
      eps = ITensor(epsilon, inds(M[j]))
      M[j] += eps
      loss_p = PastaQ.TP(L)
      M[j] -= eps
      loss_m = PastaQ.TP(L)
      grad_r[j][i] = (loss_p - loss_m) / (accuracy)

      epsilon[i] = im * accuracy
      eps = ITensor(epsilon, inds(M[j]))
      M[j] += eps
      loss_p = PastaQ.TP(L)
      M[j] -= eps
      loss_m = PastaQ.TP(L)
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
    loss_p = PastaQ.TP(L)
    M[N] -= eps
    loss_m = PastaQ.TP(L)
    grad_r[N][i] = (loss_p - loss_m) / (accuracy)
    epsilon[i] = im * accuracy
    eps = ITensor(epsilon, inds(M[N]))
    M[N] += eps
    loss_p = PastaQ.TP(L)
    M[N] -= eps
    loss_m = PastaQ.TP(L)
    grad_i[N][i] = (loss_p - loss_m) / (im * accuracy)

    epsilon[i] = 0.0
  end

  return grad_r - grad_i
end

@testset "mpo-qpt: normalization" begin
  N = 5
  χ = 4
  U = randomprocess(N; χ=χ)
  Λ = LPDO(PastaQ.unitary_mpo_to_choi_mps(U))
  @test length(Λ) == N
  logZ = 2 * lognorm(Λ.X)
  sqrt_localZ = []
  PastaQ.normalize!(Λ; (sqrt_localnorms!)=sqrt_localZ, localnorm=2)
  @test logZ ≈ N * log(2) + 2.0 * sum(log.(sqrt_localZ))
  @test abs2(norm(Λ.X)) ≈ 2^N
end

@testset "mpo-qpt: grad logZ" begin
  N = 5
  χ = 4

  Random.seed!(1234)
  U = randomprocess(N; χ=χ)
  Λ = LPDO(PastaQ.unitary_mpo_to_choi_mps(U))
  num_grad = numgradslogZ(Λ)

  sqrt_localnorms = []
  PastaQ.normalize!(Λ; (sqrt_localnorms!)=sqrt_localnorms, localnorm=2)
  @test norm(Λ.X)^2 ≈ 2^N
  alg_grad, _ = PastaQ.gradlogZ(Λ; sqrt_localnorms=sqrt_localnorms)

  alg_gradient = permutedims(ITensors.array(alg_grad[1]), [1, 3, 2])
  @test alg_gradient ≈ num_grad[1] rtol = 1e-3
  for j in 2:(N - 1)
    alg_gradient = permutedims(ITensors.array(alg_grad[j]), [2, 1, 3, 4])
    @test alg_gradient ≈ num_grad[j] rtol = 1e-3
  end
  alg_gradient = permutedims(ITensors.array(alg_grad[N]), [2, 1, 3])
  @test alg_gradient ≈ num_grad[N] rtol = 1e-3
end

@testset "mpo-qpt: grad nll" begin
  N = 4
  χ = 2
  nsamples = 10
  Random.seed!(1234)
  data_in = randompreparations(N, nsamples)
  data_out = PastaQ.convertdatapoints(randompreparations(N, nsamples))
  data = data_in .=> data_out

  U = randomprocess(N; χ=χ)
  Λ = LPDO(PastaQ.unitary_mpo_to_choi_mps(U))
  num_grad = numgradsnll(Λ, data)
  sqrt_localnorms = []
  PastaQ.normalize!(Λ; (sqrt_localnorms!)=sqrt_localnorms, localnorm=2)

  alg_grad, _ = PastaQ.gradnll(Λ, data; sqrt_localnorms=sqrt_localnorms)
  for j in 1:N
    @test ITensors.array(alg_grad[j]) ≈ num_grad[j] rtol = 1e-3
  end
end

@testset "mpo-qpt: grad TP" begin
  N = 3
  χ = 3
  Random.seed!(1234)

  Random.seed!(1234)
  U = randomprocess(N; χ=χ)
  Λ = LPDO(PastaQ.unitary_mpo_to_choi_mps(U))

  num_grad = numgradsTP(Λ; accuracy=1e-8)
  Γ_test = PastaQ.TP(Λ)
  sqrt_localnorms = []
  PastaQ.normalize!(Λ; (sqrt_localnorms!)=sqrt_localnorms, localnorm=2)

  alg_grad_logZ, logZ = PastaQ.gradlogZ(Λ; sqrt_localnorms=sqrt_localnorms)

  alg_grad, Γ = PastaQ.gradTP(Λ, alg_grad_logZ, logZ; sqrt_localnorms=sqrt_localnorms)

  @test Γ ≈ Γ_test
  alg_gradient = permutedims(ITensors.array(alg_grad[1]), [1, 3, 2])
  @test alg_gradient ≈ num_grad[1] rtol = 1e-5
  for j in 2:(N - 1)
    alg_gradient = permutedims(ITensors.array(alg_grad[j]), [1, 3, 2, 4])
    @test alg_gradient ≈ num_grad[j] rtol = 1e-5
  end
  alg_gradient = permutedims(ITensors.array(alg_grad[N]), [1, 3, 2])
  @test alg_gradient ≈ num_grad[N] rtol = 1e-5
end

@testset "mpo-qpt: full gradients" begin
  N = 3
  χ = 4
  nsamples = 10
  trace_preserving_regularizer = 0.1
  Random.seed!(1234)
  data_in = randompreparations(N, nsamples)
  data_out = PastaQ.convertdatapoints(randompreparations(N, nsamples))
  data = data_in .=> data_out

  U = randomprocess(N; χ=χ)
  Λ = LPDO(PastaQ.unitary_mpo_to_choi_mps(U))
  TP_distance = PastaQ.TP(Λ)
  logZ = log(tr(Λ))
  NLL = PastaQ.nll(Λ, data)
  num_gradZ = numgradslogZ(Λ)
  num_gradNLL = numgradsnll(Λ, data)
  num_gradTP = numgradsTP(Λ; accuracy=1e-5)

  num_grads = num_gradZ + num_gradNLL + trace_preserving_regularizer * num_gradTP

  sqrt_localnorms = []
  PastaQ.normalize!(Λ; (sqrt_localnorms!)=sqrt_localnorms, localnorm=2)

  ex_loss = PastaQ.nll(Λ, data) + 2 * lognorm(Λ.X)
  alg_grads, loss = PastaQ.gradients(
    Λ,
    data;
    sqrt_localnorms=sqrt_localnorms,
    trace_preserving_regularizer=trace_preserving_regularizer,
  )
  @test ex_loss ≈ loss
  for j in 1:N
    @test ITensors.array(alg_grads[j]) ≈ num_grads[j] rtol = 1e-3
  end
end

""" CHOI TESTS """

@testset "lpdo-qpt: normalization" begin
  N = 10
  χ = 4
  ξ = 3
  Λ = randomprocess(N; mixed=true, ξ=ξ, χ=χ)

  @test length(Λ) == N
  logZ = logtr(Λ)
  localZ = []
  PastaQ.normalize!(Λ; (sqrt_localnorms!)=localZ, localnorm=2)
  @test logZ ≈ N * log(2) + 2.0 * sum(log.(localZ))
  @test tr(Λ) ≈ 2^N
end

@testset "lpdo-qst: grad logZ" begin
  N = 5
  χ = 4
  ξ = 3

  Λ = randomprocess(N; mixed=true, χ=χ, ξ=ξ)
  num_grad = numgradslogZ(Λ)
  sqrt_localnorms = []
  PastaQ.normalize!(Λ; (sqrt_localnorms!)=sqrt_localnorms, localnorm=2)
  @test tr(Λ) ≈ 2^N
  alg_grad, _ = PastaQ.gradlogZ(Λ; sqrt_localnorms=sqrt_localnorms)

  alg_gradient = permutedims(ITensors.array(alg_grad[1]), [1, 2, 4, 3])
  for j in 2:(N - 1)
    alg_gradient = permutedims(ITensors.array(alg_grad[j]), [2, 3, 1, 4, 5])
    @test alg_gradient ≈ num_grad[j] rtol = 1e-3
  end
  alg_gradient = permutedims(ITensors.array(alg_grad[N]), [2, 3, 1, 4])
  @test alg_gradient ≈ num_grad[N] rtol = 1e-3
end

@testset "lpdo-qst: grad nll" begin
  N = 5
  χ = 4
  ξ = 3

  nsamples = 10
  Random.seed!(1234)
  data_in = randompreparations(N, nsamples)
  data_out = PastaQ.convertdatapoints(randompreparations(N, nsamples))
  data = data_in .=> data_out

  Λ = randomprocess(N; mixed=true, χ=χ, ξ=ξ)
  num_grad = numgradsnll(Λ, data)
  sqrt_localnorms = []
  PastaQ.normalize!(Λ; (sqrt_localnorms!)=sqrt_localnorms, localnorm=2)
  alg_grad, loss = PastaQ.gradnll(Λ, data; sqrt_localnorms=sqrt_localnorms)
  @test loss ≈ PastaQ.nll(Λ, data)

  alg_gradient = permutedims(ITensors.array(alg_grad[1]), [3, 4, 1, 2])
  @test alg_gradient ≈ num_grad[1] rtol = 1e-3
  for j in 2:(N - 1)
    alg_gradient = permutedims(ITensors.array(alg_grad[j]), [4, 5, 2, 3, 1])
    @test alg_gradient ≈ num_grad[j] rtol = 1e-3
  end
  alg_gradient = permutedims(ITensors.array(alg_grad[N]), [3, 4, 1, 2])
  @test alg_gradient ≈ num_grad[N] rtol = 1e-3
end

@testset "lpdo-qpt: grad TP" begin
  N = 3
  χ = 4
  ξ = 3
  Λ = randomprocess(N; mixed=true, χ=χ, ξ=ξ)

  num_grad = numgradsTP(Λ; accuracy=1e-8)
  Γ_test = PastaQ.TP(Λ)
  sqrt_localnorms = []
  PastaQ.normalize!(Λ; (sqrt_localnorms!)=sqrt_localnorms, localnorm=2)
  Γ_test = PastaQ.TP(Λ)

  alg_grad_logZ, logZ = PastaQ.gradlogZ(Λ; sqrt_localnorms=sqrt_localnorms)

  alg_grad, Γ = PastaQ.gradTP(Λ, alg_grad_logZ, logZ; sqrt_localnorms=sqrt_localnorms)

  @test Γ ≈ Γ_test
  alg_gradient = permutedims(ITensors.array(alg_grad[1]), [3, 1, 4, 2])
  @test alg_gradient ≈ num_grad[1] rtol = 1e-3
  for j in 2:(N - 1)
    alg_gradient = permutedims(ITensors.array(alg_grad[j]), [3, 1, 4, 2, 5])
    @test alg_gradient ≈ num_grad[j] rtol = 1e-3
  end
  alg_gradient = permutedims(ITensors.array(alg_grad[N]), [3, 1, 4, 2])
  @test alg_gradient ≈ num_grad[N] rtol = 1e-3
end

@testset "lpdo-qpt: full gradients" begin
  N = 3
  χ = 3
  ξ = 2

  trace_preserving_regularizer = 0.1
  nsamples = 10
  Random.seed!(1234)
  data_in = randompreparations(N, nsamples)
  data_out = PastaQ.convertdatapoints(randompreparations(N, nsamples))
  data = data_in .=> data_out

  Λ = randomprocess(N; mixed=true, χ=χ, ξ=ξ)
  num_grad = numgradsnll(Λ, data)
  sqrt_localnorms = []
  PastaQ.normalize!(Λ; (sqrt_localnorms!)=sqrt_localnorms, localnorm=2)
  alg_grad, loss = PastaQ.gradnll(Λ, data; sqrt_localnorms=sqrt_localnorms)

  TP_distance = PastaQ.TP(Λ)
  logZ = log(tr(Λ))
  NLL = PastaQ.nll(Λ, data)
  num_gradZ = numgradslogZ(Λ)
  num_gradNLL = numgradsnll(Λ, data)
  num_gradTP = numgradsTP(Λ; accuracy=1e-5)

  num_grads = num_gradZ + num_gradNLL + trace_preserving_regularizer * num_gradTP

  sqrt_localnorms = []
  PastaQ.normalize!(Λ; (sqrt_localnorms!)=sqrt_localnorms, localnorm=2)

  ex_loss = PastaQ.nll(Λ, data) + 2 * lognorm(Λ.X)
  alg_grads, loss = PastaQ.gradients(
    Λ,
    data;
    sqrt_localnorms=sqrt_localnorms,
    trace_preserving_regularizer=trace_preserving_regularizer,
  )
  @test ex_loss ≈ loss
  for j in 1:N
    @test ITensors.array(alg_grads[j]) ≈ num_grads[j] rtol = 1e-3
  end
end
