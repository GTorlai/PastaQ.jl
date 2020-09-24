using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

""" HELPER FUNCTIONS """
function numgradslogZ(L::LPDO;accuracy=1e-8)
  M = L.X
  N = length(M)
  grad_r = []
  grad_i = []
  for j in 1:N
    push!(grad_r,zeros(ComplexF64,size(M[j])))
    push!(grad_i,zeros(ComplexF64,size(M[j])))
  end
  
  epsilon = zeros(ComplexF64,size(M[1]));
  # Site 1
  for i in 1:length(epsilon)
    epsilon[i] = accuracy
    eps = ITensor(epsilon,inds(M[1]))
    M[1] += eps
    loss_p = 2.0*lognorm(M)
    M[1] -= eps
    loss_m = 2.0*lognorm(M)
    grad_r[1][i] = (loss_p-loss_m)/(accuracy)
    
    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(M[1]))
    M[1] += eps
    loss_p = 2.0*lognorm(M)
    M[1] -= eps
    loss_m = 2.0*lognorm(M)
    grad_i[1][i] = (loss_p-loss_m)/(im*accuracy)
    
    epsilon[i] = 0.0
  end

  for j in 2:N-1
    epsilon = zeros(ComplexF64,size(M[j]));
    for i in 1:length(epsilon)
      epsilon[i] = accuracy
      eps = ITensor(epsilon,inds(M[j]))
      M[j] += eps
      loss_p = 2.0*lognorm(M)
      M[j] -= eps
      loss_m = 2.0*lognorm(M)
      grad_r[j][i] = (loss_p-loss_m)/(accuracy)
      
      epsilon[i] = im*accuracy
      eps = ITensor(epsilon,inds(M[j]))
      M[j] += eps
      loss_p = 2.0*lognorm(M)
      M[j] -= eps
      loss_m = 2.0*lognorm(M)
      grad_i[j][i] = (loss_p-loss_m)/(im*accuracy)

      epsilon[i] = 0.0
    end
  end
  # Site N
  epsilon = zeros(ComplexF64,size(M[N]));
  for i in 1:length(epsilon)
    epsilon[i] = accuracy
    eps = ITensor(epsilon,inds(M[N]))
    M[N] += eps
    loss_p = 2.0*lognorm(M)
    M[N] -= eps
    loss_m = 2.0*lognorm(M)
    grad_r[N][i] = (loss_p-loss_m)/(accuracy)

    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(M[N]))
    M[N] += eps
    loss_p = 2.0*lognorm(M)
    M[N] -= eps
    loss_m = 2.0*lognorm(M)
    grad_i[N][i] = (loss_p-loss_m)/(im*accuracy)
    
    epsilon[i] = 0.0
  end

  return grad_r-grad_i
end

numgradslogZ(M::MPS; kwargs...) = numgradslogZ(LPDO(M); kwargs...)

function numgradsnll(L::LPDO,
                     data::Array;
                     accuracy=1e-8,
                     choi::Bool=false)
  M = L.X
  N = length(M)
  grad_r = []
  grad_i = []
  for j in 1:N
    push!(grad_r,zeros(ComplexF64,size(M[j])))
    push!(grad_i,zeros(ComplexF64,size(M[j])))
  end
  
  epsilon = zeros(ComplexF64,size(M[1]));
  # Site 1
  for i in 1:length(epsilon)
    epsilon[i] = accuracy
    eps = ITensor(epsilon,inds(M[1]))
    M[1] += eps
    loss_p = nll(L,data,choi=choi) 
    M[1] -= eps
    loss_m = nll(L,data,choi=choi) 
    grad_r[1][i] = (loss_p-loss_m)/(accuracy)
    
    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(M[1]))
    M[1] += eps
    loss_p = nll(L,data,choi=choi) 
    M[1] -= eps
    loss_m = nll(L,data,choi=choi) 
    grad_i[1][i] = (loss_p-loss_m)/(im*accuracy)
    
    epsilon[i] = 0.0
  end

  for j in 2:N-1
    epsilon = zeros(ComplexF64,size(M[j]));
    for i in 1:length(epsilon)
      epsilon[i] = accuracy
      eps = ITensor(epsilon,inds(M[j]))
      M[j] += eps
      loss_p = nll(L,data,choi=choi) 
      M[j] -= eps
      loss_m = nll(L,data,choi=choi) 
      grad_r[j][i] = (loss_p-loss_m)/(accuracy)
      
      epsilon[i] = im*accuracy
      eps = ITensor(epsilon,inds(M[j]))
      M[j] += eps
      loss_p = nll(L,data,choi=choi) 
      M[j] -= eps
      loss_m = nll(L,data,choi=choi) 
      grad_i[j][i] = (loss_p-loss_m)/(im*accuracy)

      epsilon[i] = 0.0
    end
 end

  # Site N
  epsilon = zeros(ComplexF64,size(M[N]));
  for i in 1:length(epsilon)
    epsilon[i] = accuracy
    eps = ITensor(epsilon,inds(M[N]))
    M[N] += eps
    loss_p = nll(L,data,choi=choi) 
    M[N] -= eps
    loss_m = nll(L,data,choi=choi) 
    grad_r[N][i] = (loss_p-loss_m)/(accuracy)

    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(M[N]))
    M[N] += eps
    loss_p = nll(L,data,choi=choi) 
    M[N] -= eps
    loss_m = nll(L,data,choi=choi) 
    grad_i[N][i] = (loss_p-loss_m)/(im*accuracy)
    
    epsilon[i] = 0.0
  end

  return grad_r-grad_i
end

numgradsnll(M::MPS, args...; kwargs...) =
  numgradsnll(LPDO(M), args...; kwargs...)

""" MPS STATE TOMOGRAPHY TESTS """

@testset "mps-qst: lognormalization" begin
  N = 10
  χ = 4
  ψ = initializetomography(N;χ=χ)
  @test length(ψ) == N
  logZ1 = 2.0*lognorm(ψ)
  logZ2,_ = lognormalize!(ψ)
  @test logZ1 ≈ logZ2
  @test norm(ψ) ≈ 1
end

@testset "mps-qst: grad logZ" begin
  N = 5
  χ = 4
  
  # 1. Unnormalized
  ψ = initializetomography(N;χ=χ)
  alg_grad,_ = gradlogZ(ψ)
  num_grad = numgradslogZ(ψ)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
  
  # 2. Globally normalized
  ψ = initializetomography(N;χ=χ)
  lognormalize!(ψ)
  @test norm(ψ)^2 ≈ 1
  alg_grad,_ = gradlogZ(ψ)
  num_grad = numgradslogZ(ψ)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end

  # 3. Locally normalized
  ψ = initializetomography(N;χ=χ)
  num_grad = numgradslogZ(ψ)

  logZ,localnorms = lognormalize!(ψ)
  @test norm(ψ)^2 ≈ 1
  alg_grad,_ = gradlogZ(ψ,localnorm=localnorms)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
end

@testset "mps-qst: grad nll" begin
  N = 5
  χ = 4
  nsamples = 100
  Random.seed!(1234)
  rawdata = rand(0:1,nsamples,N)
  bases = randombases(N,nsamples)
  data = Matrix{String}(undef, nsamples,N)
  for n in 1:nsamples
    data[n,:] = convertdatapoint(rawdata[n,:],bases[n,:],state=true)
  end
  
  # 1. Unnormalized
  ψ = initializetomography(N;χ=χ)
  num_grad = numgradsnll(ψ,data)
  alg_grad,loss = gradnll(ψ,data)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
  
  # 2. Globally normalized
  ψ = initializetomography(N;χ=χ)
  lognormalize!(ψ)
  num_grad = numgradsnll(ψ,data)
  alg_grad,loss = gradnll(ψ,data)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
  
  # 3. Locally normalized
  ψ = initializetomography(N;χ=χ)
  num_grad = numgradsnll(ψ,data)
  logZ,localnorms = lognormalize!(ψ)
  @test norm(ψ)^2 ≈ 1
  alg_grad_localnorm,loss = gradnll(ψ,data,localnorm=localnorms)
  for j in 1:N
    @test array(alg_grad_localnorm[j]) ≈ num_grad[j] rtol=1e-3
  end
end

@testset "mps-qst: full gradients" begin
  N = 5
  χ = 4
  nsamples = 100
  Random.seed!(1234)
  rawdata = rand(0:1,nsamples,N)
  bases = randombases(N,nsamples)
  data = Matrix{String}(undef, nsamples,N)
  for n in 1:nsamples
    data[n,:] = convertdatapoint(rawdata[n,:],bases[n,:],state=true)
  end
  
  # 1. Unnormalized
  ψ = initializetomography(N;χ=χ)
  logZ = 2.0*log(norm(ψ))
  NLL  = nll(ψ,data)
  ex_loss = logZ + NLL
  num_gradZ = numgradslogZ(ψ)
  num_gradNLL = numgradsnll(ψ,data)
  num_grads = num_gradZ + num_gradNLL
  
  alg_grads,loss = gradients(ψ,data)
  @test ex_loss ≈ loss
  for j in 1:N
    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
  end

  # 2. Globally normalized
  ψ = initializetomography(N;χ=χ)
  lognormalize!(ψ)
  num_gradZ = numgradslogZ(ψ)
  num_gradNLL = numgradsnll(ψ,data)
  num_grads = num_gradZ + num_gradNLL
  NLL  = nll(ψ,data)
  ex_loss = NLL
  @test norm(ψ)^2 ≈ 1
  
  alg_grads,loss = gradients(ψ,data)
  @test ex_loss ≈ loss
  for j in 1:N
    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
  end
  
  # 3. Locally normalized
  ψ = initializetomography(N;χ=χ)
  num_gradZ = numgradslogZ(ψ)
  num_gradNLL = numgradsnll(ψ,data)
  num_grads = num_gradZ + num_gradNLL
  
  logZ,localnorms = lognormalize!(ψ)
  NLL  = nll(ψ,data)
  ex_loss = NLL
  @test norm(ψ)^2 ≈ 1
  
  alg_grads,loss = gradients(ψ,data,localnorm=localnorms)
  @test ex_loss ≈ loss
  for j in 1:N
    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
  end
end

""" MPS PROCESS TOMOGRAPHY TESTS """

@testset "mps-qpt: grad logZ" begin

  N = 10
  χ = 4
  
  # 1. Unnormalized
  ψ = initializetomography(N;χ=χ)
  num_grad = numgradslogZ(ψ)
  alg_grad,logZ = gradlogZ(ψ)
  
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end

  # 2. Globally normalized
  ψ = initializetomography(N;χ=χ)
  logZ,_ = lognormalize!(ψ)
  #@test norm(ψ)^2 ≈ 2^(0.5*N)
  alg_grad,_ = gradlogZ(ψ)
  num_grad = numgradslogZ(ψ)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
  
  # 3. Locally normalized
  ψ = initializetomography(N;χ=χ)
  num_grad = numgradslogZ(ψ)

  logZ,localnorms = lognormalize!(ψ)
  #@test norm(ψ)^2 ≈ 2^(0.5*N)
  alg_grad,_ = gradlogZ(ψ,localnorm=localnorms)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
end

@testset "mps-qpt: grad nll" begin
  Nphysical = 4
  N = 2*Nphysical
  χ = 2
  nsamples = 100
  Random.seed!(1234)
  rawdata = rand(0:1,nsamples,N)
  bases = randombases(N,nsamples)
  data = Matrix{String}(undef, nsamples,N)
  for n in 1:nsamples
    data[n,:] = convertdatapoint(rawdata[n,:],bases[n,:],state=true)
  end
  
  # 1. Unnnomalized
  ψ = initializetomography(N;χ=χ)
  num_grad = numgradsnll(ψ,data,choi=true)
  alg_grad,loss = gradnll(ψ,data,choi=true)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
  
  # 2. Globally normalized
  ψ = initializetomography(N;χ=χ)
  lognormalize!(ψ)
  num_grad = numgradsnll(ψ,data,choi=true)
  #@test norm(ψ)^2 ≈ 2^(Nphysical)
  alg_grad,loss = gradnll(ψ,data;choi=true)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
  
  # 3. Locally normalized
  ψ = initializetomography(N;χ=χ)
  num_grad = numgradsnll(ψ,data,choi=true)
  logZ,localnorms = lognormalize!(ψ)
  #@test norm(ψ)^2 ≈ 2^(Nphysical)
  alg_grad,loss = gradnll(ψ,data,localnorm=localnorms,choi=true)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
end


@testset "mps-qpt: full gradients" begin
  Nphysical = 2
  N = 2 * Nphysical
  χ = 4
  nsamples = 100
  Random.seed!(1234)
  rawdata = rand(0:1,nsamples,N)
  bases = randombases(N,nsamples)
  data = Matrix{String}(undef, nsamples,N)
  for n in 1:nsamples
    data[n,:] = convertdatapoint(rawdata[n,:],bases[n,:],state=true)
  end
  
  # 1. Unnormalized
  ψ = initializetomography(N;χ=χ)
  logZ = 2.0*log(norm(ψ))
  NLL  = nll(ψ,data;choi=true)
  ex_loss = logZ + NLL - 0.5*N*log(2)
  num_gradZ = numgradslogZ(ψ)
  num_gradNLL = numgradsnll(ψ,data;choi=true)
  num_grads = num_gradZ + num_gradNLL

  alg_grads,loss = gradients(ψ,data;choi=true)
  @test ex_loss ≈ loss
  for j in 1:N
    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
  end

  # 2. Globally normalized
  ψ = initializetomography(N;χ=χ)
  lognormalize!(ψ)
  num_gradZ = numgradslogZ(ψ)
  num_gradNLL = numgradsnll(ψ,data;choi=true)
  num_grads = num_gradZ + num_gradNLL
  NLL  = nll(ψ,data;choi=true)
  ex_loss = NLL - 0.5*N*log(2)
  #@test norm(ψ)^2 ≈ 2^(Nphysical)
  
  alg_grads,loss = gradients(ψ,data;choi=true)
  @test ex_loss ≈ loss
  for j in 1:N
    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
  end
  
  # 3. Locally normalized
  ψ = initializetomography(N;χ=χ)
  num_gradZ = numgradslogZ(ψ)
  num_gradNLL = numgradsnll(ψ,data;choi=true)
  num_grads = num_gradZ + num_gradNLL
  
  logZ,localnorms = lognormalize!(ψ)
  NLL  = nll(ψ,data;choi=true)
  ex_loss = NLL - 0.5*N*log(2)
  
  alg_grads,loss = gradients(ψ,data,localnorm=localnorms,choi=true)
  @test ex_loss ≈ loss
  for j in 1:N
    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
  end
end


""" LPDO STATE TOMOGRAPHY TESTS """

@testset "lpdo-qst: lognormalization" begin
  N = 10
  χ = 4
  ξ = 2
  ρ = initializetomography(N;χ=χ,ξ=ξ)
  @test length(ρ) == N
  logZ1 = logtr(ρ)
  logZ2,_ = lognormalize!(ρ)
  @test logZ1 ≈ logZ2
  ρ = initializetomography(N;χ=χ,ξ=ξ)
  lognormalize!(ρ)
  @test tr(ρ) ≈ 1
end

@testset "lpdo-qst: density matrix properties" begin
  N = 5
  χ = 4
  ξ = 3
  ρ = initializetomography(N;χ=χ,ξ=ξ)
  @test length(ρ) == N
  lognormalize!(ρ)
  rho = MPO(ρ)
  rho_mat = fullmatrix(rho)
  @test sum(abs.(imag(diag(rho_mat)))) ≈ 0.0 atol=1e-10
  @test real(tr(rho_mat)) ≈ 1.0 atol=1e-10
  @test all(real(eigvals(rho_mat)) .≥ 0) 
end

@testset "lpdo-qst: grad logZ" begin
  N = 5
  χ = 4
  ξ = 3
  
  # 1. Unnormalized
  ρ = initializetomography(N;χ=χ,ξ=ξ)
  alg_grad,_ = gradlogZ(ρ)
  num_grad = numgradslogZ(ρ)
  alg_gradient = permutedims(array(alg_grad[1]),[1,3,2])
  @test alg_gradient ≈ num_grad[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grad[j]),[2,1,3,4])
    @test alg_gradient ≈ num_grad[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grad[N]),[2,1,3])
  @test alg_gradient ≈ num_grad[N] rtol=1e-3
  
  # 2. Globally normalizeid
  ρ = initializetomography(N;χ=χ,ξ=ξ)
  lognormalize!(ρ)
  @test tr(ρ) ≈ 1
  alg_grad,_ = gradlogZ(ρ)
  num_grad = numgradslogZ(ρ)
  
  alg_gradient = permutedims(array(alg_grad[1]),[1,3,2])
  @test alg_gradient ≈ num_grad[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grad[j]),[2,1,3,4])
    @test alg_gradient ≈ num_grad[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grad[N]),[2,1,3])
  @test alg_gradient ≈ num_grad[N] rtol=1e-3

  # 3. Locally normalized
  ρ = initializetomography(N;χ=χ,ξ=ξ)
  num_grad = numgradslogZ(ρ)

  logZ,localnorms = lognormalize!(ρ)
  @test tr(ρ) ≈ 1
  alg_grad,_ = gradlogZ(ρ,localnorm=localnorms)

  alg_gradient = permutedims(array(alg_grad[1]),[1,3,2])
  @test alg_gradient ≈ num_grad[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grad[j]),[2,1,3,4])
    @test alg_gradient ≈ num_grad[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grad[N]),[2,1,3])
  @test alg_gradient ≈ num_grad[N] rtol=1e-3

end 


@testset "lpdo-qst: grad nll" begin
  N = 5
  χ = 4
  ξ = 3
  nsamples = 100
  Random.seed!(1234)
  rawdata = rand(0:1,nsamples,N)
  bases = randombases(N,nsamples)
  data = Matrix{String}(undef, nsamples,N)
  for n in 1:nsamples
    data[n,:] = convertdatapoint(rawdata[n,:],bases[n,:],state=true)
  end
  
  # 1. Unnormalized
  ρ = initializetomography(N;χ=χ,ξ=ξ)

  num_grad = numgradsnll(ρ,data)
  alg_grad,loss = gradnll(ρ,data)
  ex_loss = nll(ρ,data)
  @test ex_loss ≈ loss
  alg_gradient = permutedims(array(alg_grad[1]),[3,1,2])
  @test alg_gradient ≈ num_grad[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grad[j]),[4,2,3,1])
    @test alg_gradient ≈ num_grad[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grad[N]),[3,1,2])
  @test alg_gradient ≈ num_grad[N] rtol=1e-3
  
  # 2. Globally normalized
  ρ = initializetomography(N;χ=χ,ξ=ξ)
  lognormalize!(ρ)
  @test tr(ρ) ≈ 1
  num_grad = numgradsnll(ρ,data)
  alg_grad,loss = gradnll(ρ,data)
  ex_loss = nll(ρ,data)
  @test ex_loss ≈ loss
  alg_gradient = permutedims(array(alg_grad[1]),[3,1,2])
  @test alg_gradient ≈ num_grad[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grad[j]),[4,2,3,1])
    @test alg_gradient ≈ num_grad[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grad[N]),[3,1,2])
  @test alg_gradient ≈ num_grad[N] rtol=1e-3

  # 3. Locally normalized
  ρ = initializetomography(N;χ=χ,ξ=ξ)
  num_grad = numgradsnll(ρ,data)
  logZ,localnorms = lognormalize!(ρ)
  @test tr(ρ) ≈ 1
  alg_grad,loss = gradnll(ρ,data,localnorm=localnorms)
  ex_loss = nll(ρ,data)
  @test ex_loss ≈ loss
  alg_gradient = permutedims(array(alg_grad[1]),[3,1,2])
  @test alg_gradient ≈ num_grad[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grad[j]),[4,2,3,1])
    @test alg_gradient ≈ num_grad[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grad[N]),[3,1,2])
  @test alg_gradient ≈ num_grad[N] rtol=1e-3

end

"""
PROCESS TOMOGRAPHY WITH LPDO
"""


@testset "lpdo-qpt: grad nll" begin
  Nphysical = 1
  N = 2 * Nphysical
  χ = 4
  ξ = 3
  nsamples = 100
  Random.seed!(1234)
  rawdata = rand(0:1,nsamples,N)
  bases = randombases(N,nsamples)
  data = Matrix{String}(undef, nsamples,N)
  for n in 1:nsamples
    data[n,:] = convertdatapoint(rawdata[n,:],bases[n,:],state=true)
  end
  
  # 1. Unnormalized
  Λ = initializetomography(N;χ=χ,ξ=ξ)

  num_grad = numgradsnll(Λ,data,choi=true)
  alg_grad,loss = gradnll(Λ,data,choi=true)
  ex_loss = nll(Λ,data,choi=true)
  @test ex_loss ≈ loss
  alg_gradient = permutedims(array(alg_grad[1]),[3,1,2])
  @test alg_gradient ≈ num_grad[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grad[j]),[4,2,3,1])
    @test alg_gradient ≈ num_grad[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grad[N]),[3,1,2])
  @test alg_gradient ≈ num_grad[N] rtol=1e-3
  
  # 2. Globally normalized
  Λ = initializetomography(N;χ=χ,ξ=ξ) 
  lognormalize!(Λ)
  num_grad = numgradsnll(Λ,data,choi=true)
  ex_loss = nll(Λ,data,choi=true) 
  alg_grad,loss = gradnll(Λ,data,choi=true)
  @test ex_loss ≈ loss 
  alg_gradient = permutedims(array(alg_grad[1]),[3,1,2])
  @test alg_gradient ≈ num_grad[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grad[j]),[4,2,3,1])
    @test alg_gradient ≈ num_grad[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grad[N]),[3,1,2])
  @test alg_gradient ≈ num_grad[N] rtol=1e-3

  # 3. Locally normalized
  Λ = initializetomography(N;χ=χ,ξ=ξ)
  num_grad = numgradsnll(Λ,data,choi=true)
  logZ,localnorms = lognormalize!(Λ)
  ex_loss = nll(Λ,data,choi=true) 
  alg_grad,loss = gradnll(Λ,data,localnorm=localnorms,choi=true)
  @test ex_loss ≈ loss
  alg_gradient = permutedims(array(alg_grad[1]),[3,1,2])
  @test alg_gradient ≈ num_grad[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grad[j]),[4,2,3,1])
    @test alg_gradient ≈ num_grad[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grad[N]),[3,1,2])
  @test alg_gradient ≈ num_grad[N] rtol=1e-3
  
end

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

@testset "trace distance" begin 

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


