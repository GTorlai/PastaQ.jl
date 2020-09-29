using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

function splitstatewrapper(N::Int64,χ::Int64,ξ::Int64)
  split_version = true
  if split_version
    return randomstate(2*N;mixed=true,χ=χ,ξ=ξ)
  else
    return randomprocess(2*N;mixed=trueχ=χ,ξ=ξ)
  end
end

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
    loss_p = PastaQ.nll(L,data,choi=choi) 
    M[1] -= eps
    loss_m = PastaQ.nll(L,data,choi=choi) 
    grad_r[1][i] = (loss_p-loss_m)/(accuracy)
    
    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(M[1]))
    M[1] += eps
    loss_p = PastaQ.nll(L,data,choi=choi) 
    M[1] -= eps
    loss_m = PastaQ.nll(L,data,choi=choi) 
    grad_i[1][i] = (loss_p-loss_m)/(im*accuracy)
    
    epsilon[i] = 0.0
  end

  for j in 2:N-1
    epsilon = zeros(ComplexF64,size(M[j]));
    for i in 1:length(epsilon)
      epsilon[i] = accuracy
      eps = ITensor(epsilon,inds(M[j]))
      M[j] += eps
      loss_p = PastaQ.nll(L,data,choi=choi) 
      M[j] -= eps
      loss_m = PastaQ.nll(L,data,choi=choi) 
      grad_r[j][i] = (loss_p-loss_m)/(accuracy)
      
      epsilon[i] = im*accuracy
      eps = ITensor(epsilon,inds(M[j]))
      M[j] += eps
      loss_p = PastaQ.nll(L,data,choi=choi) 
      M[j] -= eps
      loss_m = PastaQ.nll(L,data,choi=choi) 
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
    loss_p = PastaQ.nll(L,data,choi=choi) 
    M[N] -= eps
    loss_m = PastaQ.nll(L,data,choi=choi) 
    grad_r[N][i] = (loss_p-loss_m)/(accuracy)

    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(M[N]))
    M[N] += eps
    loss_p = PastaQ.nll(L,data,choi=choi) 
    M[N] -= eps
    loss_m = PastaQ.nll(L,data,choi=choi) 
    grad_i[N][i] = (loss_p-loss_m)/(im*accuracy)
    
    epsilon[i] = 0.0
  end

  return grad_r-grad_i
end

numgradsnll(M::MPS, args...; kwargs...) =
  numgradsnll(LPDO(M), args...; kwargs...)

""" MPS STATE TOMOGRAPHY TESTS """

@testset "mps-qst: normalization" begin
  N = 10
  χ = 4
  ψ = randomstate(N;χ=χ)
  @test length(ψ) == N
  logZ = lognorm(ψ)
  localZ = []
  normalize!(ψ; localnorms! = localZ)
  @test logZ ≈ sum(log.(localZ))
  @test norm(ψ) ≈ 1
end

@testset "mps-qst: grad logZ" begin
  N = 5
  χ = 4
  
  # 1. Unnormalized
  ψ = randomstate(N;χ=χ)
  alg_grad,_ = PastaQ.gradlogZ(ψ)
  num_grad = numgradslogZ(ψ)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
  
  # 2. Globally normalized
  ψ = randomstate(N;χ=χ)
  normalize!(ψ)
  @test norm(ψ)^2 ≈ 1
  alg_grad,_ = PastaQ.gradlogZ(ψ)
  num_grad = numgradslogZ(ψ)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end

  # 3. Locally normalized
  ψ = randomstate(N;χ=χ)
  num_grad = numgradslogZ(ψ)

  localnorms = []
  normalize!(ψ; localnorms! = localnorms)
  @test norm(ψ) ≈ 1
  alg_grad,_ = PastaQ.gradlogZ(ψ; localnorms = localnorms)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
end

@testset "mps-qst: grad nll" begin
  N = 5
  χ = 4
  nsamples = 10
  Random.seed!(1234)
  rawdata = rand(0:1,nsamples,N)
  bases = randombases(N,nsamples)
  data = Matrix{String}(undef, nsamples,N)
  for n in 1:nsamples
    data[n,:] = PastaQ.convertdatapoint(rawdata[n,:],bases[n,:],state=true)
  end
  
  # 1. Unnormalized
  ψ = randomstate(N;χ=χ)
  num_grad = numgradsnll(ψ,data)
  alg_grad,loss = PastaQ.gradnll(ψ,data)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
  
  # 2. Globally normalized
  ψ = randomstate(N;χ=χ)
  normalize!(ψ)
  num_grad = numgradsnll(ψ,data)
  alg_grad,loss = PastaQ.gradnll(ψ,data)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
  
  # 3. Locally normalized
  ψ = randomstate(N;χ=χ)
  num_grad = numgradsnll(ψ,data)
  localnorms = []
  normalize!(ψ; localnorms! = localnorms)
  @test norm(ψ) ≈ 1
  alg_grad_localnorm, loss = PastaQ.gradnll(ψ, data; localnorms = localnorms)
  for j in 1:N
    @test array(alg_grad_localnorm[j]) ≈ num_grad[j] rtol=1e-3
  end
end

@testset "mps-qst: full gradients" begin
  N = 5
  χ = 4
  nsamples = 10
  Random.seed!(1234)
  rawdata = rand(0:1,nsamples,N)
  bases = randombases(N,nsamples)
  data = Matrix{String}(undef, nsamples,N)
  for n in 1:nsamples
    data[n,:] = PastaQ.convertdatapoint(rawdata[n,:],bases[n,:],state=true)
  end
  
  # 1. Unnormalized
  ψ = randomstate(N;χ=χ)
  logZ = 2.0*log(norm(ψ))
  NLL  = PastaQ.nll(ψ,data)
  ex_loss = logZ + NLL
  num_gradZ = numgradslogZ(ψ)
  num_gradNLL = numgradsnll(ψ,data)
  num_grads = num_gradZ + num_gradNLL
  
  alg_grads,loss = PastaQ.gradients(ψ,data)
  @test ex_loss ≈ loss
  for j in 1:N
    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
  end

  # 2. Globally normalized
  ψ = randomstate(N;χ=χ)
  normalize!(ψ)
  num_gradZ = numgradslogZ(ψ)
  num_gradNLL = numgradsnll(ψ,data)
  num_grads = num_gradZ + num_gradNLL
  NLL  = PastaQ.nll(ψ,data)
  ex_loss = NLL
  @test norm(ψ)^2 ≈ 1
  
  alg_grads,loss = PastaQ.gradients(ψ,data)
  @test ex_loss ≈ loss
  for j in 1:N
    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
  end
  
  # 3. Locally normalized
  ψ = randomstate(N;χ=χ)
  num_gradZ = numgradslogZ(ψ)
  num_gradNLL = numgradsnll(ψ,data)
  num_grads = num_gradZ + num_gradNLL
  
  localnorms = []
  normalize!(ψ; localnorms! = localnorms)
  NLL  = PastaQ.nll(ψ,data)
  ex_loss = NLL
  @test norm(ψ)^2 ≈ 1
  
  alg_grads,loss = PastaQ.gradients(ψ, data; localnorms = localnorms)
  @test ex_loss ≈ loss
  for j in 1:N
    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
  end
end

""" MPS PROCESS TOMOGRAPHY TESTS """
# TODO: Change after the unsplit tomography is in place

@testset "mps-qpt: grad nll" begin
  Nphysical = 4
  N = Nphysical
  χ = 2
  nsamples = 10
  Random.seed!(1234)
  rawdata = rand(0:1,nsamples,N)
  bases = randombases(N,nsamples)
  data = Matrix{String}(undef, nsamples,N)
  for n in 1:nsamples
    data[n,:] = PastaQ.convertdatapoint(rawdata[n,:],bases[n,:],state=true)
  end
  
  # 1. Unnnomalized
  ψ = randomstate(N;χ=χ)
  num_grad = numgradsnll(ψ,data,choi=true)
  alg_grad,loss = PastaQ.gradnll(ψ,data,choi=true)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
  
  # 2. Globally normalized
  ψ = randomstate(N;χ=χ)
  normalize!(ψ)
  num_grad = numgradsnll(ψ,data,choi=true)
  #@test norm(ψ)^2 ≈ 2^(Nphysical)
  alg_grad,loss = PastaQ.gradnll(ψ,data;choi=true)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
  
  # 3. Locally normalized
  ψ = randomstate(N;χ=χ)
  num_grad = numgradsnll(ψ,data,choi=true)
  localnorms = []
  normalize!(ψ; localnorms! = localnorms)
  #@test norm(ψ)^2 ≈ 2^(Nphysical)
  alg_grad,loss = PastaQ.gradnll(ψ, data, localnorms = localnorms, choi = true)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
end


@testset "mps-qpt: full gradients" begin
  Nphysical = 2
  N = Nphysical
  χ = 4
  nsamples = 10
  Random.seed!(1234)
  rawdata = rand(0:1,nsamples,N)
  bases = randombases(N,nsamples)
  data = Matrix{String}(undef, nsamples,N)
  for n in 1:nsamples
    data[n,:] = PastaQ.convertdatapoint(rawdata[n,:],bases[n,:],state=true)
  end
  
  # 1. Unnormalized
  ψ = randomstate(N;χ=χ)
  logZ = 2.0*log(norm(ψ))
  NLL  = PastaQ.nll(ψ,data;choi=true)
  ex_loss = logZ + NLL - 0.5*N*log(2)
  num_gradZ = numgradslogZ(ψ)
  num_gradNLL = numgradsnll(ψ,data;choi=true)
  num_grads = num_gradZ + num_gradNLL

  alg_grads,loss = PastaQ.gradients(ψ,data;choi=true)
  @test ex_loss ≈ loss
  for j in 1:N
    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
  end

  # 2. Globally normalized
  ψ = randomstate(N;χ=χ)
  normalize!(ψ)
  num_gradZ = numgradslogZ(ψ)
  num_gradNLL = numgradsnll(ψ,data;choi=true)
  num_grads = num_gradZ + num_gradNLL
  NLL  = PastaQ.nll(ψ,data;choi=true)
  ex_loss = NLL - 0.5*N*log(2)
  #@test norm(ψ)^2 ≈ 2^(Nphysical)
  
  alg_grads,loss = PastaQ.gradients(ψ,data;choi=true)
  @test ex_loss ≈ loss
  for j in 1:N
    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
  end
  
  # 3. Locally normalized
  ψ = randomstate(N;χ=χ)
  num_gradZ = numgradslogZ(ψ)
  num_gradNLL = numgradsnll(ψ,data;choi=true)
  num_grads = num_gradZ + num_gradNLL
  
  localnorms = []
  normalize!(ψ; localnorms! = localnorms)
  NLL  = PastaQ.nll(ψ,data;choi=true)
  ex_loss = NLL - 0.5*N*log(2)
  
  alg_grads,loss = PastaQ.gradients(ψ, data; localnorms = localnorms, choi = true)
  @test ex_loss ≈ loss
  for j in 1:N
    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
  end
end
""" LPDO STATE TOMOGRAPHY TESTS """

@testset "lpdo-qst: normalization" begin
  N = Nqubits = 5
  N = 2*Nqubits
  χ = 4
  ξ = 2
  
  #ρ = randomprocess(N;mixed=true,χ=χ,ξ=ξ)
  ρ = splitstatewrapper(Nqubits,χ,ξ)

  @test length(ρ) == N
  logZ = logtr(ρ)
  sqrt_localZ = []
  normalize!(ρ; sqrt_localnorms! = sqrt_localZ)
  @test logZ ≈ 2 * sum(log.(sqrt_localZ))
  #ρ = randomprocess(N;mixed=true,χ=χ,ξ=ξ)
  ρ = splitstatewrapper(Nqubits,χ,ξ)
  normalize!(ρ)
  @test tr(ρ) ≈ 1

  #ρ = MPO(randomprocess(N;mixed=true,χ=χ,ξ=ξ))
  ρ = MPO(splitstatewrapper(Nqubits,χ,ξ))
  trρ = tr(ρ)
  localtrρ = []
  normalize!(ρ; localnorms! = localtrρ)
  @test trρ ≈ prod(localtrρ)
  @test tr(ρ) ≈ 1
end

@testset "lpdo-qst: grad logZ" begin
  Nqubits = 3
  N = 2*Nqubits
  χ = 4
  ξ = 3
  
  # 1. Unnormalized
  #ρ = randomprocess(Nqubits;mixed=true,χ=χ,ξ=ξ)
  ρ = splitstatewrapper(Nqubits,χ,ξ)
  alg_grad,_ = PastaQ.gradlogZ(ρ)
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
  #ρ = randomprocess(Nqubits;mixed=true,χ=χ,ξ=ξ)
  ρ = splitstatewrapper(Nqubits,χ,ξ)
  normalize!(ρ)
  @test tr(ρ) ≈ 1
  alg_grad,_ = PastaQ.gradlogZ(ρ)
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
  #ρ = randomprocess(Nqubits;mixed=true,χ=χ,ξ=ξ)
  ρ = splitstatewrapper(Nqubits,χ,ξ)
  num_grad = numgradslogZ(ρ)

  sqrt_localnorms = []
  normalize!(ρ; sqrt_localnorms! = sqrt_localnorms)
  @test tr(ρ) ≈ 1
  alg_grad,_ = PastaQ.gradlogZ(ρ, sqrt_localnorms = sqrt_localnorms)

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
  Nqubits = 3
  N = 2*Nqubits
  χ = 4
  ξ = 3
  nsamples = 10
  Random.seed!(1234)
  rawdata = rand(0:1,nsamples,N)
  bases = randombases(N,nsamples)
  data = Matrix{String}(undef, nsamples,N)
  for n in 1:nsamples
    data[n,:] = PastaQ.convertdatapoint(rawdata[n,:],bases[n,:],state=true)
  end
  
  # 1. Unnormalized
  #ρ = randomprocess(Nqubits;mixed=true,χ=χ,ξ=ξ)
  ρ = splitstatewrapper(Nqubits,χ,ξ)

  num_grad = numgradsnll(ρ,data)
  alg_grad,loss = PastaQ.gradnll(ρ,data)
  ex_loss = PastaQ.nll(ρ,data)
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
  #ρ = randomprocess(Nqubits;mixed=true,χ=χ,ξ=ξ)
  ρ = splitstatewrapper(Nqubits,χ,ξ)
  normalize!(ρ)
  @test tr(ρ) ≈ 1
  num_grad = numgradsnll(ρ,data)
  alg_grad,loss = PastaQ.gradnll(ρ,data)
  ex_loss = PastaQ.nll(ρ,data)
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
  #ρ = randomprocess(Nqubits;mixed=true,χ=χ,ξ=ξ)
  ρ = splitstatewrapper(Nqubits,χ,ξ)
  num_grad = numgradsnll(ρ,data)
  sqrt_localnorms = []
  normalize!(ρ; sqrt_localnorms! = sqrt_localnorms)
  @test tr(ρ) ≈ 1
  alg_grad,loss = PastaQ.gradnll(ρ, data; sqrt_localnorms = sqrt_localnorms)
  ex_loss = PastaQ.nll(ρ,data)
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

#TODO This works with the plit version only
@testset "lpdo-qpt: grad nll" begin
  Nqubits= 3
  N = 2*Nqubits
  χ = 4
  ξ = 3
  nsamples = 10
  Random.seed!(1234)
  rawdata = rand(0:1,nsamples,N)
  bases = randombases(N,nsamples)
  data = Matrix{String}(undef, nsamples,N)
  for n in 1:nsamples
    data[n,:] = PastaQ.convertdatapoint(rawdata[n,:],bases[n,:],state=true)
  end
  
  # 1. Unnormalized
  #Λ = randomprocess(N;mixed=true,χ=χ,ξ=ξ)
  Λ = splitstatewrapper(Nqubits,χ,ξ)
  num_grad = numgradsnll(Λ,data,choi=true)
  alg_grad,loss = PastaQ.gradnll(Λ,data,choi=true)
  ex_loss = PastaQ.nll(Λ,data,choi=true)
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
  #Λ = randomprocess(N;mixed=true,χ=χ,ξ=ξ)
  Λ = splitstatewrapper(Nqubits,χ,ξ)
  normalize!(Λ)
  num_grad = numgradsnll(Λ,data,choi=true)
  ex_loss = PastaQ.nll(Λ,data,choi=true) 
  alg_grad,loss = PastaQ.gradnll(Λ,data,choi=true)
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
  #Λ = randomprocess(N;mixed=true,χ=χ,ξ=ξ)
  Λ = splitstatewrapper(Nqubits,χ,ξ)
  num_grad = numgradsnll(Λ,data,choi=true)
  sqrt_localnorms = []
  normalize!(Λ; sqrt_localnorms! = sqrt_localnorms)
  ex_loss = PastaQ.nll(Λ, data; choi = true) 
  alg_grad,loss = PastaQ.gradnll(Λ, data; sqrt_localnorms = sqrt_localnorms, choi = true)
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

@testset "splitunitary" begin
  
  N = 4 
  U0 = randomprocess(N;mixed=false,χ=4)
  U0prod = prod(U0)
  V0 = randomprocess(U0;mixed=false,χ=4)
  Ψ = PastaQ.splitunitary(U0)
  U = PastaQ.unsplitunitary(Ψ)
  @test prod(U) ≈ prod(U0)
end

@testset "splitchoi" begin
  N = 3
  gates = randomcircuit(N,2)
  Λ0 = runcircuit(N,gates;process=true,noise="AD",γ=0.1)
  ρ = PastaQ.splitchoi(Λ0)
  Λ = PastaQ.unsplitchoi(ρ)
  @test prod(Λ0.M) ≈ prod(Λ.M)

  Λ0 = randomprocess(N;mixed=true)
  ρ = PastaQ.splitchoi(Λ0)
  Λ = PastaQ.unsplitchoi(ρ)
  @test prod(Λ.M.X) ≈ prod(Λ.M.X)

end
