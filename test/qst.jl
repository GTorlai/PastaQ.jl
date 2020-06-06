using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

""" HELPER FUNCTIONS """
function numgradslogZ(M::Union{MPS,MPO};accuracy=1e-8)
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

function numgradsnll(psi::MPS,data::Array;accuracy=1e-8)
  N = length(psi)
  grad_r = []
  grad_i = []
  for j in 1:N
    push!(grad_r,zeros(ComplexF64,size(psi[j])))
    push!(grad_i,zeros(ComplexF64,size(psi[j])))
  end
  
  epsilon = zeros(ComplexF64,size(psi[1]));
  # Site 1
  for i in 1:length(epsilon)
    epsilon[i] = accuracy
    eps = ITensor(epsilon,inds(psi[1]))
    psi[1] += eps
    loss_p = nll(psi,data) 
    psi[1] -= eps
    loss_m = nll(psi,data) 
    grad_r[1][i] = (loss_p-loss_m)/(accuracy)
    
    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(psi[1]))
    psi[1] += eps
    loss_p = nll(psi,data) 
    psi[1] -= eps
    loss_m = nll(psi,data) 
    grad_i[1][i] = (loss_p-loss_m)/(im*accuracy)
    
    epsilon[i] = 0.0
  end

  for j in 2:N-1
    epsilon = zeros(ComplexF64,size(psi[j]));
    for i in 1:length(epsilon)
      epsilon[i] = accuracy
      eps = ITensor(epsilon,inds(psi[j]))
      psi[j] += eps
      loss_p = nll(psi,data) 
      psi[j] -= eps
      loss_m = nll(psi,data) 
      grad_r[j][i] = (loss_p-loss_m)/(accuracy)
      
      epsilon[i] = im*accuracy
      eps = ITensor(epsilon,inds(psi[j]))
      psi[j] += eps
      loss_p = nll(psi,data) 
      psi[j] -= eps
      loss_m = nll(psi,data) 
      grad_i[j][i] = (loss_p-loss_m)/(im*accuracy)

      epsilon[i] = 0.0
    end
 end

  # Site N
  epsilon = zeros(ComplexF64,size(psi[N]));
  for i in 1:length(epsilon)
    epsilon[i] = accuracy
    eps = ITensor(epsilon,inds(psi[N]))
    psi[N] += eps
    loss_p = nll(psi,data) 
    psi[N] -= eps
    loss_m = nll(psi,data) 
    grad_r[N][i] = (loss_p-loss_m)/(accuracy)

    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(psi[N]))
    psi[N] += eps
    loss_p = nll(psi,data) 
    psi[N] -= eps
    loss_m = nll(psi,data) 
    grad_i[N][i] = (loss_p-loss_m)/(im*accuracy)
    
    epsilon[i] = 0.0
  end

  return grad_r-grad_i
end

function numgradsnll(psi::MPS,data::Array,bases::Array;accuracy=1e-8)
  N = length(psi)
  grad_r = []
  grad_i = []
  for j in 1:N
    push!(grad_r,zeros(ComplexF64,size(psi[j])))
    push!(grad_i,zeros(ComplexF64,size(psi[j])))
  end
  
  epsilon = zeros(ComplexF64,size(psi[1]));
  # Site 1
  for i in 1:length(epsilon)
    epsilon[i] = accuracy
    eps = ITensor(epsilon,inds(psi[1]))
    psi[1] += eps
    loss_p = nll(psi,data,bases) 
    psi[1] -= eps
    loss_m = nll(psi,data,bases) 
    grad_r[1][i] = (loss_p-loss_m)/(accuracy)
    
    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(psi[1]))
    psi[1] += eps
    loss_p = nll(psi,data,bases) 
    psi[1] -= eps
    loss_m = nll(psi,data,bases) 
    grad_i[1][i] = (loss_p-loss_m)/(im*accuracy)
    
    epsilon[i] = 0.0
  end

  for j in 2:N-1
    epsilon = zeros(ComplexF64,size(psi[j]));
    for i in 1:length(epsilon)
      epsilon[i] = accuracy
      eps = ITensor(epsilon,inds(psi[j]))
      psi[j] += eps
      loss_p = nll(psi,data,bases) 
      psi[j] -= eps
      loss_m = nll(psi,data,bases) 
      grad_r[j][i] = (loss_p-loss_m)/(accuracy)
      
      epsilon[i] = im*accuracy
      eps = ITensor(epsilon,inds(psi[j]))
      psi[j] += eps
      loss_p = nll(psi,data,bases) 
      psi[j] -= eps
      loss_m = nll(psi,data,bases) 
      grad_i[j][i] = (loss_p-loss_m)/(im*accuracy)

      epsilon[i] = 0.0
    end
 end

  # Site N
  epsilon = zeros(ComplexF64,size(psi[N]));
  for i in 1:length(epsilon)
    epsilon[i] = accuracy
    eps = ITensor(epsilon,inds(psi[N]))
    psi[N] += eps
    loss_p = nll(psi,data,bases) 
    psi[N] -= eps
    loss_m = nll(psi,data,bases) 
    grad_r[N][i] = (loss_p-loss_m)/(accuracy)

    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(psi[N]))
    psi[N] += eps
    loss_p = nll(psi,data,bases) 
    psi[N] -= eps
    loss_m = nll(psi,data,bases) 
    grad_i[N][i] = (loss_p-loss_m)/(im*accuracy)
    
    epsilon[i] = 0.0
  end

  return grad_r-grad_i
end




#""" MPS TOMOGRAPHY TESTS """
#
#@testset "mps-qst: lognormalization" begin
#  N = 10
#  χ = 4
#  psi = initializeQST(N,χ)
#  @test length(psi) == N
#  logZ1 = 2.0*lognorm(psi)
#  logZ2,_ = lognormalize!(psi)
#  @test logZ1 ≈ logZ2
#  psi = initializeQST(N,χ)
#  lognormalize!(psi)
#  @test norm(psi) ≈ 1
#end
#
#@testset "mps-qst: grad logZ" begin
#  N = 5
#  χ = 4
#  psi = initializeQST(N,χ)
#  alg_grad,_ = gradlogZ(psi)
#  num_grad = numgradslogZ(psi)
#  for j in 1:N
#    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
#  end
#  psi = initializeQST(N,χ)
#  logZ,localnorms = lognormalize!(psi)
#  @test norm(psi)^2 ≈ 1
#  alg_grad_localnorm,_ = gradlogZ(psi,localnorm=localnorms)
#  for j in 1:N
#    @test array(alg_grad[j]) ≈ array(alg_grad_localnorm[j]) rtol=1e-3
#  end
#end
#
#@testset "mps-qst: grad nll" begin
#  N = 5
#  χ = 4
#  nsamples = 100
#  Random.seed!(1234)
#  data = rand(0:1,nsamples,N)
#  bases = generatemeasurementsettings(N,nsamples,bases_id=["X","Y","Z"]) 
#  psi = initializeQST(N,χ)
#  num_grad = numgradsnll(psi,data,bases)
#  alg_grad,loss = gradnll(psi,data,bases)
#  for j in 1:N
#    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
#  end
#  psi = initializeQST(N,χ)
#  logZ,localnorms = lognormalize!(psi)
#  @test norm(psi)^2 ≈ 1
#  alg_grad_localnorm,loss = gradnll(psi,data,bases,localnorm=localnorms)
#  for j in 1:N
#    @test array(alg_grad[j]) ≈ array(alg_grad_localnorm[j]) rtol=1e-3
#  end
#end
#
#
#@testset "mps-qst: full gradients" begin
#  N = 5
#  χ = 4
#  nsamples = 100
#  Random.seed!(1234)
#  data = rand(0:1,nsamples,N)
#  bases = generatemeasurementsettings(N,nsamples,bases_id=["X","Y","Z"])
#
#  psi = initializeQST(N,χ)
#  logZ = 2.0*log(norm(psi))
#  NLL  = nll(psi,data,bases)
#  ex_loss = logZ + NLL
#  num_gradZ = numgradslogZ(psi)
#  num_gradNLL = numgradsnll(psi,data,bases)
#  num_grads = num_gradZ + num_gradNLL
#
#  alg_grads,loss = gradients(psi,data,bases)
#  @test ex_loss ≈ loss
#  for j in 1:N
#    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
#  end
#
#  psi = initializeQST(N,χ)
#  logZ,localnorms = lognormalize!(psi)
#  NLL  = nll(psi,data,bases)
#  ex_loss = NLL
#  @test norm(psi)^2 ≈ 1
#  
#  alg_grads,loss = gradients(psi,data,bases,localnorm=localnorms)
#  @test ex_loss ≈ loss
#  for j in 1:N
#    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
#  end
#end
#

""" LPDO TOMOGRAPHY TESTS """
#
#@testset "lpdo-qst: lognormalization" begin
#  N = 10
#  χ = 4
#  ξ = 2
#  lpdo = initializeQST(N,χ,ξ)
#  @test length(lpdo) == N
#  logZ1 = 2.0*lognorm(lpdo)
#  logZ2,_ = lognormalize!(lpdo)
#  @test logZ1 ≈ logZ2
#  lpdo = initializeQST(N,χ,ξ)
#  lognormalize!(lpdo)
#  @test norm(lpdo) ≈ 1
#end

@testset "lpdo-qst: complex grad logZ" begin
  N = 5
  χ = 4
  ξ = 3
  lpdo = initializeQST(N,χ,ξ)
  alg_grad,_ = gradlogZ(lpdo)
  num_grad = numgradslogZ(lpdo)
  
  alg_gradient = permutedims(array(alg_grad[1]),[1,3,2])
  @test alg_gradient ≈ num_grad[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grad[j]),[2,1,3,4])
    @test alg_gradient ≈ num_grad[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grad[N]),[2,1,3])
  @test alg_gradient ≈ num_grad[N] rtol=1e-3
  
  lpdo = initializeQST(N,χ,ξ)
  logZ,localnorms = lognormalize!(lpdo)
  @test norm(lpdo)^2 ≈ 1
  alg_grad,_ = gradlogZ(lpdo,localnorm=localnorms)
  
  alg_gradient = permutedims(array(alg_grad[1]),[1,3,2])
  @test alg_gradient ≈ num_grad[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grad[j]),[2,1,3,4])
    @test alg_gradient ≈ num_grad[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grad[N]),[2,1,3])
  @test alg_gradient ≈ num_grad[N] rtol=1e-3
end


