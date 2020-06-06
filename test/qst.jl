using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

function numgradslogZ(psi::MPS;accuracy=1e-8)
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
    loss_p = 2.0*lognorm(psi)
    psi[1] -= eps
    loss_m = 2.0*lognorm(psi)
    grad_r[1][i] = (loss_p-loss_m)/(accuracy)
    
    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(psi[1]))
    psi[1] += eps
    loss_p = 2.0*lognorm(psi)
    psi[1] -= eps
    loss_m = 2.0*lognorm(psi)
    grad_i[1][i] = (loss_p-loss_m)/(im*accuracy)
    
    epsilon[i] = 0.0
  end

  for j in 2:N-1
    epsilon = zeros(ComplexF64,size(psi[j]));
    for i in 1:length(epsilon)
      epsilon[i] = accuracy
      eps = ITensor(epsilon,inds(psi[j]))
      psi[j] += eps
      loss_p = 2.0*lognorm(psi)
      psi[j] -= eps
      loss_m = 2.0*lognorm(psi)
      grad_r[j][i] = (loss_p-loss_m)/(accuracy)
      
      epsilon[i] = im*accuracy
      eps = ITensor(epsilon,inds(psi[j]))
      psi[j] += eps
      loss_p = 2.0*lognorm(psi)
      psi[j] -= eps
      loss_m = 2.0*lognorm(psi)
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
    loss_p = 2.0*lognorm(psi)
    psi[N] -= eps
    loss_m = 2.0*lognorm(psi)
    grad_r[N][i] = (loss_p-loss_m)/(accuracy)

    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(psi[N]))
    psi[N] += eps
    loss_p = 2.0*lognorm(psi)
    psi[N] -= eps
    loss_m = 2.0*lognorm(psi)
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

@testset "qst: lognormalization" begin
  N = 10
  qst = QST(N=N)
  logZ1 = 2.0*lognorm(qst.psi)
  logZ2,_ = lognormalize!(qst.psi)
  @test logZ1 ≈ logZ2
  qst = QST(N=N)
  lognormalize!(qst.psi)
  @test norm(qst.psi) ≈ 1
end

@testset "qst: complex grad logZ" begin
  N = 5
  qst = QST(N=N)
  alg_grad,_ = gradlogZ(qst.psi)
  num_grad = numgradslogZ(qst.psi)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
  qst = QST(N=N)
  logZ,localnorms = lognormalize!(qst.psi)
  @test norm(qst.psi)^2 ≈ 1
  alg_grad_localnorm,_ = gradlogZ(qst.psi,localnorm=localnorms)
  for j in 1:N
    @test array(alg_grad[j]) ≈ array(alg_grad_localnorm[j]) rtol=1e-3
  end
end

@testset "qst: complex grad nll with bases" begin
  N = 5
  nsamples = 100
  Random.seed!(1234)
  data = rand(0:1,nsamples,N)
  bases = generatemeasurementsettings(N,nsamples,bases_id=["X","Y","Z"]) 
  qst = QST(N=N)
  num_grad = numgradsnll(qst.psi,data,bases)
  alg_grad,loss = gradnll(qst.psi,data,bases)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
  qst = QST(N=N)
  logZ,localnorms = lognormalize!(qst.psi)
  @test norm(qst.psi)^2 ≈ 1
  alg_grad_localnorm,loss = gradnll(qst.psi,data,bases,localnorm=localnorms)
  for j in 1:N
    @test array(alg_grad[j]) ≈ array(alg_grad_localnorm[j]) rtol=1e-3
  end
end


@testset "qst: full gradients" begin
  N = 5
  nsamples = 100
  Random.seed!(1234)
  data = rand(0:1,nsamples,N)
  bases = generatemeasurementsettings(N,nsamples,bases_id=["X","Y","Z"])

  qst = QST(N=N)
  logZ = 2.0*log(norm(qst.psi))
  NLL  = nll(qst.psi,data,bases)
  ex_loss = logZ + NLL
  num_gradZ = numgradslogZ(qst.psi)
  num_gradNLL = numgradsnll(qst.psi,data,bases)
  num_grads = num_gradZ + num_gradNLL

  alg_grads,loss = gradients(qst.psi,data,bases)
  @test ex_loss ≈ loss
  for j in 1:N
    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
  end

  qst = QST(N=N)
  logZ,localnorms = lognormalize!(qst.psi)
  NLL  = nll(qst.psi,data,bases)
  ex_loss = NLL
  @test norm(qst.psi)^2 ≈ 1
  
  alg_grads,loss = gradients(qst.psi,data,bases,localnorm=localnorms)
  @test ex_loss ≈ loss
  for j in 1:N
    @test array(alg_grads[j]) ≈ num_grads[j] rtol=1e-3
  end
end


