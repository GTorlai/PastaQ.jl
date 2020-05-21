using PastaQ
using ITensors
using Test
using LinearAlgebra

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
    loss_p = lognormalization(psi)
    psi[1] -= eps
    loss_m = lognormalization(psi)
    grad_r[1][i] = (loss_p-loss_m)/(accuracy)
    
    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(psi[1]))
    psi[1] += eps
    loss_p = lognormalization(psi)
    psi[1] -= eps
    loss_m = lognormalization(psi)
    grad_i[1][i] = (loss_p-loss_m)/(im*accuracy)
    
    epsilon[i] = 0.0
  end

  for j in 2:N-1
    epsilon = zeros(ComplexF64,size(psi[j]));
    for i in 1:length(epsilon)
      epsilon[i] = accuracy
      eps = ITensor(epsilon,inds(psi[j]))
      psi[j] += eps
      loss_p = lognormalization(psi)
      psi[j] -= eps
      loss_m = lognormalization(psi)
      grad_r[j][i] = (loss_p-loss_m)/(accuracy)
      
      epsilon[i] = im*accuracy
      eps = ITensor(epsilon,inds(psi[j]))
      psi[j] += eps
      loss_p = lognormalization(psi)
      psi[j] -= eps
      loss_m = lognormalization(psi)
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
    loss_p = lognormalization(psi)
    psi[N] -= eps
    loss_m = lognormalization(psi)
    grad_r[N][i] = (loss_p-loss_m)/(accuracy)

    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(psi[N]))
    psi[N] += eps
    loss_p = lognormalization(psi)
    psi[N] -= eps
    loss_m = lognormalization(psi)
    grad_i[N][i] = (loss_p-loss_m)/(im*accuracy)
    
    epsilon[i] = 0.0
  end

  return grad_r-grad_i
end



@testset "qst: lognormalization" begin
  N = 10
  qst = QST(N=N)
  logZ = lognormalization(qst.psi)
  Z = normalization(qst.psi)
  @test logZ ≈ log(Z)
end

@testset "qst: grad logZ" begin
  N = 5
  qst = QST(N=N)
  alg_grad,_ = gradlogZ(qst.psi)
  num_grad = numgradslogZ(qst.psi)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] atol=1e-5
  end
end

