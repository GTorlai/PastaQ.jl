using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

#function convertdatapoint(datapoint::Array{Int64}, basis::Array{String})
#  data0 = PastaQ.convertdatapoint(datapoint,basis)
#  return PastaQ.convertdatapoint(data0; state = true)
#end

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

#numgradslogZ(M::MPS; kwargs...) = numgradslogZ(LPDO(M); kwargs...)

function numgradsnll(C::Choi,
                     data_in::Array,
                     data_out::Array;
                     accuracy=1e-8)
  L = C.M
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
    loss_p = PastaQ.nll(C,data_in,data_out) 
    M[1] -= eps
    loss_m = PastaQ.nll(C,data_in,data_out) 
    grad_r[1][i] = (loss_p-loss_m)/(accuracy)
    
    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(M[1]))
    M[1] += eps
    loss_p = PastaQ.nll(C,data_in,data_out) 
    M[1] -= eps
    loss_m = PastaQ.nll(C,data_in,data_out) 
    grad_i[1][i] = (loss_p-loss_m)/(im*accuracy)
    
    epsilon[i] = 0.0
  end

  for j in 2:N-1
    epsilon = zeros(ComplexF64,size(M[j]));
    for i in 1:length(epsilon)
      epsilon[i] = accuracy
      eps = ITensor(epsilon,inds(M[j]))
      M[j] += eps
      loss_p = PastaQ.nll(C,data_in,data_out) 
      M[j] -= eps
      loss_m = PastaQ.nll(C,data_in,data_out) 
      grad_r[j][i] = (loss_p-loss_m)/(accuracy)
      
      epsilon[i] = im*accuracy
      eps = ITensor(epsilon,inds(M[j]))
      M[j] += eps
      loss_p = PastaQ.nll(C,data_in,data_out) 
      M[j] -= eps
      loss_m = PastaQ.nll(C,data_in,data_out) 
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
    loss_p = PastaQ.nll(C,data_in,data_out) 
    M[N] -= eps
    loss_m = PastaQ.nll(C,data_in,data_out) 
    grad_r[N][i] = (loss_p-loss_m)/(accuracy)

    epsilon[i] = im*accuracy
    eps = ITensor(epsilon,inds(M[N]))
    M[N] += eps
    loss_p = PastaQ.nll(C,data_in,data_out) 
    M[N] -= eps
    loss_m = PastaQ.nll(C,data_in,data_out) 
    grad_i[N][i] = (loss_p-loss_m)/(im*accuracy)
    
    epsilon[i] = 0.0
  end

  return grad_r-grad_i
end

numgradsnll(M::MPS, args...; kwargs...) =
  numgradsnll(LPDO(M), args...; kwargs...)

""" MPO TESTS """

@testset "mpo-qpt: normalization" begin
  N = 10
  χ = 4
  U = randomprocess(N;χ=χ)
  Λ = PastaQ.makeChoi(U)
  @test length(Λ) == N
  logZ = lognorm(Λ.M.X)
  localZ = []
  normalize!(Λ; localnorms! = localZ)
  @test logZ ≈ sum(log.(localZ))
  @test norm(Λ.M.X) ≈ 1
end

@testset "mpo-qpt: grad logZ" begin
  N = 5
  χ = 4
  
  # 1. Unnormalized
  U = randomprocess(N; χ = χ)
  Λ = PastaQ.makeChoi(U)
  alg_grad,_ = PastaQ.gradlogZ(Λ)
  num_grad = numgradslogZ(Λ.M)
  
  alg_gradient = permutedims(array(alg_grad[1]),[1,3,2])
  @test alg_gradient ≈ num_grad[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grad[j]),[2,1,3,4])
    @test alg_gradient ≈ num_grad[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grad[N]),[2,1,3])
  @test alg_gradient ≈ num_grad[N] rtol=1e-3
  
  # 2. Globally normalized
  U = randomprocess(N; χ = χ)
  Λ = PastaQ.makeChoi(U)
  normalize!(Λ)
  @test norm(Λ.M.X)^2 ≈ 1
  alg_grad,_ = PastaQ.gradlogZ(Λ)
  num_grad = numgradslogZ(Λ.M)
  
  alg_gradient = permutedims(array(alg_grad[1]),[1,3,2])
  @test alg_gradient ≈ num_grad[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grad[j]),[2,1,3,4])
    @test alg_gradient ≈ num_grad[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grad[N]),[2,1,3])
  @test alg_gradient ≈ num_grad[N] rtol=1e-3

  # 3. Locally normalized
  U = randomprocess(N; χ = χ)
  Λ = PastaQ.makeChoi(U)
  num_grad = numgradslogZ(Λ.M)
  
  localnorms = []
  normalize!(Λ; localnorms! = localnorms)
  @test norm(Λ.M.X)^2 ≈ 1
  alg_grad,_ = PastaQ.gradlogZ(Λ; localnorms = localnorms)
  
  alg_gradient = permutedims(array(alg_grad[1]),[1,3,2])
  @test alg_gradient ≈ num_grad[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grad[j]),[2,1,3,4])
    @test alg_gradient ≈ num_grad[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grad[N]),[2,1,3])
  @test alg_gradient ≈ num_grad[N] rtol=1e-3
  
end

@testset "mpo-qpt: grad nll" begin
  N = 4
  χ = 2
  nsamples = 10
  Random.seed!(1234)
  data_in  = randompreparations(N,nsamples)
  data_out = randompreparations(N,nsamples)
  
  # 1. Unnnomalized
  U = randomprocess(N;χ=χ)
  Λ = PastaQ.makeChoi(U)
  num_grad = numgradsnll(Λ,data_in,data_out)
  alg_grad,_ = PastaQ.gradnll(Λ,data_in,data_out)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
  
  # 2. Globally normalized
  U = randomprocess(N;χ=χ)
  Λ = PastaQ.makeChoi(U)
  normalize!(Λ)
  num_grad = numgradsnll(Λ,data_in,data_out)
  alg_grad,_ = PastaQ.gradnll(Λ,data_in,data_out)
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
  
  # 3. Locally normalized
  U = randomprocess(N;χ=χ)
  Λ = PastaQ.makeChoi(U)
  num_grad = numgradsnll(Λ,data_in,data_out)
  localnorms = []
  normalize!(Λ; localnorms! = localnorms)
  alg_grad,_ = PastaQ.gradnll(Λ,data_in,data_out; localnorms = localnorms) 
  for j in 1:N
    @test array(alg_grad[j]) ≈ num_grad[j] rtol=1e-3
  end
end


@testset "mpo-qpt: full gradients" begin
  N = 3
  χ = 4
  nsamples = 10
  Random.seed!(1234)
  data_in  = randompreparations(N,nsamples)
  data_out = randompreparations(N,nsamples)
  
  # 1. Unnormalized
  U = randomprocess(N;χ=χ)
  Λ = PastaQ.makeChoi(U)
  logZ = 2.0*lognorm(Λ.M.X)
  NLL  = PastaQ.nll(Λ,data_in,data_out)
  ex_loss = logZ + NLL - N*log(2)
  num_gradZ = numgradslogZ(Λ.M)
  num_gradNLL = numgradsnll(Λ,data_in,data_out)
  num_grads = num_gradZ + num_gradNLL

  alg_grads,loss = PastaQ.gradients(Λ,data_in,data_out)
  @test ex_loss ≈ loss
  alg_gradient = permutedims(array(alg_grads[1]),[1,3,2])
  @test alg_gradient ≈ num_grads[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grads[j]),[2,1,3,4])
    @test alg_gradient ≈ num_grads[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grads[N]),[2,1,3])
  @test alg_gradient ≈ num_grads[N] rtol=1e-3
  

  ## 2. Globally normalized
  U = randomprocess(N;χ=χ)
  Λ = PastaQ.makeChoi(U)
  normalize!(Λ)
  NLL  = PastaQ.nll(Λ,data_in,data_out)
  ex_loss = NLL - N*log(2)
  num_gradZ = numgradslogZ(Λ.M)
  num_gradNLL = numgradsnll(Λ,data_in,data_out)
  num_grads = num_gradZ + num_gradNLL

  alg_grads,loss = PastaQ.gradients(Λ,data_in,data_out)
  @test ex_loss ≈ loss
  alg_gradient = permutedims(array(alg_grads[1]),[1,3,2])
  @test alg_gradient ≈ num_grads[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grads[j]),[2,1,3,4])
    @test alg_gradient ≈ num_grads[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grads[N]),[2,1,3])
  @test alg_gradient ≈ num_grads[N] rtol=1e-3
  
  # 3. Locally normalized
  U = randomprocess(N;χ=χ)
  Λ = PastaQ.makeChoi(U)
  num_gradZ = numgradslogZ(Λ.M)
  num_gradNLL = numgradsnll(Λ,data_in,data_out)
  num_grads = num_gradZ + num_gradNLL
  
  localnorms = []
  normalize!(Λ; localnorms! = localnorms)
  
  NLL  = PastaQ.nll(Λ,data_in,data_out)
  ex_loss = NLL - N*log(2)

  alg_grads,loss = PastaQ.gradients(Λ,data_in,data_out; localnorms = localnorms)
  @test ex_loss ≈ loss
  alg_gradient = permutedims(array(alg_grads[1]),[1,3,2])
  @test alg_gradient ≈ num_grads[1] rtol=1e-3
  for j in 2:N-1
    alg_gradient = permutedims(array(alg_grads[j]),[2,1,3,4])
    @test alg_gradient ≈ num_grads[j] rtol=1e-3
  end
  alg_gradient = permutedims(array(alg_grads[N]),[2,1,3])
  @test alg_gradient ≈ num_grads[N] rtol=1e-3
end





#""" CHOI TESTS """
#
##TODO This works with the plit version only
#@testset "lpdo-qpt: grad nll" begin
#  Nqubits= 3
#  N = 2*Nqubits
#  χ = 4
#  ξ = 3
#  nsamples = 10
#  Random.seed!(1234)
#  rawdata = rand(0:1,nsamples,N)
#  bases = randombases(N,nsamples)
#  data = Matrix{String}(undef, nsamples,N)
#  for n in 1:nsamples
#    data[n,:] = PastaQ.convertdatapoint(rawdata[n,:],bases[n,:],state=true)
#  end
#  
#  # 1. Unnormalized
#  #Λ = randomprocess(N;mixed=true,χ=χ,ξ=ξ)
#  Λ = splitstatewrapper(Nqubits,χ,ξ)
#  num_grad = numgradsnll(Λ,data,choi=true)
#  alg_grad,loss = PastaQ.gradnll(Λ,data,choi=true)
#  ex_loss = PastaQ.nll(Λ,data,choi=true)
#  @test ex_loss ≈ loss
#  alg_gradient = permutedims(array(alg_grad[1]),[3,1,2])
#  @test alg_gradient ≈ num_grad[1] rtol=1e-3
#  for j in 2:N-1
#    alg_gradient = permutedims(array(alg_grad[j]),[4,2,3,1])
#    @test alg_gradient ≈ num_grad[j] rtol=1e-3
#  end
#  alg_gradient = permutedims(array(alg_grad[N]),[3,1,2])
#  @test alg_gradient ≈ num_grad[N] rtol=1e-3
#  
#  # 2. Globally normalized
#  #Λ = randomprocess(N;mixed=true,χ=χ,ξ=ξ)
#  Λ = splitstatewrapper(Nqubits,χ,ξ)
#  normalize!(Λ)
#  num_grad = numgradsnll(Λ,data,choi=true)
#  ex_loss = PastaQ.nll(Λ,data,choi=true) 
#  alg_grad,loss = PastaQ.gradnll(Λ,data,choi=true)
#  @test ex_loss ≈ loss 
#  alg_gradient = permutedims(array(alg_grad[1]),[3,1,2])
#  @test alg_gradient ≈ num_grad[1] rtol=1e-3
#  for j in 2:N-1
#    alg_gradient = permutedims(array(alg_grad[j]),[4,2,3,1])
#    @test alg_gradient ≈ num_grad[j] rtol=1e-3
#  end
#  alg_gradient = permutedims(array(alg_grad[N]),[3,1,2])
#  @test alg_gradient ≈ num_grad[N] rtol=1e-3
#
#  # 3. Locally normalized
#  #Λ = randomprocess(N;mixed=true,χ=χ,ξ=ξ)
#  Λ = splitstatewrapper(Nqubits,χ,ξ)
#  num_grad = numgradsnll(Λ,data,choi=true)
#  sqrt_localnorms = []
#  normalize!(Λ; sqrt_localnorms! = sqrt_localnorms)
#  ex_loss = PastaQ.nll(Λ, data; choi = true) 
#  alg_grad,loss = PastaQ.gradnll(Λ, data; sqrt_localnorms = sqrt_localnorms, choi = true)
#  @test ex_loss ≈ loss
#  alg_gradient = permutedims(array(alg_grad[1]),[3,1,2])
#  @test alg_gradient ≈ num_grad[1] rtol=1e-3
#  for j in 2:N-1
#    alg_gradient = permutedims(array(alg_grad[j]),[4,2,3,1])
#    @test alg_gradient ≈ num_grad[j] rtol=1e-3
#  end
#  alg_gradient = permutedims(array(alg_grad[N]),[3,1,2])
#  @test alg_gradient ≈ num_grad[N] rtol=1e-3
#  
#end
#
