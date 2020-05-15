using LinearAlgebra
using ITensors
using Random

struct QPT
  #Kp::Int            # Number of input states 
  prep::String        # name tag for preparation states
  P::Vector{ITensor}  # Vector of preparation states
  #Km::Int            # Numbe of measurement POVMs
  povm::String        # name tag for measurement povms
  M::Vector{ITensor}  # Vector of POVM measurements
  N::Int              # Number of qubits
  chi::Int            # Bond dimension
  mpo::MPO            # Matrix product operator
  seed::Int           # Seed for RNG
end

function QPT(;prep::String,povm::String,N::Int,chi::Int,seed::Int)
  Random.seed!(seed);

  # Set the input states
  if prep == "Pauli6"
    #Kp = 6
    P = ITensor[]
    i = Index(2)
    mat = [1 0; 0 0]
    push!(P,ITensor(mat,i,i'))
    mat = [0 0; 0 1]
    push!(P,ITensor(mat,i,i'))
    mat = 1/2. * [1 1; 1 1]
    push!(P,ITensor(mat,i,i'))
    mat = 1/2. * [1 -1; -1 1]
    push!(P,ITensor(mat,i,i'))
    mat = 1/2. * [1 1im; -1im 1]
    push!(P,ITensor(mat,i,i'))
    mat = 1/2. * [1 -1im; 1im 1]
    push!(P,ITensor(mat,i,i'))
  else
    error("Only Pauli6 preparation is implemented")
  end
  
  # Set the POVM elements
  if povm == "Pauli6"
    #Km = 6
    M = ITensor[]
    i = Index(2)
    mat = 1/3. * [1 0; 0 0]
    push!(M,ITensor(mat,i,i'))
    mat = 1/3. * [0 0; 0 1]
    push!(M,ITensor(mat,i,i'))
    mat = 1/6. * [1 1; 1 1]
    push!(M,ITensor(mat,i,i'))
    mat = 1/6. * [1 -1; -1 1]
    push!(M,ITensor(mat,i,i'))
    mat = 1/6. * [1 -1im; 1im 1]
    push!(M,ITensor(mat,i,i'))
    mat = 1/6. * [1 1im; -1im 1]
    push!(M,ITensor(mat,i,i'))
    @assert sum(M) ≈ delta(i,i') atol=1e-8
  else
    error("Only Pauli6 POVMs are implemented")
  end
   
  # Initialize the MPO randomly
  d = 2 # qubits
  link  = [Index(chi; tags="link, l=$l") for l in 1:N-1]
  s_i   = [Index(d; tags="site_in, s_i=$s") for s in 1:N]
  s_o   = [Index(d; tags="site_out, s_o=$s") for s in 1:N]
  A = ITensor[]
  sigma = 0.1
  rand_mat = sigma*(ones(d,d,chi)-rand(d,d,chi))+im*sigma*(ones(d,d,chi)-rand(d,d,chi))
  push!(A,ITensor(rand_mat,s_i[1],s_o[1],link[1]))
  for j in 2:N-1
    rand_mat = sigma*(ones(chi,d,d,chi)-rand(chi,d,d,chi))+im*sigma*(ones(chi,d,d,chi)-rand(chi,d,d,chi))
    push!(A,ITensor(rand_mat,link[j-1],s_i[j],s_o[j],link[j]))
  end
  rand_mat = sigma*(ones(chi,d,d)-rand(chi,d,d))+im*sigma*(ones(chi,d,d)-rand(chi,d,d))
  push!(A,ITensor(rand_mat,link[N-1],s_i[N],s_o[N]))
  mpo = MPO(A)

  return QPT(prep,P,povm,M,N,chi,mpo,seed)

end

" LOAD THE QPT MPO "
# Set the MPO tensor given input array
function SetMPO(qpt::QPT,mpo_list)
  @assert qpt.N == length(mpo_list)
  for j in 1:qpt.N
    qpt.mpo[j] = ITensor(mpo_list[j],inds(qpt.mpo[j]))
  end
end

" LOAD THE TARGET MPO "
function SetTargetMPO(qpt::QPT,mpo_list)
  @assert qpt.N == length(mpo_list)
  target_mpo = ITensor[]
  for j in 1:qpt.N
    push!(target_mpo,ITensor(mpo_list[j],inds(qpt.mpo[j])))
  end
  return MPO(target_mpo)
end

function QuantumProcessFidelity(qpt::QPT,target::MPO)
  normalization = Normalization(qpt.mpo,choi=true)
  fidelity = dag(qpt.mpo[1]) * prime(target[1],"link");
  for j in 2:qpt.N-1
      fidelity = fidelity* dag(qpt.mpo[j]);
      fidelity = fidelity * prime(target[j],"link")
  end
  fidelity = fidelity * dag(qpt.mpo[qpt.N]);
  fidelity = fidelity * prime(target[qpt.N],"link")
  fidelity = abs2(fidelity[])/normalization
  fidelity = fidelity / (2^(2*N))
  return fidelity
end

" NORMALIZATION "
function Normalization(mpo::MPO;choi::Bool=false)
  # Site 1
  norm = dag(mpo[1]) * prime(mpo[1],"link");
  for j in 2:qpt.N-1
      norm = norm * dag(mpo[j]);
      norm = norm * prime(mpo[j],"link")
  end
  norm = norm * dag(mpo[qpt.N]);
  norm = norm * prime(mpo[qpt.N],"link")
  if choi
    norm = real(norm[]) / 2^length(mpo)
  else
    norm = real(norm[])
  end
  return norm
end;

" LOSS "
function Loss(qpt::QPT,batch)
    
  cost = 0.0
  for n in 1:size(batch)[1]
    sample = batch[n,:]
    replaceinds!(qpt.P[sample[1]],inds(qpt.P[sample[1]]),[inds(qpt.mpo[1])[1],inds(qpt.mpo[1])[1]'])
    replaceinds!(qpt.M[sample[qpt.N+1]],inds(qpt.M[sample[qpt.N+1]]),[inds(qpt.mpo[1])[2],inds(qpt.mpo[1])[2]'])
    prob = dag(qpt.mpo[1]) * qpt.P[sample[1]] * qpt.M[sample[qpt.N+1]]
    prob = prob * prime(qpt.mpo[1])
    for j in 2:qpt.N-1
      replaceinds!(qpt.P[sample[j]],inds(qpt.P[sample[j]]),[inds(qpt.mpo[j])[2],inds(qpt.mpo[j])[2]'])
      replaceinds!(qpt.M[sample[qpt.N+j]],inds(qpt.M[sample[qpt.N+j]]),[inds(qpt.mpo[j])[3],inds(qpt.mpo[j])[3]'])
      prob = prob * dag(qpt.mpo[j]) * qpt.P[sample[j]] * qpt.M[sample[qpt.N+j]]
      prob = prob * prime(qpt.mpo[j])
    end
    replaceinds!(qpt.P[sample[qpt.N]],inds(qpt.P[sample[qpt.N]]),[inds(qpt.mpo[qpt.N])[2],inds(qpt.mpo[qpt.N])[2]'])
    replaceinds!(qpt.M[sample[2*qpt.N]],inds(qpt.M[sample[2*qpt.N]]),[inds(qpt.mpo[qpt.N])[3],inds(qpt.mpo[qpt.N])[3]'])
    prob = prob * dag(qpt.mpo[qpt.N]) * qpt.P[sample[qpt.N]] * qpt.M[sample[2*qpt.N]]
    prob = prob * prime(qpt.mpo[qpt.N])
    cost -= log(real(prob[]))
  end
  return cost/size(batch)[1]
end;

" GRADIENTS OF LOG_Z "
function GradientLogZ(qpt::QPT)

  R = ITensor[]
  L = ITensor[]
  
  # Sweep right to get L
  tensor =  qpt.mpo[1] * prime(dag(qpt.mpo[1]),"link")
  push!(L,tensor)
  for j in 2:qpt.N-1
      tensor = L[j-1] * qpt.mpo[j];
      tensor = tensor * prime(dag(qpt.mpo[j]),"link") 
      push!(L,tensor)
  end
  
  Z = L[qpt.N-1] * qpt.mpo[qpt.N]
  Z = Z * prime(dag(qpt.mpo[qpt.N]),"link")
  Z = real(Z[])

  # Sweep left to get R
  tensor = qpt.mpo[qpt.N] * prime(dag(qpt.mpo[qpt.N]),"link")
  push!(R,tensor)
  #for j in reverse(2:N-1)
  for j in 2:N-1
      tensor = R[j-1] * qpt.mpo[qpt.N+1-j]
      tensor = tensor * prime(dag(qpt.mpo[qpt.N+1-j]),"link") 
      push!(R,tensor)
  end
  
  # Get the gradients of the normalization
  gradients = ITensor[]
  
  # Site 1
  tensor = qpt.mpo[1] * R[qpt.N-1]
  push!(gradients,noprime(tensor)/Z)
  
  for j in 2:qpt.N-1
      tensor = L[j-1] * qpt.mpo[j] * R[qpt.N-j]
      push!(gradients,noprime(tensor)/Z)
  end
  tensor = L[qpt.N-1] * qpt.mpo[qpt.N]
  push!(gradients,noprime(tensor)/Z)
   
  return 2*gradients

end;

" GRADIENTS OF KL "
function GradientKL(qpt::QPT,batch)
  gradients = ITensor[]
  for j in 1:qpt.N
    push!(gradients,ITensor(inds(qpt.mpo[j])))
  end
  
  for n in 1:size(batch)[1]
    sample = batch[n,:]
    R = ITensor[]
    L = ITensor[]

    # Sweep right to get L
    # Site 1
    replaceinds!(qpt.P[sample[1]],inds(qpt.P[sample[1]]),[inds(qpt.mpo[1])[1],inds(qpt.mpo[1])[1]'])
    replaceinds!(qpt.M[sample[qpt.N+1]],inds(qpt.M[sample[qpt.N+1]]),[inds(qpt.mpo[1])[2],inds(qpt.mpo[1])[2]'])
    tensor = dag(qpt.mpo[1]) * qpt.P[sample[1]] * qpt.M[sample[qpt.N+1]]
    tensor = tensor * prime(qpt.mpo[1])
    push!(L,tensor)
    # Site: 2,...,N-1
    for j in 2:qpt.N-1
      replaceinds!(qpt.P[sample[j]],inds(qpt.P[sample[j]]),[inds(qpt.mpo[j])[2],inds(qpt.mpo[j])[2]'])
      replaceinds!(qpt.M[sample[qpt.N+j]],inds(qpt.M[sample[qpt.N+j]]),[inds(qpt.mpo[j])[3],inds(qpt.mpo[j])[3]'])
      tensor = tensor * dag(qpt.mpo[j]) * qpt.P[sample[j]] * qpt.M[sample[qpt.N+j]]
      tensor = tensor * prime(qpt.mpo[j])
      push!(L,tensor) 
    end
    replaceinds!(qpt.P[sample[qpt.N]],inds(qpt.P[sample[qpt.N]]),[inds(qpt.mpo[qpt.N])[2],inds(qpt.mpo[qpt.N])[2]'])
    replaceinds!(qpt.M[sample[2*qpt.N]],inds(qpt.M[sample[2*qpt.N]]),[inds(qpt.mpo[qpt.N])[3],inds(qpt.mpo[qpt.N])[3]'])
    tensor = tensor * dag(qpt.mpo[qpt.N]) * qpt.P[sample[qpt.N]] * qpt.M[sample[2*qpt.N]]
    tensor = tensor * prime(qpt.mpo[qpt.N])
    prob = real(tensor[])

    # Sweep left to get R
    # Site 1
    replaceinds!(qpt.P[sample[qpt.N]],inds(qpt.P[sample[qpt.N]]),[inds(qpt.mpo[qpt.N])[2],inds(qpt.mpo[qpt.N])[2]'])
    replaceinds!(qpt.M[sample[2*qpt.N]],inds(qpt.M[sample[2*qpt.N]]),[inds(qpt.mpo[qpt.N])[3],inds(qpt.mpo[qpt.N])[3]'])
    tensor = dag(qpt.mpo[qpt.N]) * qpt.P[sample[qpt.N]] * qpt.M[sample[2*qpt.N]]
    tensor = tensor * prime(qpt.mpo[qpt.N])
    push!(R,tensor)
    # Site: 2,...,N-1
    for j in 2:qpt.N-1
      replaceinds!(qpt.P[sample[qpt.N+1-j]],inds(qpt.P[sample[qpt.N+1-j]]),[inds(qpt.mpo[qpt.N+1-j])[2],inds(qpt.mpo[qpt.N+1-j])[2]'])
      replaceinds!(qpt.M[sample[2*qpt.N+1-j]],inds(qpt.M[sample[2*qpt.N+1-j]]),[inds(qpt.mpo[qpt.N+1-j])[3],inds(qpt.mpo[qpt.N+1-j])[3]'])
      tensor = tensor * dag(qpt.mpo[qpt.N+1-j]) * qpt.P[sample[qpt.N+1-j]] * qpt.M[sample[2*qpt.N+1-j]]
      tensor = tensor * prime(qpt.mpo[qpt.N+1-j])
      push!(R,tensor)
    end

    #Site 1
    replaceinds!(qpt.P[sample[1]],inds(qpt.P[sample[1]]),[inds(qpt.mpo[1])[1],inds(qpt.mpo[1])[1]'])
    replaceinds!(qpt.M[sample[N+1]],inds(qpt.M[sample[qpt.N+1]]),[inds(qpt.mpo[1])[2],inds(qpt.mpo[1])[2]'])
    tensor = qpt.P[sample[1]] * qpt.M[sample[qpt.N+1]] * prime(qpt.mpo[1])
    tensor = tensor * R[qpt.N-1]
    gradients[1] += noprime(tensor)/prob
    for j in 2:qpt.N-1
      replaceinds!(qpt.P[sample[j]],inds(qpt.P[sample[j]]),[inds(qpt.mpo[j])[2],inds(qpt.mpo[j])[2]'])
      replaceinds!(qpt.M[sample[qpt.N+j]],inds(qpt.M[sample[qpt.N+j]]),[inds(qpt.mpo[j])[3],inds(qpt.mpo[j])[3]'])
      tensor = qpt.P[sample[j]] * qpt.M[sample[qpt.N+j]] * prime(qpt.mpo[j])
      tensor = L[j-1] * tensor * R[qpt.N-j]
      gradients[j] += noprime(tensor)/prob
    end
    replaceinds!(qpt.P[sample[qpt.N]],inds(qpt.P[sample[qpt.N]]),[inds(qpt.mpo[qpt.N])[2],inds(qpt.mpo[qpt.N])[2]'])
    replaceinds!(qpt.M[sample[2*qpt.N]],inds(qpt.M[sample[2*qpt.N]]),[inds(qpt.mpo[qpt.N])[3],inds(qpt.mpo[qpt.N])[3]'])
    tensor = qpt.P[sample[qpt.N]] * qpt.M[sample[2*qpt.N]] * prime(qpt.mpo[qpt.N])
    tensor = L[qpt.N-1] * tensor
    gradients[qpt.N] += noprime(tensor)/prob
  end
  return -2*gradients/size(batch)[1]

end;
