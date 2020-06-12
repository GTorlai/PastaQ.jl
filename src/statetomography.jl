"""
Initializer for MPS state tomography
"""
function initializeQST(N::Int,χ::Int;d::Int=2,seed::Int=1234,σ::Float64=0.1)
  rng = MersenneTwister(seed)
  
  sites = [Index(d; tags="Site, n=$s") for s in 1:N]
  links = [Index(χ; tags="Link, l=$l") for l in 1:N-1]

  M = ITensor[]
  # Site 1 
  rand_mat = σ * (ones(d,χ) - 2*rand!(rng,zeros(d,χ)))
  rand_mat += im * σ * (ones(d,χ) - 2*rand!(rng,zeros(d,χ)))
  push!(M,ITensor(rand_mat,sites[1],links[1]))
  # Site 2..N-1
  for j in 2:N-1
    rand_mat = σ * (ones(χ,d,χ) - 2*rand!(rng,zeros(χ,d,χ)))
    rand_mat += im * σ * (ones(χ,d,χ) - 2*rand!(rng,zeros(χ,d,χ)))
    push!(M,ITensor(rand_mat,links[j-1],sites[j],links[j]))
  end
  # Site N
  rand_mat = σ * (ones(χ,d) - 2*rand!(rng,zeros(χ,d)))
  rand_mat += im * σ * (ones(χ,d) - 2*rand!(rng,zeros(χ,d)))
  push!(M,ITensor(rand_mat,links[N-1],sites[N]))
  
  psi = MPS(M)
  return psi
end

"""
Initializer for LPDO state tomography
"""
function initializeQST(N::Int,χ::Int,ξ::Int;d::Int=2,seed::Int=1234,σ::Float64=0.1)
  rng = MersenneTwister(seed)
  
  sites = [Index(d; tags="Site,n=$s") for s in 1:N]
  links = [Index(χ; tags="Link,l=$l") for l in 1:N-1]
  kraus = [Index(ξ; tags="Kraus,k=$s") for s in 1:N]

  M = ITensor[]
  # Site 1 
  rand_mat = σ * (ones(d,χ,ξ) - 2*rand!(rng,zeros(d,χ,ξ)))
  rand_mat += im * σ * (ones(d,χ,ξ) - 2*rand!(rng,zeros(d,χ,ξ)))
  push!(M,ITensor(rand_mat,sites[1],links[1],kraus[1]))
  # Site 2..N-1
  for j in 2:N-1
    rand_mat = σ * (ones(d,χ,ξ,χ) - 2*rand!(rng,zeros(d,χ,ξ,χ)))
    rand_mat += im * σ * (ones(d,χ,ξ,χ) - 2*rand!(rng,zeros(d,χ,ξ,χ)))
    push!(M,ITensor(rand_mat,sites[j],links[j-1],kraus[j],links[j]))
  end
  # Site N
  rand_mat = σ * (ones(d,χ,ξ) - 2*rand!(rng,zeros(d,χ,ξ)))
  rand_mat += im * σ * (ones(d,χ,ξ) - 2*rand!(rng,zeros(d,χ,ξ)))
  push!(M,ITensor(rand_mat,sites[N],links[N-1],kraus[N]))
  
  lpdo = MPO(M)
  return lpdo
end

"""
Normalize the MPS/LPDO locally and store the local norms
"""
function lognormalize!(M::Union{MPS,MPO})
  localnorms = []
  blob = dag(M[1]) * prime(M[1],"Link")
  localZ = norm(blob)
  logZ = 0.5*log(localZ)
  blob /= sqrt(localZ)
  M[1] /= (localZ^0.25)
  push!(localnorms,localZ^0.25)
  for j in 2:length(M)-1
    blob = blob * dag(M[j]);
    blob = blob * prime(M[j],"Link")
    localZ = norm(blob)
    logZ += 0.5*log(localZ)
    blob /= sqrt(localZ)
    M[j] /= (localZ^0.25)
    push!(localnorms,localZ^0.25)  
  end
  blob = blob * dag(M[length(M)]);
  blob = blob * prime(M[length(M)],"Link")
  localZ = norm(blob)
  M[length(M)] /= sqrt(localZ)
  push!(localnorms,sqrt(localZ))
  logZ += log(real(blob[]))
  return logZ,localnorms
end

"""
Gradients of logZ for MPS/LPDO
"""
function gradlogZ(M::Union{MPS,MPO};localnorm=nothing)
  N = length(M)
  L = Vector{ITensor}(undef, N-1)
  R = Vector{ITensor}(undef, N)
  
  if isnothing(localnorm)
    localnorm = ones(N)
  end
  
  # Sweep right to get L
  L[1] = dag(M[1]) * prime(M[1],"Link")
  for j in 2:N-1
    L[j] = L[j-1] * dag(M[j])
    L[j] = L[j] * prime(M[j],"Link")
  end
  Z = L[N-1] * dag(M[N])
  Z = real((Z * prime(M[N],"Link"))[])

  # Sweep left to get R
  R[N] = dag(M[N]) * prime(M[N],"Link")
  for j in reverse(2:N-1)
    R[j] = R[j+1] * dag(M[j])
    R[j] = R[j] * prime(M[j],"Link")
  end
  # Get the gradients of the normalization
  gradients = Vector{ITensor}(undef, N)
  gradients[1] = prime(M[1],"Link") * R[2]/(localnorm[1]*Z)
  for j in 2:N-1
    gradients[j] = (L[j-1] * prime(M[j],"Link") * R[j+1])/(localnorm[j]*Z)
  end
  gradients[N] = (L[N-1] * prime(M[N],"Link"))/(localnorm[N]*Z)
  return 2*gradients,log(Z)
end

"""
Gradients of NLL for MPS 
"""
function gradnll(psi::MPS,data::Array;localnorm=nothing)
  loss = 0.0

  N = length(psi)

  s = siteinds(psi)

  links = [linkind(psi, n) for n in 1:N-1]

  ElT = eltype(psi[1])

  L = Vector{ITensor{1}}(undef, N)
  Lpsi = Vector{ITensor}(undef, N)
  for n in 1:N-1
    L[n] = ITensor(ElT, undef, links[n])
    Lpsi[n] = ITensor(ElT, undef, s[n], links[n])
  end
  Lpsi[N] = ITensor(ElT, undef, s[N])

  R = Vector{ITensor{1}}(undef, N)
  Rpsi = Vector{ITensor}(undef, N)
  for n in N:-1:2
    R[n] = ITensor(ElT, undef, links[n-1])
    Rpsi[n] = ITensor(ElT, undef, links[n-1], s[n])
  end
  Rpsi[1] = ITensor(ElT, undef, s[1])

  if isnothing(localnorm)
    localnorm = ones(N)
  end

  psidag = dag(psi)

  gradients = [ITensor(ElT, inds(psi[j])) for j in 1:N]

  grads = [ITensor(ElT, undef, inds(psi[j])) for j in 1:N]

  for n in 1:size(data)[1]
    x = data[n,:] 
    
    """ LEFT ENVIRONMENTS """
    L[1] .= psidag[1] .* measproj(x[1],s[1])
    for j in 2:N-1
      Lpsi[j] .= L[j-1] .* psidag[j]
      L[j] .= Lpsi[j] .* measproj(x[j],s[j])
    end
    Lpsi[N] .= L[N-1] .* psidag[N]
    psix = (Lpsi[N] * measproj(x[N],s[N]))[]
    prob = abs2(psix)
    loss -= log(prob)/size(data)[1]
    
    """ RIGHT ENVIRONMENTS """
    R[N] .= psidag[N] .* measproj(x[N],s[N])
    for j in reverse(2:N-1)
      Rpsi[j] .= psidag[j] .* R[j+1]
      R[j] .= Rpsi[j] .* measproj(x[j],s[j])
    end

    """ GRADIENTS """
    grads[1] .= measproj(x[1],s[1]) .* R[2] 
    gradients[1] .+= grads[1] ./ (localnorm[1] * psix)
    for j in 2:N-1
      Rpsi[j] .= L[j-1] .* measproj(x[j],s[j])
      grads[j] .= Rpsi[j] .* R[j+1]
      gradients[j] .+= grads[j] ./ (localnorm[j] * psix)
    end
    grads[N] .= L[N-1] .* measproj(x[N],s[N])
    gradients[N] .+= grads[N] ./ (localnorm[N] * psix)
  end
  for g in gradients
    g .= -2/size(data)[1] .* g
  end
  return gradients,loss 
end

"""
Gradients of NLL for LPDO 
"""
function gradnll(lpdo::MPO,data::Array;localnorm=nothing)
  loss = 0.0

  N = length(lpdo)
  
  s = Index[]
  for j in 1:N
    push!(s,firstind(lpdo[j],"Site"))
  end

  links = [linkind(lpdo, n) for n in 1:N-1]
  
  kraus = Index[]
  for j in 1:N
    push!(kraus,firstind(lpdo[j],"Kraus"))
  end

  ElT = eltype(lpdo[1])

  L = Vector{ITensor{2}}(undef, N)
  Llpdo = Vector{ITensor}(undef, N)
  Lgrad = Vector{ITensor}(undef,N)
  for n in 1:N-1
    L[n] = ITensor(ElT, undef, links[n]',links[n])
  end
  for n in 2:N-1
    Llpdo[n] = ITensor(ElT, undef, kraus[n],links[n]',links[n-1])
  end
  for n in 1:N-2
    Lgrad[n] = ITensor(ElT,undef,links[n],kraus[n+1],links[n+1]')
  end
  Lgrad[N-1] = ITensor(ElT,undef,links[N-1],kraus[N])

  R = Vector{ITensor{2}}(undef, N)
  Rlpdo = Vector{ITensor}(undef, N)
  for n in N:-1:2
    R[n] = ITensor(ElT, undef, links[n-1]',links[n-1])
  end 
  for n in N-1:-1:2
    Rlpdo[n] = ITensor(ElT, undef, links[n-1]',kraus[n],links[n])
  end
  
  Agrad = Vector{ITensor}(undef, N)
  Agrad[1] = ITensor(ElT, undef, kraus[1],links[1]',s[1])
  for n in 2:N-1
    Agrad[n] = ITensor(ElT, undef, links[n-1],kraus[n],links[n]',s[n])
  end

  T = Vector{ITensor}(undef,N)
  Tp = Vector{ITensor}(undef,N)
  T[1] = ITensor(ElT, undef, kraus[1],links[1])
  Tp[1] = prime(T[1],"Link")
  for n in 2:N-1
    T[n] = ITensor(ElT, undef, kraus[n],links[n],links[n-1])
    Tp[n] = prime(T[n],"Link")
  end
  T[N] = ITensor(ElT, undef, kraus[N],links[N-1])
  Tp[N] = prime(T[N],"Link")
  
  if isnothing(localnorm)
    localnorm = ones(N)
  end

  grads = Vector{ITensor}(undef,N)
  gradients = Vector{ITensor}(undef,N)
  grads[1] = ITensor(ElT, undef,links[1],kraus[1],s[1])
  gradients[1] = ITensor(ElT,links[1],kraus[1],s[1])
  for n in 2:N-1
    grads[n] = ITensor(ElT, undef,links[n],links[n-1],kraus[n],s[n])
    gradients[n] = ITensor(ElT,links[n],links[n-1],kraus[n],s[n])
  end
  grads[N] = ITensor(ElT, undef,links[N-1],kraus[N],s[N])
  gradients[N] = ITensor(ElT, links[N-1],kraus[N],s[N])
  
  for n in 1:size(data)[1]
    x = data[n,:]
    """ LEFT ENVIRONMENTS """
    T[1] .= lpdo[1] .* dag(measproj(x[1],s[1]))
    L[1] .= prime(T[1],"Link") .* dag(T[1])
    for j in 2:N-1
      T[j] .= lpdo[j] .* dag(measproj(x[j],s[j]))
      Llpdo[j] .= prime(T[j],"Link") .* L[j-1]
      L[j] .= Llpdo[j] .* dag(T[j])
    end
    T[N] .= lpdo[N] .* dag(measproj(x[N],s[N]))
    prob = L[N-1] * prime(T[N],"Link")
    prob = prob * dag(T[N])
    prob = real(prob[])
    loss -= log(prob)/size(data)[1]
    
    """ RIGHT ENVIRONMENTS """
    R[N] .= prime(T[N],"Link") .* dag(T[N])
    for j in reverse(2:N-1)
      Rlpdo[j] .= prime(T[j],"Link") .* R[j+1] 
      R[j] .= Rlpdo[j] .* dag(T[j])
    end
    
    """ GRADIENTS """
    Tp[1] .= prime(lpdo[1],"Link") .* dag(measproj(x[1],s[1]))
    Agrad[1] .=  Tp[1] .* measproj(x[1],s[1])
    grads[1] .= R[2] .* Agrad[1]
    gradients[1] .+= grads[1] ./ (localnorm[1]*prob)
    for j in 2:N-1
      Tp[j] .= prime(lpdo[j],"Link") .* dag(measproj(x[j],s[j]))
      Lgrad[j-1] .= L[j-1] .* Tp[j]
      Agrad[j] .= Lgrad[j-1] .* measproj(x[j],s[j])
      grads[j] .= R[j+1] .* Agrad[j] 
      gradients[j] .+= grads[j] ./ (localnorm[j]*prob)
    end
    Tp[N] .= prime(lpdo[N],"Link") .* dag(measproj(x[N],s[N]))
    Lgrad[N-1] .= L[N-1] .* Tp[N]
    grads[N] .= Lgrad[N-1] .* measproj(x[N],s[N])
    gradients[N] .+= grads[N] ./ (localnorm[N]*prob)
  end
  for g in gradients
    g .= -2/size(data)[1] .* g
  end
  return gradients,loss 
end

"""
Compute the total gradients
"""
function gradients(M::Union{MPS,MPO},data::Array;localnorm=nothing)
  g_logZ,logZ = gradlogZ(M,localnorm=localnorm)
  g_nll, nll  = gradnll(M,data,localnorm=localnorm)
  grads = g_logZ + g_nll
  loss = logZ + nll
  return grads,loss
end

"""
Run QST
"""
function statetomography(model::Union{MPS,MPO},
                         opt::Optimizer;
                         data::Array,
                         batchsize::Int64=500,
                         epochs::Int64=10000,
                         target::MPS,
                         localnorm::Bool=false)
  for j in 1:length(model)
    replaceind!(target[j],firstind(target[j],"Site"),firstind(model[j],"Site"))
  end
  num_batches = Int(floor(size(data)[1]/batchsize))
  
  for ep in 1:epochs
    data = data[shuffle(1:end),:]
    
    avg_loss = 0.0
    for b in 1:num_batches
      batch = data[(b-1)*batchsize+1:b*batchsize,:]
      
      if localnorm == true
        model_norm = copy(model)
        logZ,localnorms = lognormalize!(model_norm) 
        grads,loss = gradients(model_norm,batch,localnorm=localnorms)
      else
        grads,loss = gradients(model,batch)
      end
      avg_loss += loss/Float64(num_batches)
      update!(model,grads,opt)
    end
    F = fidelity(model,target)
    print("Ep = ",ep,"  ")
    print("Loss = ")
    @printf("%.5E",avg_loss)
    print("  ")
    print("Fidelity = ")
    @printf("%.3E",F)
    print("\n")
  end
end


""" UTILITY FUNCTIONS """

"""
Contract the Kraus indices to get the density matrix MPO
"""
function getdensityoperator(lpdo::MPO)
  noprime!(lpdo)
  N = length(lpdo)
  M = ITensor[]
  prime!(lpdo[1],tags="Site")
  prime!(lpdo[1],tags="Link")
  tmp = noprime(dag(lpdo[1])) * lpdo[1]
  Cdn = combiner(inds(tmp,tags="Link"),tags="Link,l=1")
  push!(M,tmp * Cdn)
  
  for j in 2:N-1
    prime!(lpdo[j],tags="Site")
    prime!(lpdo[j],tags="Link")
    tmp = noprime(dag(lpdo[j])) * lpdo[j] 
    Cup = Cdn
    Cdn = combiner(inds(tmp,tags="Link,l=$j"),tags="Link,l=$j")
    push!(M,tmp * Cup * Cdn)
  end
  prime!(lpdo[N],tags="Site")
  prime!(lpdo[N],tags="Link")
  tmp = noprime(dag(lpdo[N])) * lpdo[N]
  Cup = Cdn
  push!(M,tmp * Cdn)
  rho = MPO(M)
  noprime!(lpdo)
  return rho
end

function fidelity(psi::MPS,target::MPS)
  psi_eval = copy(psi)
  lognormalize!(psi_eval)
  @assert norm(psi_eval) ≈ 1
  fidelity = abs2(inner(psi_eval,target))
  return fidelity
end

function fidelity(lpdo::MPO,target::MPS)
  #for j in 1:length(lpdo)
  #  replaceind!(target[j],firstind(target[j],"Site"),firstind(lpdo[j],"Site"))
  #end
  lpdo_eval = copy(lpdo)
  lognormalize!(lpdo_eval)
  @assert norm(lpdo_eval) ≈ 1
  A = lpdo_eval * target
  fidelity = abs(inner(A,A))
  return fidelity
end

"""
Negative log likelihood for MPS
"""
function nll(psi::MPS,data::Array)
  N = length(psi)
  loss = 0.0
  s = siteinds(psi)
  for n in 1:size(data)[1]
    x = data[n,:]
    psix = dag(psi[1]) * measproj(x[1],s[1])
    for j in 2:N
      psi_r = dag(psi[j]) * measproj(x[j],s[j])
      psix = psix * psi_r
    end
    prob = abs2(psix[])
    loss -= log(prob)/size(data)[1]
  end
  return loss
end

"""
Negative log likelihood for LPDO
"""
function nll(lpdo::MPO,data::Array)
  N = length(lpdo)
  loss = 0.0
  s = Index[]
  for j in 1:N
    push!(s,firstind(lpdo[j],"Site"))
  end
  for n in 1:size(data)[1]
    x = data[n,:]
    prob = prime(lpdo[1],"Link") * dag(measproj(x[1],s[1]))
    prob = prob * dag(lpdo[1]) * measproj(x[1],s[1])
    for j in 2:N
      prob = prob * prime(lpdo[j],"Link") * dag(measproj(x[j],s[j]))
      prob = prob * dag(lpdo[j]) * measproj(x[j],s[j])
    end
    loss -= log(real(prob[]))/size(data)[1]
  end
  return loss
end

