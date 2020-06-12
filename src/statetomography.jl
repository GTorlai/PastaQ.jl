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
#function gradnll(psi::MPS,data::Array,bases::Array;localnorm=nothing)
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

#"""
#Gradients of NLL for LPDO 
#"""
#function gradnll(lpdo::MPO,data::Array,bases::Array;localnorm=nothing)
#  loss = 0.0
#
#  N = length(lpdo)
#  L = Vector{ITensor}(undef, N-1)
#  R = Vector{ITensor}(undef, N)
#  
#  if isnothing(localnorm)
#    localnorm = ones(N)
#  end
#  
#  gradients = ITensor[]
#  for j in 1:N
#    push!(gradients,ITensor(inds(lpdo[j])))
#  end
#  for n in 1:size(data)[1]
#    x = data[n,:] 
#    x.+=1
#    basis = bases[n,:]
#    
#    """ LEFT ENVIRONMENTS """
#    if (basis[1] == "Z")
#      T = lpdo[1] * setelt(firstind(lpdo[1],tags="Site")=>x[1])
#    else
#      rotation = makegate(lpdo,"m$(basis[1])",1)
#      T = lpdo[1] * rotation * prime(setelt(firstind(lpdo[1],tags="Site")=>x[1]))
#    end
#    L[1] = prime(T,"Link") * dag(T)
#    for j in 2:N-1
#      if (basis[j] == "Z")
#        T = lpdo[j] * setelt(firstind(lpdo[j],tags="Site")=>x[j])
#      else
#        rotation = makegate(lpdo,"m$(basis[j])",j)
#        T = lpdo[j] * rotation * prime(setelt(firstind(lpdo[j],tags="Site")=>x[j]))
#      end
#      L[j] = L[j-1] * prime(T,"Link")
#      L[j] = L[j] * dag(T)
#    end
#    if (basis[N] == "Z")
#      T = lpdo[N] * setelt(firstind(lpdo[N],tags="Site")=>x[N])
#    else
#      rotation = makegate(lpdo,"m$(basis[N])",N)
#      T = lpdo[N] * rotation * prime(setelt(firstind(lpdo[N],tags="Site")=>x[N]))
#    end
#    prob = L[N-1] * prime(T,"Link")
#    prob = prob * dag(T)
#    prob = real(prob[])
#    loss -= log(prob)/size(data)[1]
#    
#    """ RIGHT ENVIRONMENTS """
#    if (basis[N] == "Z")
#      T = lpdo[N] * setelt(firstind(lpdo[N],tags="Site")=>x[N])
#    else
#      rotation = makegate(lpdo,"m$(basis[N])",N)
#      T = lpdo[N] * rotation * prime(setelt(firstind(lpdo[N],tags="Site")=>x[N]))
#    end
#    R[N] = prime(T,"Link") * dag(T)
#    for j in reverse(2:N-1)
#      if (basis[j] == "Z")
#        T = lpdo[j] * setelt(firstind(lpdo[j],tags="Site")=>x[j])
#      else
#        rotation = makegate(lpdo,"m$(basis[j])",j)
#        T = lpdo[j] * rotation * prime(setelt(firstind(lpdo[j],tags="Site")=>x[j]))
#      end
#      R[j] = R[j+1] * prime(T,"Link")
#      R[j] = R[j] * dag(T)
#    end
#
#    """ GRADIENTS """
#    if (basis[1] == "Z")
#      Tup = prime(lpdo[1],"Link") * setelt(firstind(lpdo[1],tags="Site")=>x[1])
#      gradients[1] += (Tup * R[2] * setelt(firstind(lpdo[1],tags="Site")=>x[1]))/(localnorm[1]*prob)
#    else
#      rotation = makegate(lpdo,"m$(basis[1])",1)
#      Tup = prime(lpdo[1],"Link") * rotation * prime(setelt(firstind(lpdo[1],tags="Site")=>x[1]))
#      Tdown = dag(rotation) * prime(setelt(firstind(lpdo[1],tags="Site")=>x[1]))
#      gradients[1] += (Tup * R[2] * Tdown)/(localnorm[1]*prob)
#    end
#    for j in 2:N-1
#      if (basis[j] == "Z")
#        Tup = prime(lpdo[j],"Link") * setelt(firstind(lpdo[j],tags="Site")=>x[j])
#        gradients[1j] += (L[j-1] * Tup * R[j+1] * setelt(firstind(lpdo[j],tags="Site")=>x[j]))/(localnorm[j]*prob)
#      else
#        rotation = makegate(lpdo,"m$(basis[j])",j)
#        Tup = prime(lpdo[j],"Link") * rotation * prime(setelt(firstind(lpdo[j],tags="Site")=>x[j]))
#        Tdown = dag(rotation) * prime(setelt(firstind(lpdo[j],tags="Site")=>x[j]))
#        gradients[j] += (L[j-1] * Tup * R[j+1] * Tdown)/(localnorm[j]*prob)
#      end
#    end
#    if (basis[N] == "Z")
#      Tup = prime(lpdo[N],"Link") * setelt(firstind(lpdo[N],tags="Site")=>x[N])
#      gradients[N] += (Tup * L[N-1] * setelt(firstind(lpdo[N],tags="Site")=>x[N]))/(localnorm[N]*prob)
#    else
#      rotation = makegate(lpdo,"m$(basis[N])",N)
#      Tup = prime(lpdo[N],"Link") * rotation * prime(setelt(firstind(lpdo[N],tags="Site")=>x[N]))
#      Tdown = dag(rotation) * prime(setelt(firstind(lpdo[N],tags="Site")=>x[N]))
#      gradients[N] += (Tup * L[N-1] * Tdown)/(localnorm[N]*prob)
#    end
#  end
#  gradients = -2.0*gradients/size(data)[1]
#  return gradients,loss 
#end

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
function statetomography!(model::Union{MPS,MPO},
                          opt::Optimizer;
                          samples::Array,
                          bases::Array,
                          batchsize::Int64=500,
                          epochs::Int64=10000,
                          target::MPS,
                          localnorm::Bool=false)
  for j in 1:length(model)
    replaceind!(target[j],firstind(target[j],"Site"),firstind(model[j],"Site"))
  end
  num_batches = Int(floor(size(samples)[1]/batchsize))
  
  for ep in 1:epochs
    permut = shuffle(1:size(samples)[1])
    samples = samples[permut,:]
    bases   = bases[permut,:]
    
    avg_loss = 0.0
    for b in 1:num_batches
      batch_samples = samples[(b-1)*batchsize+1:b*batchsize,:]
      batch_bases   = bases[(b-1)*batchsize+1:b*batchsize,:]
      
      if localnorm == true
        model_norm = copy(model)
        logZ,localnorms = lognormalize!(model_norm) 
        grads,loss = gradients(model_norm,batch_samples,batch_bases,localnorm=localnorms)
      else
        grads,loss = gradients(model,batch_samples,batch_bases)
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

#"""
#Negative log likelihood for LPDO
#"""
#function nll(lpdo::MPO,data::Array,bases::Array)
#  N = length(lpdo)
#  loss = 0.0
#  for n in 1:size(data)[1]
#    x = data[n,:]
#    x .+= 1
#    basis = bases[n,:]
#    
#    if (basis[1] == "Z")
#      prob = prime(lpdo[1],"Link") * setelt(firstind(lpdo[1],tags="Site")=>x[1])
#      Tdag = dag(lpdo[1]) * setelt(firstind(lpdo[1],tags="Site")=>x[1])
#      prob = prob * Tdag 
#    else
#      rotation = makegate(lpdo,"m$(basis[1])",1)
#      prob = prime(lpdo[1],"Link") * rotation * prime(setelt(firstind(lpdo[1],tags="Site")=>x[1]))
#      Tdag = dag(lpdo[1]) * dag(rotation) * prime(setelt(firstind(lpdo[1],tags="Site")=>x[1]))
#      prob = prob * Tdag 
#    end
#    for j in 2:N-1
#      if (basis[j] == "Z")
#        prob = prob * prime(lpdo[j],"Link") * setelt(firstind(lpdo[j],tags="Site")=>x[j])
#        Tdag = dag(lpdo[j]) * setelt(firstind(lpdo[j],tags="Site")=>x[j])
#        prob = prob * Tdag
#      else
#        rotation = makegate(lpdo,"m$(basis[j])",j)
#        prob = prob * prime(lpdo[j],"Link") * rotation * prime(setelt(firstind(lpdo[j],tags="Site")=>x[j]))
#        Tdag = dag(lpdo[j]) * dag(rotation) * prime(setelt(firstind(lpdo[j],tags="Site")=>x[j]))
#        prob = prob * Tdag
#      end
#    end
#    if (basis[N] == "Z")
#      prob = prob * prime(lpdo[N],"Link") * setelt(firstind(lpdo[N],tags="Site")=>x[N])
#      Tdag = dag(lpdo[N]) * setelt(firstind(lpdo[N],tags="Site")=>x[N])
#      prob = prob * Tdag
#    else
#      rotation = makegate(lpdo,"m$(basis[N])",N)
#      prob = prob * prime(lpdo[N],"Link") * rotation * prime(setelt(firstind(lpdo[N],tags="Site")=>x[N]))
#      Tdag = dag(lpdo[N]) * dag(rotation) * prime(setelt(firstind(lpdo[N],tags="Site")=>x[N]))
#      prob = prob * Tdag
#    end
#    loss -= log(real(prob[]))/size(data)[1]
#  end
#  return loss
#end





""" OLD """

#"""
#Negative log likelihood for MPS
#"""
#function nll(psi::MPS,data::Array,bases::Array)
#  N = length(psi)
#  loss = 0.0
#  s = siteinds(psi)
#  for n in 1:size(data)[1]
#    x = data[n,:]
#    x .+= 1
#    basis = bases[n,:]
#    
#    if (basis[1] == "Z")
#      psix = dag(psi[1]) * setelt(s[1]=>x[1])
#    else
#      rotation = makegate(psi,"m$(basis[1])",1)
#      psi_r = dag(psi[1]) * dag(rotation)
#      psix = noprime!(psi_r) * setelt(s[1]=>x[1])
#    end
#    for j in 2:N-1
#      if (basis[j] == "Z")
#        psix = psix * dag(psi[j]) * setelt(s[j]=>x[j])
#      else
#        rotation = makegate(psi,"m$(basis[j])",j)
#        psi_r = dag(psi[j]) * dag(rotation)
#        psix = psix * noprime!(psi_r) * setelt(s[j]=>x[j])
#      end
#    end
#    if (basis[N] == "Z")
#      psix = (psix * dag(psi[N]) * setelt(s[N]=>x[N]))[]
#    else
#      rotation = makegate(psi,"m$(basis[N])",N)
#      psi_r = dag(psi[N]) * dag(rotation)
#      psix = (psix * noprime!(psi_r) * setelt(s[N]=>x[N]))[]
#    end
#    prob = abs2(psix)
#    loss -= log(prob)/size(data)[1]
#  end
#  return loss
#end



