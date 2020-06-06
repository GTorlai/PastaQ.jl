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
Negative log likelihood (multiple bases)
"""
function nll(psi::MPS,data::Array,bases::Array)
  N = length(psi)
  loss = 0.0
  for n in 1:size(data)[1]
    x = data[n,:]
    x .+= 1
    basis = bases[n,:]
    
    if (basis[1] == "Z")
      psix = dag(psi[1]) * setelt(siteind(psi,1)=>x[1])
    else
      rotation = makegate(psi,"m$(basis[1])",1)
      psi_r = dag(psi[1]) * dag(rotation)
      psix = noprime!(psi_r) * setelt(siteind(psi,1)=>x[1])
    end
    for j in 2:N-1
      if (basis[j] == "Z")
        psix = psix * dag(psi[j]) * setelt(siteind(psi,j)=>x[j])
      else
        rotation = makegate(psi,"m$(basis[j])",j)
        psi_r = dag(psi[j]) * dag(rotation)
        psix = psix * noprime!(psi_r) * setelt(siteind(psi,j)=>x[j])
      end
    end
    if (basis[N] == "Z")
      psix = (psix * dag(psi[N]) * setelt(siteind(psi,N)=>x[N]))[]
    else
      rotation = makegate(psi,"m$(basis[N])",N)
      psi_r = dag(psi[N]) * dag(rotation)
      psix = (psix * noprime!(psi_r) * setelt(siteind(psi,N)=>x[N]))[]
    end
    prob = abs2(psix)
    loss -= log(prob)/size(data)[1]
  end
  return loss
end

"""
Gradients of logZ 
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
Gradients of NLL with local norms (computational basis)
"""
function gradnll(psi::MPS,data::Array,bases::Array;localnorm=nothing)
  loss = 0.0

  N = length(psi)
  L = Vector{ITensor}(undef, N-1)
  R = Vector{ITensor}(undef, N)
  
  if isnothing(localnorm)
    localnorm = ones(N)
  end
  
  gradients = ITensor[]
  for j in 1:N
    push!(gradients,ITensor(inds(psi[j])))
  end
  for n in 1:size(data)[1]
    x = data[n,:] 
    x.+=1
    basis = bases[n,:]
    
    """ LEFT ENVIRONMENTS """
    if (basis[1] == "Z")
      L[1] = dag(psi[1]) * setelt(siteind(psi,1)=>x[1])
    else
      rotation = makegate(psi,"m$(basis[1])",1)
      psi_r = dag(psi[1]) * dag(rotation)
      L[1] = noprime!(psi_r) * setelt(siteind(psi,1)=>x[1])
    end
    for j in 2:N-1
      if (basis[j] == "Z")
        L[j] = L[j-1] * dag(psi[j]) * setelt(siteind(psi,j)=>x[j])
      else
        rotation = makegate(psi,"m$(basis[j])",j)
        psi_r = dag(psi[j]) * dag(rotation)
        L[j] = L[j-1] * noprime!(psi_r) * setelt(siteind(psi,j)=>x[j])
      end
    end
    if (basis[N] == "Z")
      psix = (L[N-1] * dag(psi[N]) * setelt(siteind(psi,N)=>x[N]))[]
    else
      rotation = makegate(psi,"m$(basis[N])",N)
      psi_r = dag(psi[N]) * dag(rotation)
      psix = (L[N-1] * noprime!(psi_r) * setelt(siteind(psi,N)=>x[N]))[]
    end
    prob = abs2(psix)
    loss -= log(prob)/size(data)[1]
    
    """ RIGHT ENVIRONMENTS """
    if (basis[N] == "Z")
      R[N] = dag(psi[N]) * setelt(siteind(psi,N)=>x[N])
    else
      rotation = makegate(psi,"m$(basis[N])",N)
      psi_r = dag(psi[N]) * dag(rotation)
      R[N] = noprime!(psi_r) * setelt(siteind(psi,N)=>x[N])
    end
    for j in reverse(2:N-1)
      if (basis[j] == "Z")
        R[j] = R[j+1] * dag(psi[j]) * setelt(siteind(psi,j)=>x[j])
      else
        rotation = makegate(psi,"m$(basis[j])",j)
        psi_r = dag(psi[j]) * dag(rotation)
        R[j] = R[j+1] * noprime!(psi_r) * setelt(siteind(psi,j)=>x[j])
      end
    end
    
    """ GRADIENTS """
    if (basis[1] == "Z")
      gradients[1] += (R[2] * setelt(siteind(psi,1)=>x[1]))/(localnorm[1]*psix)
    else
      rotation = makegate(psi,"m$(basis[1])",1)
      projection = dag(rotation) * prime(setelt(siteind(psi,1)=>x[1]))
      gradients[1] += (R[2] * projection)/(localnorm[1]*psix)
    end
    for j in 2:N-1
      if (basis[j] == "Z")
        gradients[j] += (L[j-1] * setelt(siteind(psi,j)=>x[j]) * R[j+1])/(localnorm[j]*psix)
      else
        rotation = makegate(psi,"m$(basis[j])",j)
        projection = dag(rotation) * prime(setelt(siteind(psi,j)=>x[j]))
        gradients[j] += (L[j-1] * projection * R[j+1])/(localnorm[j]*psix)
      end
    end
    if (basis[N] == "Z")
      gradients[N] += (L[N-1] * setelt(siteind(psi,N)=>x[N]))/(localnorm[N]*psix)
    else
      rotation = makegate(psi,"m$(basis[N])",N)
      projection = dag(rotation) * prime(setelt(siteind(psi,N)=>x[N]))
      gradients[N] += (L[N-1] * projection)/(localnorm[N]*psix)
    end
  end
  gradients = -2*gradients/size(data)[1]
  return gradients,loss 
end

"""
Compute the total gradients
"""
function gradients(psi::MPS,data::Array,bases::Array;localnorm=nothing)
  g_logZ,logZ = gradlogZ(psi,localnorm=localnorm)
  g_nll, nll  = gradnll(psi,data,bases,localnorm=localnorm)
  grads = g_logZ + g_nll
  loss = logZ + nll
  return grads,loss
end

function statetomography!(psi::MPS,opt::Optimizer;
                          samples::Array,
                          bases::Array,
                          batchsize::Int64=500,
                          epochs::Int64=10000,
                          targetpsi::MPS,
                          localnorm::Bool=false)
  for j in 1:length(psi)
    replaceinds!(targetpsi[j],inds(targetpsi[j],"Site"),inds(psi[j],"Site"))
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
        psi_norm = copy(psi)
        logZ,localnorms = lognormalize!(psi_norm) 
        grads,loss = gradients(psi_norm,batch_samples,batch_bases,localnorm=localnorms)
      else
        grads,loss = gradients(psi,batch_samples,batch_bases)
      end
      avg_loss += loss/Float64(num_batches)
      updateSGD!(psi,grads,opt)
    end
    psi_eval = copy(psi)
    lognormalize!(psi_eval)
    @assert norm(psi_eval) ≈ 1
    fidelity = abs2(inner(psi_eval,targetpsi))
    print("Ep = ",ep,"  ")
    print("Loss = ")
    @printf("%.5E",avg_loss)
    print("  ")
    print("Fidelity = ")
    @printf("%.3E",fidelity)
    print("\n")
  end
end




""" LPDO """

"""
Initialize for LPDO state tomography
"""
function initializeQST(N::Int,χ::Int,ξ::Int;d::Int=2,seed::Int=1234,σ::Float64=0.1)
  rng = MersenneTwister(seed)
  
  sites = [Index(d; tags="Site,  n=$s") for s in 1:N]
  links = [Index(χ; tags="Link,  l=$l") for l in 1:N-1]
  kraus = [Index(ξ; tags="Kraus, k=$s") for s in 1:N]

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

#"""
#Normalize the LPDO locally and store the local norms
#"""
#function lognormalize!(lpdo::MPO)
#  localnorms = []
#  blob = dag(lpdo[1]) * prime(lpdo[1],"Link")
#  localZ = norm(blob)
#  logZ = 0.5*log(localZ)
#  blob /= sqrt(localZ)
#  lpdo[1] /= (localZ^0.25)
#  push!(localnorms,localZ^0.25)
#  for j in 2:length(lpdo)-1
#    blob = blob * dag(lpdo[j])
#    blob = blob * prime(lpdo[j],"Link")
#    localZ = norm(blob)
#    logZ += 0.5*log(localZ)
#    blob /= sqrt(localZ)
#    lpdo[j] /= (localZ^0.25)
#    push!(localnorms,localZ^0.25)  
#  end
#  blob = blob * dag(lpdo[length(lpdo)]);
#  blob = blob * prime(lpdo[length(lpdo)],"Link")
#  localZ = norm(blob)
#  lpdo[length(lpdo)] /= sqrt(localZ)
#  push!(localnorms,sqrt(localZ))
#  logZ += log(real(blob[]))
#  return logZ,localnorms
#end


