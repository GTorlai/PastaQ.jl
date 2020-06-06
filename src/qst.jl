struct QST
  N::Int  
  d::Int
  χ::Int
  seed::Int
  rng::MersenneTwister
  σ::Float64
  parstype::String
  psi::MPS
end

function QST(;N::Int,
             d::Int=2,
             χ::Int=2,
             seed::Int=1234,
             σ::Float64=0.1,
             parstype="real")

  rng = MersenneTwister(seed)
  
  sites = [Index(d; tags="Site, n=$s") for s in 1:N]
  links = [Index(χ; tags="Link, l=$l") for l in 1:N-1]

  M = ITensor[]
  # Site 1 
  rand_mat = σ * (ones(d,χ) - 2*rand!(rng,zeros(d,χ)))
  if parstype == "complex"
    rand_mat += im * σ * (ones(d,χ) - 2*rand!(rng,zeros(d,χ)))
  end
  push!(M,ITensor(rand_mat,sites[1],links[1]))
  # Site 2..N-1
  for j in 2:N-1
    rand_mat = σ * (ones(χ,d,χ) - 2*rand!(rng,zeros(χ,d,χ)))
    if parstype == "complex"
      rand_mat += im * σ * (ones(χ,d,χ) - 2*rand!(rng,zeros(χ,d,χ)))
    end
    push!(M,ITensor(rand_mat,links[j-1],sites[j],links[j]))
  end
  # Site N
  rand_mat = σ * (ones(χ,d) - 2*rand!(rng,zeros(χ,d)))
  if parstype == "complex"
    rand_mat += im * σ * (ones(χ,d) - 2*rand!(rng,zeros(χ,d)))
  end
  push!(M,ITensor(rand_mat,links[N-1],sites[N]))
  
  psi = MPS(M)

  return QST(N,d,χ,seed,rng,σ,parstype,psi)
end

"""
Normalize the MPS locally and store the local norms
"""
function lognormalize!(psi::MPS)
  localnorms = []
  blob = dag(psi[1]) * prime(psi[1],"Link")
  localZ = norm(blob)
  logZ = 0.5*log(localZ)
  blob /= sqrt(localZ)
  psi[1] /= (localZ^0.25)
  push!(localnorms,localZ^0.25)
  for j in 2:length(psi)-1
    blob = blob * dag(psi[j]);
    blob = blob * prime(psi[j],"Link")
    localZ = norm(blob)
    logZ += 0.5*log(localZ)
    blob /= sqrt(localZ)
    psi[j] /= (localZ^0.25)
    push!(localnorms,localZ^0.25)  
  end
  blob = blob * dag(psi[length(psi)]);
  blob = blob * prime(psi[length(psi)],"Link")
  localZ = norm(blob)
  psi[length(psi)] /= sqrt(localZ)
  push!(localnorms,sqrt(localZ))
  logZ += log(real(blob[]))
  return logZ,localnorms
end

"""
Gradients of logZ
"""
function gradlogZ(psi::MPS)
  N = length(psi)
  L = Vector{ITensor}(undef, N-1)
  R = Vector{ITensor}(undef, N)
  
  # Sweep right to get L
  L[1] = dag(psi[1]) * prime(psi[1],"Link")
  for j in 2:N-1
    L[j] = L[j-1] * dag(psi[j])
    L[j] = L[j] * prime(psi[j],"Link")
  end
  Z = L[N-1] * dag(psi[N])
  Z = real((Z * prime(psi[N],"Link"))[])

  # Sweep left to get R
  R[N] = dag(psi[N]) * prime(psi[N],"Link")
  for j in reverse(2:N-1)
    R[j] = R[j+1] * dag(psi[j])
    R[j] = R[j] * prime(psi[j],"Link")
  end
  # Get the gradients of the normalization
  gradients = Vector{ITensor}(undef, N)
  gradients[1] = prime(psi[1],"Link") * R[2]/Z
  for j in 2:N-1
    gradients[j] = (L[j-1] * prime(psi[j],"Link") * R[j+1])/Z
  end
  gradients[N] = (L[N-1] * prime(psi[N],"Link"))/Z
  return 2*gradients,log(Z)
end

"""
Gradients of logZ using local normalizations
"""
function gradlogZ(psi::MPS,localnorm::Array)
  N = length(psi)
  L = Vector{ITensor}(undef, N-1)
  R = Vector{ITensor}(undef, N)
  
  # Sweep right to get L
  L[1] = dag(psi[1]) * prime(psi[1],"Link")
  for j in 2:N-1
    L[j] = L[j-1] * dag(psi[j])
    L[j] = L[j] * prime(psi[j],"Link")
  end
  Z = L[N-1] * dag(psi[N])
  Z = real((Z * prime(psi[N],"Link"))[])

  # Sweep left to get R
  R[N] = dag(psi[N]) * prime(psi[N],"Link")
  for j in reverse(2:N-1)
    R[j] = R[j+1] * dag(psi[j])
    R[j] = R[j] * prime(psi[j],"Link")
  end
  # Get the gradients of the normalization
  gradients = Vector{ITensor}(undef, N)
  gradients[1] = prime(psi[1],"Link") * R[2]/(localnorm[1]*Z)
  for j in 2:N-1
    gradients[j] = (L[j-1] * prime(psi[j],"Link") * R[j+1])/(localnorm[j]*Z)
  end
  gradients[N] = (L[N-1] * prime(psi[N],"Link"))/(localnorm[N]*Z)
  return 2*gradients,log(Z)
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
Gradients of NLL (multiple bases)
"""
function gradnll(psi::MPS,data::Array,bases::Array)
  loss = 0.0

  N = length(psi)
  L = Vector{ITensor}(undef, N-1)
  R = Vector{ITensor}(undef, N)
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
      gradients[1] += (R[2] * setelt(siteind(psi,1)=>x[1]))/psix
    else
      rotation = makegate(psi,"m$(basis[1])",1)
      projection = dag(rotation) * prime(setelt(siteind(psi,1)=>x[1]))
      gradients[1] += (R[2] * projection)/psix
    end
    for j in 2:N-1
      if (basis[j] == "Z")
        gradients[j] += (L[j-1] * setelt(siteind(psi,j)=>x[j]) * R[j+1])/psix
      else
        rotation = makegate(psi,"m$(basis[j])",j)
        projection = dag(rotation) * prime(setelt(siteind(psi,j)=>x[j]))
        gradients[j] += (L[j-1] * projection * R[j+1])/psix
      end
    end
    if (basis[N] == "Z")
      gradients[N] += (L[N-1] * setelt(siteind(psi,N)=>x[N]))/psix
    else
      rotation = makegate(psi,"m$(basis[N])",N)
      projection = dag(rotation) * prime(setelt(siteind(psi,N)=>x[N]))
      gradients[N] += (L[N-1] * projection)/psix
    end
  end
  gradients = -2*gradients/size(data)[1]
  
  return gradients,loss 
end

"""
Gradients of NLL with local norms (computational basis)
"""
function gradnll(psi::MPS,data::Array,bases::Array,localnorm::Array)
  loss = 0.0

  N = length(psi)
  L = Vector{ITensor}(undef, N-1)
  R = Vector{ITensor}(undef, N)
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






""" OLD FUNCTIONS """

#"""
#<x|psi>
#"""
#function psiofx(psi::MPS,x::Array)
#  return inner(psi,productMPS(siteinds(psi),x))
#end
#
#"""
#|<x|psi>|^2
#"""
#function psi2ofx(psi::MPS,x::Array)
#  psix = psiofx(psi,x)
#  return abs2(psix)
#end
#
#
#"""
#Negative log likelihood (computational basis)
#"""
#function nll(psi::MPS,data::Array)
#  loss = 0.0
#  for n in 1:size(data)[1]
#    x = data[n,:]
#    x .+= 1
#    loss -= log(psi2ofx(psi,x))/size(data)[1]
#  end
#  return loss
#end
#
#
#"""
#Negative log likelihood (multiple bases)
#"""
#function nll(psi::MPS,data::Array,bases::Array)
#  loss = 0.0
#  for n in 1:size(data)[1]
#    x = data[n,:]
#    x .+= 1
#    basis = bases[n,:]
#    gates = makemeasurementgates(basis)
#    tensors = compilecircuit(psi,gates)
#    psi_r = runcircuit(psi,tensors)
#    loss -= log(psi2ofx(psi_r,x))/size(data)[1]
#  end
#  return loss
#end
#
#
#"""
#Gradients of NLL (computational basis)
#"""
#function gradnll(psi::MPS,data::Array)
#  loss = 0.0
#
#  N = length(psi)
#  L = Vector{ITensor}(undef, N-1)
#  R = Vector{ITensor}(undef, N)
#  gradients = ITensor[]
#  for j in 1:N
#    push!(gradients,ITensor(inds(psi[j])))
#  end
#  for n in 1:size(data)[1]
#    x = data[n,:] 
#    x.+=1
#    L[1] = dag(psi[1]) * setelt(siteind(psi,1)=>x[1])
#    for j in 2:N-1
#      L[j] = L[j-1] * dag(psi[j]) * setelt(siteind(psi,j)=>x[j]) 
#    end
#    psix = (L[N-1] * dag(psi[N]) * setelt(siteind(psi,N)=>x[N]))[]
#    prob = abs2(psix)
#    loss -= log(prob)/size(data)[1]
#    
#    R[N] = dag(psi[N]) * setelt(siteind(psi,N)=>x[N])
#    for j in reverse(2:N-1)
#      R[j] = R[j+1] * dag(psi[j]) * setelt(siteind(psi,j)=>x[j])
#    end
#    gradients[1] += (R[2] * setelt(siteind(psi,1)=>x[1]))/psix
#    for j in 2:N-1
#      gradients[j] += (L[j-1] * setelt(siteind(psi,j)=>x[j]) * R[j+1])/psix
#    end
#    gradients[N] += (L[N-1] *setelt(siteind(psi,N)=>x[N]))/psix
#  end
#  gradients = -2*gradients/size(data)[1]
#  return gradients,loss 
#end
#
#"""
#Gradients of NLL with local norms (computational basis)
#"""
#function gradnll(psi::MPS,data::Array;localnorm::Array)
#  loss = 0.0
#
#  N = length(psi)
#  L = Vector{ITensor}(undef, N-1)
#  R = Vector{ITensor}(undef, N)
#  gradients = ITensor[]
#  for j in 1:N
#    push!(gradients,ITensor(inds(psi[j])))
#  end
#  for n in 1:size(data)[1]
#    x = data[n,:] 
#    x.+=1
#    L[1] = dag(psi[1]) * setelt(siteind(psi,1)=>x[1])
#    for j in 2:N-1
#      L[j] = L[j-1] * dag(psi[j]) * setelt(siteind(psi,j)=>x[j]) 
#    end
#    psix = (L[N-1] * dag(psi[N]) * setelt(siteind(psi,N)=>x[N]))[]
#    prob = abs2(psix)
#    loss -= log(prob)/size(data)[1]
#    
#    R[N] = dag(psi[N]) * setelt(siteind(psi,N)=>x[N])
#    for j in reverse(2:N-1)
#      R[j] = R[j+1] * dag(psi[j]) * setelt(siteind(psi,j)=>x[j])
#    end
#
#    gradients[1] += (R[2] * setelt(siteind(psi,1)=>x[1]))/(localnorm[1]*psix)
#    for j in 2:N-1
#      gradients[j] += (L[j-1] * setelt(siteind(psi,j)=>x[j]) * R[j+1])/(localnorm[j]*psix)
#    end
#    gradients[N] += (L[N-1] *setelt(siteind(psi,N)=>x[N]))/(localnorm[N]*psix)
#  end
#  gradients = -2*gradients/size(data)[1]
#  return gradients,loss 
#end


#
