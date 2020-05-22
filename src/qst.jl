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

function normalization(psi::MPS)
  return inner(psi,psi)
end

function normalize!(psi::MPS)
  Z = normalization(psi)
  for j in 1:length(psi)
    psi[j] /= sqrt(Z^(1/(length(psi))))
  end
end

function lognormalization(psi::MPS)
  blob = dag(psi[1]) * prime(psi[1],"Link")
  localZ = norm(blob)
  logZ = 0.5*log(localZ)
  blob /= sqrt(localZ)

  for j in 2:length(psi)-1
    blob = blob * dag(psi[j]);
    blob = blob * prime(psi[j],"Link")
    localZ = norm(blob)
    logZ += 0.5*log(localZ)
    blob /= sqrt(localZ)
  end
  blob = blob * dag(psi[length(psi)]);
  blob = blob * prime(psi[length(psi)],"Link")
  logZ += log(real(blob[]))
  return logZ
end

function lognormalize!(psi::MPS)
  localnorms = []
  blob = dag(psi[1]) * prime(psi[1],"Link")
  localZ = norm(blob)
  logZ = 0.5*log(localZ)
  blob /= sqrt(localZ)
  psi[1] /= (localZ^0.25)#sqrt(localZ)
  push!(localnorms,localZ^0.25)
  for j in 2:length(psi)-1
    blob = blob * dag(psi[j]);
    blob = blob * prime(psi[j],"Link")
    localZ = norm(blob)
    logZ += 0.5*log(localZ)
    blob /= sqrt(localZ)
    psi[j] /= (localZ^0.25)#sqrt(localZ)
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

function psiofx(psi::MPS,x::Array)
  return inner(psi,productMPS(siteinds(psi),x))
end

function psi2ofx(psi::MPS,x::Array)
  psix = psiofx(psi,x)
  return norm(psix)^2
end

function nll(psi::MPS,data::Array)
  loss = 0.0
  for n in 1:size(data)[1]
    x = data[n,:]
    x .+= 1
    loss -= log(psi2ofx(psi,x))/size(data)[1]
  end
  return loss
end

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

function gradnll(psi::MPS,data::Array)
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
    L[1] = psi[1] * setelt(siteind(psi,1)=>x[1])
    for j in 2:N-1
      L[j] = L[j-1] * psi[j] * setelt(siteind(psi,j)=>x[j]) 
    end
    psix = real((L[N-1] * psi[N] * setelt(siteind(psi,N)=>x[N]))[])
    prob = norm(psix)^2
    loss -= log(prob)/size(data)[1]
    R[N] = psi[N] * setelt(siteind(psi,N)=>x[N])
    for j in reverse(2:N-1)
      R[j] = R[j+1] * psi[j] * setelt(siteind(psi,j)=>x[j])
    end
    gradients[1] += (R[2] * setelt(siteind(psi,1)=>x[1]))/psix
    for j in 2:N-1
      gradients[j] += (L[j-1] * setelt(siteind(psi,j)=>x[j]) * R[j+1])/psix
    end
    gradients[N] += (L[N-1] *setelt(siteind(psi,N)=>x[N]))/psix
  end
  gradients = -2*gradients/size(data)[1]
  return gradients,loss 
end

function gradnll(psi::MPS,data::Array,localnorm::Array)
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
    L[1] = psi[1] * setelt(siteind(psi,1)=>x[1])
    for j in 2:N-1
      L[j] = L[j-1] * psi[j] * setelt(siteind(psi,j)=>x[j]) 
    end
    psix = real((L[N-1] * psi[N] * setelt(siteind(psi,N)=>x[N]))[])
    prob = norm(psix)^2
    loss -= log(prob)/size(data)[1]
    R[N] = psi[N] * setelt(siteind(psi,N)=>x[N])
    for j in reverse(2:N-1)
      R[j] = R[j+1] * psi[j] * setelt(siteind(psi,j)=>x[j])
    end
    gradients[1] += (R[2] * setelt(siteind(psi,1)=>x[1]))/(localnorm[1]*psix)
    for j in 2:N-1
      gradients[j] += (L[j-1] * setelt(siteind(psi,j)=>x[j]) * R[j+1])/(localnorm[j]*psix)
    end
    gradients[N] += (L[N-1] *setelt(siteind(psi,N)=>x[N]))/(localnorm[N]*psix)
  end
  gradients = -2*gradients/size(data)[1]
  return gradients,loss 
end

function statetomography(qst::QST,opt::Optimizer;
                         data::Array,
                         batchsize::Int64=500,
                         epochs::Int64=10000,
                         targetpsi::MPS)
  for j in 1:qst.N
    replaceinds!(targetpsi[j],inds(targetpsi[j],"Site"),inds(qst.psi[j],"Site"))
  end
  num_batches = Int(floor(size(data)[1]/batchsize))
  
  for ep in 1:epochs
    data = data[shuffle(1:end),:]
    loss = 0.0
    for b in 1:num_batches
      batch = data[(b-1)*batchsize+1:b*batchsize,:]
      g_logZ, logZ = gradlogZ(qst.psi)
      g_nll,nll    = gradnll(qst.psi,data)
      gradients = g_logZ + g_nll
      loss += (logZ + nll)/Float64(num_batches)
      updateSGD!(qst.psi,gradients,opt)
    end
    psi = copy(qst.psi)
    normalize!(psi)
    overlap = abs(inner(psi,targetpsi))
    print("Ep = ",ep,"  ")
    print("Loss = ")
    @printf("%.5E",loss)
    print("  ")
    print("Overlap = ")
    @printf("%.3E",overlap)
    print("\n")
  end
end
