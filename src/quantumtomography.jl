"""
Initializer for MPS tomography
"""
function initializetomography(N::Int,
                              χ::Int;
                              d::Int=2,
                              seed::Int=1234,
                              σ::Float64=0.1)
  rng = MersenneTwister(seed)
  
  sites = [Index(d; tags="Site, n=$s") for s in 1:N]
  links = [Index(χ; tags="Link, l=$l") for l in 1:N-1]

  M = ITensor[]
  # Site 1 
  rand_mat = σ * (ones(d,χ) - 2*rand(rng,d,χ))
  rand_mat += im * σ * (ones(d,χ) - 2*rand(rng,d,χ))
  push!(M,ITensor(rand_mat,sites[1],links[1]))
  # Site 2..N-1
  for j in 2:N-1
    rand_mat = σ * (ones(χ,d,χ) - 2*rand(rng,χ,d,χ))
    rand_mat += im * σ * (ones(χ,d,χ) - 2*rand(rng,χ,d,χ))
    push!(M,ITensor(rand_mat,links[j-1],sites[j],links[j]))
  end
  # Site N
  rand_mat = σ * (ones(χ,d) - 2*rand(rng,χ,d))
  rand_mat += im * σ * (ones(χ,d) - 2*rand(rng,χ,d))
  push!(M,ITensor(rand_mat,links[N-1],sites[N]))
  
  psi = MPS(M)
  return psi
end

"""
Initializer for LPDO tomography
"""
function initializetomography(N::Int,
                              χ::Int,
                              ξ::Int;
                              d::Int=2,
                              seed::Int=1234,
                              σ::Float64=0.1)
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
function lognormalize!(M::Union{MPS,MPO}; choi::Bool=false)
  
  # PROCESS TOMOGRAPHY
  if choi
    choinorm = 2^length(M)
    choi_local = 1/2^0.25 

    localnorms = []
    blob = dag(M[1]) * prime(M[1],"Link")
    localZ = norm(blob)
    logZ = 0.5*log(localZ)
    blob /= sqrt(localZ)
    M[1] /= (choi_local * localZ^0.25)
    push!(localnorms,localZ^0.25)
    for j in 2:length(M)-1
      blob = blob * dag(M[j]);
      blob = blob * prime(M[j],"Link")
      localZ = norm(blob)
      logZ += 0.5*log(localZ)
      blob /= sqrt(localZ)
      M[j] /= (choi_local * localZ^0.25)
      push!(localnorms,localZ^0.25)  
    end
    blob = blob * dag(M[length(M)]);
    blob = blob * prime(M[length(M)],"Link")
    localZ = norm(blob)
    M[length(M)] /= (choi_local * sqrt(localZ))
    push!(localnorms,sqrt(localZ))
    logZ += log(real(blob[]))
    logZ += log(choinorm)
    localnorms .= localnorms .* choi_local
  
  # STATE TOMOGRAPHY
  else
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
  end
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
function gradnll(psi::MPS,data::Array;localnorm=nothing,choi::Bool=false)
  N = length(psi)

  s = siteinds(psi)

  links = [linkind(psi, n) for n in 1:N-1]

  ElT = eltype(psi[1])

  nthreads = Threads.nthreads()

  L = [Vector{ITensor{1}}(undef, N) for _ in 1:nthreads]
  Lpsi = [Vector{ITensor}(undef, N) for _ in 1:nthreads]

  R = [Vector{ITensor{1}}(undef, N) for _ in 1:nthreads]
  Rpsi = [Vector{ITensor}(undef, N) for _ in 1:nthreads]

  for nthread in 1:nthreads
    for n in 1:N-1
      L[nthread][n] = ITensor(ElT, undef, links[n])
      Lpsi[nthread][n] = ITensor(ElT, undef, s[n], links[n])
    end
    Lpsi[nthread][N] = ITensor(ElT, undef, s[N])

    for n in N:-1:2
      R[nthread][n] = ITensor(ElT, undef, links[n-1])
      Rpsi[nthread][n] = ITensor(ElT, undef, links[n-1], s[n])
    end
    Rpsi[nthread][1] = ITensor(ElT, undef, s[1])
  end

  if isnothing(localnorm)
    localnorm = ones(N)
  end

  psidag = dag(psi)

  gradients = [[ITensor(ElT, inds(psi[j])) for j in 1:N] for _ in 1:nthreads]

  grads = [[ITensor(ElT, undef, inds(psi[j])) for j in 1:N] for _ in 1:nthreads]

  loss = zeros(nthreads)
  
  if choi
    Threads.@threads for n in 1:size(data)[1]
      nthread = Threads.threadid()
      
      x = data[n,:] 
      """ LEFT ENVIRONMENTS """
      L[nthread][1] .= psidag[1] .* dag(gate(x[1],s[1]))
      for j in 2:N-1
        Lpsi[nthread][j] .= L[nthread][j-1] .* psidag[j]
        if iseven(j)
          L[nthread][j] .= Lpsi[nthread][j] .* gate(x[j],s[j])
        else
          L[nthread][j] .= Lpsi[nthread][j] .* dag(gate(x[j],s[j]))
        end
      end
      Lpsi[nthread][N] .= L[nthread][N-1] .* psidag[N]
      psix = (Lpsi[nthread][N] * gate(x[N],s[N]))[]
      prob = abs2(psix)
      loss[nthread] -= log(prob)/size(data)[1]
      
      """ RIGHT ENVIRONMENTS """
      R[nthread][N] .= psidag[N] .* gate(x[N],s[N])
      for j in reverse(2:N-1)
        Rpsi[nthread][j] .= psidag[j] .* R[nthread][j+1]
        if iseven(j)
          R[nthread][j] .= Rpsi[nthread][j] .* gate(x[j],s[j])
        else
          R[nthread][j] .= Rpsi[nthread][j] .* dag(gate(x[j],s[j]))
        end
      end

      """ GRADIENTS """
      # TODO: fuse into one call to mul!
      grads[nthread][1] .= dag(gate(x[1],s[1])) .* R[nthread][2] 
      gradients[nthread][1] .+= (1 / (localnorm[1] * psix)) .* grads[nthread][1]
      for j in 2:N-1
        if iseven(j)
          Rpsi[nthread][j] .= L[nthread][j-1] .* gate(x[j],s[j])
        else
          Rpsi[nthread][j] .= L[nthread][j-1] .* dag(gate(x[j],s[j]))
        end
        # TODO: fuse into one call to mul!
        grads[nthread][j] .= Rpsi[nthread][j] .* R[nthread][j+1]
        gradients[nthread][j] .+= (1 / (localnorm[j] * psix)) .* grads[nthread][j]
      end
      grads[nthread][N] .= L[nthread][N-1] .* gate(x[N], s[N])
      gradients[nthread][N] .+= (1 / (localnorm[N] * psix)) .* grads[nthread][N]
    end

    for nthread in 1:nthreads
      for g in gradients[nthread]
        g .= (-2/size(data)[1]) .* g
      end
    end

    gradients_tot = [ITensor(ElT, inds(psi[j])) for j in 1:N]
    loss_tot = 0.0
    for nthread in 1:nthreads
      gradients_tot .+= gradients[nthread]
      loss_tot += loss[nthread]
    end

  else
    Threads.@threads for n in 1:size(data)[1]

      nthread = Threads.threadid()

      x = data[n,:] 
      
      """ LEFT ENVIRONMENTS """
      L[nthread][1] .= psidag[1] .* gate(x[1],s[1])
      for j in 2:N-1
        Lpsi[nthread][j] .= L[nthread][j-1] .* psidag[j]
        L[nthread][j] .= Lpsi[nthread][j] .* gate(x[j],s[j])
      end
      Lpsi[nthread][N] .= L[nthread][N-1] .* psidag[N]
      psix = (Lpsi[nthread][N] * gate(x[N],s[N]))[]
      prob = abs2(psix)
      loss[nthread] -= log(prob)/size(data)[1]
      
      """ RIGHT ENVIRONMENTS """
      R[nthread][N] .= psidag[N] .* gate(x[N],s[N])
      for j in reverse(2:N-1)
        Rpsi[nthread][j] .= psidag[j] .* R[nthread][j+1]
        R[nthread][j] .= Rpsi[nthread][j] .* gate(x[j],s[j])
      end

      """ GRADIENTS """
      # TODO: fuse into one call to mul!
      grads[nthread][1] .= gate(x[1],s[1]) .* R[nthread][2] 
      gradients[nthread][1] .+= (1 / (localnorm[1] * psix)) .* grads[nthread][1]
      for j in 2:N-1
        Rpsi[nthread][j] .= L[nthread][j-1] .* gate(x[j],s[j])
        # TODO: fuse into one call to mul!
        grads[nthread][j] .= Rpsi[nthread][j] .* R[nthread][j+1]
        gradients[nthread][j] .+= (1 / (localnorm[j] * psix)) .* grads[nthread][j]
      end
      grads[nthread][N] .= L[nthread][N-1] .* gate(x[N], s[N])
      gradients[nthread][N] .+= (1 / (localnorm[N] * psix)) .* grads[nthread][N]
    end
  
    for nthread in 1:nthreads
      for g in gradients[nthread]
        g .= (-2/size(data)[1]) .* g
      end
    end

    gradients_tot = [ITensor(ElT, inds(psi[j])) for j in 1:N]
    loss_tot = 0.0
    for nthread in 1:nthreads
      gradients_tot .+= gradients[nthread]
      loss_tot += loss[nthread]
    end
  end

  return gradients_tot, loss_tot
end

"""
Gradients of NLL for LPDO 
"""
function gradnll(lpdo::MPO,data::Array;localnorm=nothing,choi::Bool=false)
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
  
  if choi 
    for n in 1:size(data)[1]
      x = data[n,:]
      """ LEFT ENVIRONMENTS """
      T[1] .= lpdo[1] .* gate(x[1],s[1])
      L[1] .= prime(T[1],"Link") .* dag(T[1])
      for j in 2:N-1
        if iseven(j)
          T[j] .= lpdo[j] .* dag(gate(x[j],s[j]))
        else
          T[j] .= lpdo[j] .* gate(x[j],s[j])
        end
        Llpdo[j] .= prime(T[j],"Link") .* L[j-1]
        L[j] .= Llpdo[j] .* dag(T[j])
      end
      T[N] .= lpdo[N] .* dag(gate(x[N],s[N]))
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
      Tp[1] .= prime(lpdo[1],"Link") .* gate(x[1],s[1])
      Agrad[1] .=  Tp[1] .* dag(gate(x[1],s[1]))
      grads[1] .= R[2] .* Agrad[1]
      gradients[1] .+= (1 / (localnorm[1] * prob)) .* grads[1]
      for j in 2:N-1
        if iseven(j)
          Tp[j] .= prime(lpdo[j],"Link") .* dag(gate(x[j],s[j]))
          Lgrad[j-1] .= L[j-1] .* Tp[j]
          Agrad[j] .= Lgrad[j-1] .* gate(x[j],s[j])
        else
          Tp[j] .= prime(lpdo[j],"Link") .* gate(x[j],s[j])
          Lgrad[j-1] .= L[j-1] .* Tp[j]
          Agrad[j] .= Lgrad[j-1] .* dag(gate(x[j],s[j]))
        end
        grads[j] .= R[j+1] .* Agrad[j] 
        gradients[j] .+= (1 / (localnorm[j] * prob)) .* grads[j]
      end
      Tp[N] .= prime(lpdo[N],"Link") .* dag(gate(x[N],s[N]))
      Lgrad[N-1] .= L[N-1] .* Tp[N]
      grads[N] .= Lgrad[N-1] .* gate(x[N],s[N])
      gradients[N] .+= (1 / (localnorm[N] * prob)) .* grads[N]
    end
  else
    for n in 1:size(data)[1]
      x = data[n,:]
      """ LEFT ENVIRONMENTS """
      T[1] .= lpdo[1] .* dag(gate(x[1],s[1]))
      L[1] .= prime(T[1],"Link") .* dag(T[1])
      for j in 2:N-1
        T[j] .= lpdo[j] .* dag(gate(x[j],s[j]))
        Llpdo[j] .= prime(T[j],"Link") .* L[j-1]
        L[j] .= Llpdo[j] .* dag(T[j])
      end
      T[N] .= lpdo[N] .* dag(gate(x[N],s[N]))
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
      Tp[1] .= prime(lpdo[1],"Link") .* dag(gate(x[1],s[1]))
      Agrad[1] .=  Tp[1] .* gate(x[1],s[1])
      grads[1] .= R[2] .* Agrad[1]
      gradients[1] .+= (1 / (localnorm[1] * prob)) .* grads[1]
       for j in 2:N-1
        Tp[j] .= prime(lpdo[j],"Link") .* dag(gate(x[j],s[j]))
        Lgrad[j-1] .= L[j-1] .* Tp[j]
        Agrad[j] .= Lgrad[j-1] .* gate(x[j],s[j])
        grads[j] .= R[j+1] .* Agrad[j] 
        gradients[j] .+= (1 / (localnorm[j] * prob)) .* grads[j]
      end
      Tp[N] .= prime(lpdo[N],"Link") .* dag(gate(x[N],s[N]))
      Lgrad[N-1] .= L[N-1] .* Tp[N]
      grads[N] .= Lgrad[N-1] .* gate(x[N],s[N])
      gradients[N] .+= (1 / (localnorm[N] * prob)) .* grads[N]
    end
  end
  for g in gradients
    g .= -2/size(data)[1] .* g
  end
  return gradients,loss 
end

"""
Compute the total gradients
"""
function gradients(M::Union{MPS,MPO},data::Array;localnorm=nothing,choi::Bool=false)
  g_logZ,logZ = gradlogZ(M,localnorm=localnorm)
  g_nll, nll  = gradnll(M,data,localnorm=localnorm,choi=choi)
  grads = g_logZ + g_nll
  loss = logZ + nll
  return grads,loss
end



"""
Run QST
"""
function statetomography(data::Array,opt::Optimizer; kwargs...)
  N = size(data)[2]
  mixed::Bool = get(kwargs,:mixed,false)
  χ::Int64    = get(kwargs,:χ,10)
  d::Int64    = get(kwargs,:d,2)
  seed::Int64 = get(kwargs,:seed,1234)
  σ::Float64  = get(kwargs,:σ,0.1)
  
  if !mixed
    M0 = initializetomography(N,χ;d=d,seed=seed,σ=σ)
  else
    ξ::Int64 = get(kwargs,:ξ,2)
    M0 = initializetomography(N,χ,ξ;d=d,seed=seed,σ=σ)
  end 

  M = statetomography(M0,data,opt; kwargs...)
  return M
end

function processtomography(data_in::Array,data_out::Array,opt::Optimizer; kwargs...)
  N = size(data_in)[2]
  @assert size(data_in) == size(data_out)
  
  data = Matrix{String}(undef, size(data_in)[1],2*N)
  
  for n in 1:size(data_in)[1]
    for j in 1:N
      data[n,2*j-1] = data_in[n,j]
      data[n,2*j]   = data_out[n,j]
    end
  end
  return statetomography(data,opt; choi=true,kwargs...)
end

"""
Run QST
"""
function statetomography(model::Union{MPS,MPO},data::Array,opt::Optimizer; kwargs...)
                         
  localnorm::Bool = get(kwargs,:localnorm,true)
  globalnorm::Bool = get(kwargs,:globalnorm,false)
  batchsize::Int64 = get(kwargs,:batchsize,500)
  epochs::Int64 = get(kwargs,:epochs,1000)
  target = get(kwargs,:target,nothing)
  choi::Bool = get(kwargs,:choi,false)
  


  data = "state" .* data
  
  if (localnorm && globalnorm)
    error("Both input norms are set to true")
  end
  
  model = copy(model)
  if !isnothing(target)
    target = copy(target)
    if typeof(target)==MPS
      for j in 1:length(model)
        replaceind!(target[j],firstind(target[j],"Site"),firstind(model[j],"Site"))
      end
    else
      for j in 1:length(model)
        replaceind!(target[j],inds(target[j],"Site")[1],firstind(model[j],"Site"))
        replaceind!(target[j],inds(target[j],"Site")[2],prime(firstind(model[j],"Site")))
      end
    end
  end
  
  num_batches = Int(floor(size(data)[1]/batchsize))
  
  tot_time = 0.0
  for ep in 1:epochs
    ep_time = @elapsed begin
  
    data = data[shuffle(1:end),:]
    
    avg_loss = 0.0
    for b in 1:num_batches
      batch = data[(b-1)*batchsize+1:b*batchsize,:]
      
      if localnorm
        model_norm = copy(model)
        logZ,localnorms = lognormalize!(model_norm; choi=choi) 
        grads,loss = gradients(model_norm,batch,localnorm=localnorms,choi=choi)
      elseif globalnorm
        lognormalize!(model; choi=choi)
        grads,loss = gradients(model,batch,choi=choi)
      else
        grads,loss = gradients(model,batch,choi=choi)
      end
      avg_loss += loss/Float64(num_batches)
      update!(model,grads,opt)
    end

    end # end @elapsed

    print("Ep = $ep  ")
    @printf("Loss = %.5E  ",avg_loss)
    if !isnothing(target)
      if choi==true && typeof(target)==MPO
        F = fullfidelity(model,target)
      else
        F = fidelity(model,target,choi=choi)
      end
      @printf("Fidelity = %.3E  ",F)
    end
    @printf("Time = %.3f sec",ep_time)
    print("\n")

    tot_time += ep_time
  end
  @printf("Total Time = %.3f sec",tot_time)
  lognormalize!(model,choi=choi)
  return model 
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
  tmp = dag(lpdo[1]) * noprime(lpdo[1])
  Cdn = combiner(inds(tmp,tags="Link"),tags="Link,l=1")
  push!(M,tmp * Cdn)
  
  for j in 2:N-1
    prime!(lpdo[j],tags="Site")
    prime!(lpdo[j],tags="Link")
    tmp = dag(lpdo[j]) * noprime(lpdo[j]) 
    Cup = Cdn
    Cdn = combiner(inds(tmp,tags="Link,l=$j"),tags="Link,l=$j")
    push!(M,tmp * Cup * Cdn)
  end
  prime!(lpdo[N],tags="Site")
  prime!(lpdo[N],tags="Link")
  tmp = dag(lpdo[N]) * noprime(lpdo[N])
  Cup = Cdn
  push!(M,tmp * Cdn)
  rho = MPO(M)
  noprime!(lpdo)
  return rho
end

function fidelity(psi::MPS,target::MPS;choi::Bool=false)
  psi_eval = copy(psi)
  lognormalize!(psi_eval,choi=choi)
  fidelity = abs2(inner(psi_eval,target))
  if choi
    #@show norm(target)^2
    #@show 2^(0.5*length(psi_eval))
    #@assert norm(target)^2 ≈ 2^(0.5*length(psi_eval)) 
    fidelity = fidelity / (2^(length(psi_eval)))
  end
  return fidelity
end

function fidelity(lpdo::MPO,target::MPS;choi::Bool=false)
  lpdo_eval = copy(lpdo)
  lognormalize!(lpdo_eval)
  @assert norm(lpdo_eval) ≈ 1
  A = *(lpdo_eval,target,method="densitymatrix",cutoff=1e-10)
  fidelity = abs(inner(A,A))
  if choi
    fidelity = fidelity /(2^(2*length(psi)))
  end
  return fidelity
end

function fullfidelity(ρ::MPO,σ::MPO;choi::Bool=false)
  @assert length(ρ) < 12
  ρ_mat = fullmatrix(getdensityoperator(ρ))
  σ_mat = fullmatrix(σ)
  
  ρ_mat ./= tr(ρ_mat)
  σ_mat ./= tr(σ_mat)
  
  F = sqrt(ρ_mat) * (σ_mat * sqrt(ρ_mat))
  F = real(tr(sqrt(F)))
  return F
end

"""
Negative log likelihood for MPS
"""
function nll(psi::MPS,data::Array;choi::Bool=false)
  N = length(psi)
  @assert N==size(data)[2]
  loss = 0.0
  s = siteinds(psi)
  if choi
    for n in 1:size(data)[1]
      x = data[n,:]
      psix = dag(psi[1]) * dag(gate(x[1],s[1]))
      for j in 2:N
        if iseven(j)
          psi_r = dag(psi[j]) * gate(x[j],s[j])
        else
          psi_r = dag(psi[j]) * dag(gate(x[j],s[j]))
        end
        psix = psix * psi_r
      end
      prob = abs2(psix[])
      loss -= log(prob)/size(data)[1]
    end

  else
    for n in 1:size(data)[1]
      x = data[n,:]
      psix = dag(psi[1]) * gate(x[1],s[1])
      for j in 2:N
        psi_r = dag(psi[j]) * gate(x[j],s[j])
        psix = psix * psi_r
      end
      prob = abs2(psix[])
      loss -= log(prob)/size(data)[1]
    end
  end
  return loss
end

"""
Negative log likelihood for LPDO
"""
function nll(lpdo::MPO,data::Array;choi::Bool=false)
  N = length(lpdo)
  loss = 0.0
  s = Index[]
  for j in 1:N
    push!(s,firstind(lpdo[j],"Site"))
  end
  
  if choi
    for n in 1:size(data)[1]
      x = data[n,:]
      prob = prime(lpdo[1],"Link") * gate(x[1],s[1])
      prob = prob * dag(lpdo[1]) * dag(gate(x[1],s[1]))
      for j in 2:N
        if iseven(j)
          prob = prob * prime(lpdo[j],"Link") * dag(gate(x[j],s[j]))
          prob = prob * dag(lpdo[j]) * gate(x[j],s[j])
        else
          prob = prob * prime(lpdo[j],"Link") * gate(x[j],s[j])
          prob = prob * dag(lpdo[j]) * dag(gate(x[j],s[j]))
        end
      end
      loss -= log(real(prob[]))/size(data)[1]
    end
  else
    for n in 1:size(data)[1]
      x = data[n,:]
      prob = prime(lpdo[1],"Link") * dag(gate(x[1],s[1]))
      prob = prob * dag(lpdo[1]) * gate(x[1],s[1])
      for j in 2:N
        prob = prob * prime(lpdo[j],"Link") * dag(gate(x[j],s[j]))
        prob = prob * dag(lpdo[j]) * gate(x[j],s[j])
      end
      loss -= log(real(prob[]))/size(data)[1]
    end
  end
  return loss
end







#
##function measureonesiteop(psi::MPS,local_op::ITensor)
##  site = getsitenumber(firstind(local_op,tags="Site"))
##  orthogonalize!(psi,site)
##  if abs(1.0-norm(psi[site])) > 1E-8
##    error("MPS is not normalized, norm=$(norm(psi[site]))")
##  end
##  psi_m = copy(psi)
##  psi_m[site] = psi_m[site] * local_op
##  measurement = inner(psi,psi_m)
##  return measurement
##end
##
##function measureonesiteop(psi::MPS,opID::String)
##  N = length(psi)
##  sites = siteinds(psi)
##  measurement = Vector{Float64}(undef,N)
##  for j in 1:N
##    local_op = op(sites[j],opID)
##    measurement[j] = measureonesiteop(psi,local_op)
##  end
##  return measurement
##end
##
