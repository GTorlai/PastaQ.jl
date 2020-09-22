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
  
  return MPS(M)
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
  kraus = [Index(ξ; tags="purifier,k=$s") for s in 1:N]

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
  
  return MPO(M)
end


"""
    lognormalize!(L::LPDO)
    lognormalize!(M::MPS)

Normalize the MPS/LPDO and returns the log of the norm and a vector of the local norms of each site.
"""
function lognormalize!(L::LPDO)
  N = length(L)
  localnorms = []
  # TODO: replace with:
  # noprime(ket(L, 1), siteind(L, 1))
  blob = noprime(ket(L, 1), "Site") * bra(L, 1)
  localZ = norm(blob)
  logZ = 0.5 * log(localZ)
  blob /= sqrt(localZ)
  L.X[1] /= (localZ^0.25)
  push!(localnorms, localZ^0.25)
  for j in 2:length(L)-1
    # TODO: replace with:
    # noprime(ket(L, j), siteind(L, j))
    blob = blob * noprime(ket(L, j), "Site")
    blob = blob * bra(L, j)
    localZ = norm(blob)
    logZ += 0.5*log(localZ)
    blob /= sqrt(localZ)
    L.X[j] /= (localZ^0.25)
    push!(localnorms, localZ^0.25)  
  end
  # TODO: replace with:
  # noprime(ket(L, N), siteind(L, N))
  blob = blob * noprime(ket(L, N), "Site")
  blob = blob * bra(L, N)
  localZ = norm(blob)
  L.X[length(L)] /= sqrt(localZ)
  push!(localnorms, sqrt(localZ))
  logZ += log(real(blob[]))
  return logZ, localnorms
end

lognormalize!(M::MPS) = lognormalize!(LPDO(M))

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
function gradnll(ψ::MPS, data::Array; localnorm=nothing, choi::Bool=false)
  N = length(ψ)

  s = siteinds(ψ)

  links = [linkind(ψ, n) for n in 1:N-1]

  ElT = eltype(ψ[1])

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

  ψdag = dag(ψ)

  gradients = [[ITensor(ElT, inds(ψ[j])) for j in 1:N] for _ in 1:nthreads]

  grads = [[ITensor(ElT, undef, inds(ψ[j])) for j in 1:N] for _ in 1:nthreads]

  loss = zeros(nthreads)
 
  Threads.@threads for n in 1:size(data)[1]

    nthread = Threads.threadid()

    x = data[n,:] 
    
    """ LEFT ENVIRONMENTS """
    if choi
      L[nthread][1] .= ψdag[1] .* dag(gate(x[1],s[1]))
    else
      L[nthread][1] .= ψdag[1] .* gate(x[1],s[1])
    end
    for j in 2:N-1
      Lpsi[nthread][j] .= L[nthread][j-1] .* ψdag[j]
      if isodd(j) & choi
        L[nthread][j] .= Lpsi[nthread][j] .* dag(gate(x[j],s[j]))
      else
        L[nthread][j] .= Lpsi[nthread][j] .* gate(x[j],s[j])
      end
    end
    Lpsi[nthread][N] .= L[nthread][N-1] .* ψdag[N]
    ψx = (Lpsi[nthread][N] * gate(x[N],s[N]))[]
    prob = abs2(ψx)
    loss[nthread] -= log(prob)/size(data)[1]
    
    """ RIGHT ENVIRONMENTS """
    R[nthread][N] .= ψdag[N] .* gate(x[N],s[N])
    for j in reverse(2:N-1)
      Rpsi[nthread][j] .= ψdag[j] .* R[nthread][j+1]
      if isodd(j) & choi
        R[nthread][j] .= Rpsi[nthread][j] .* dag(gate(x[j],s[j]))
      else
        R[nthread][j] .= Rpsi[nthread][j] .* gate(x[j],s[j])
      end
    end

    """ GRADIENTS """
    # TODO: fuse into one call to mul!
    if choi
      grads[nthread][1] .= dag(gate(x[1],s[1])) .* R[nthread][2]
    else
      grads[nthread][1] .= gate(x[1],s[1]) .* R[nthread][2]
    end
    gradients[nthread][1] .+= (1 / (localnorm[1] * ψx)) .* grads[nthread][1]
    for j in 2:N-1
      if isodd(j) & choi
        Rpsi[nthread][j] .= L[nthread][j-1] .* dag(gate(x[j],s[j]))
      else
        Rpsi[nthread][j] .= L[nthread][j-1] .* gate(x[j],s[j])
      end
        
      # TODO: fuse into one call to mul!
      grads[nthread][j] .= Rpsi[nthread][j] .* R[nthread][j+1]
      gradients[nthread][j] .+= (1 / (localnorm[j] * ψx)) .* grads[nthread][j]
    end
    grads[nthread][N] .= L[nthread][N-1] .* gate(x[N], s[N])
    gradients[nthread][N] .+= (1 / (localnorm[N] * ψx)) .* grads[nthread][N]
  end
  
  for nthread in 1:nthreads
    for g in gradients[nthread]
      g .= (-2/size(data)[1]) .* g
    end
  end

  gradients_tot = [ITensor(ElT, inds(ψ[j])) for j in 1:N]
  loss_tot = 0.0
  for nthread in 1:nthreads
    gradients_tot .+= gradients[nthread]
    loss_tot += loss[nthread]
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
    push!(kraus,firstind(lpdo[j],"purifier"))
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
    if choi
      T[1] .= lpdo[1] .* gate(x[1],s[1])
      L[1] .= prime(T[1],"Link") .* dag(T[1])
    else
      T[1] .= lpdo[1] .* dag(gate(x[1],s[1]))
      L[1] .= prime(T[1],"Link") .* dag(T[1])
    end
    for j in 2:N-1
      if isodd(j) & choi
        T[j] .= lpdo[j] .* gate(x[j],s[j])
      else
        T[j] .= lpdo[j] .* dag(gate(x[j],s[j]))
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
    if choi
      Tp[1] .= prime(lpdo[1],"Link") .* gate(x[1],s[1])
      Agrad[1] .=  Tp[1] .* dag(gate(x[1],s[1]))
    else
      Tp[1] .= prime(lpdo[1],"Link") .* dag(gate(x[1],s[1]))
      Agrad[1] .=  Tp[1] .* gate(x[1],s[1])
    end
    grads[1] .= R[2] .* Agrad[1]
    gradients[1] .+= (1 / (localnorm[1] * prob)) .* grads[1]
    for j in 2:N-1
      if isodd(j) & choi
        Tp[j] .= prime(lpdo[j],"Link") .* gate(x[j],s[j])
        Lgrad[j-1] .= L[j-1] .* Tp[j]
        Agrad[j] .= Lgrad[j-1] .* dag(gate(x[j],s[j]))
      else
        Tp[j] .= prime(lpdo[j],"Link") .* dag(gate(x[j],s[j]))
        Lgrad[j-1] .= L[j-1] .* Tp[j]
        Agrad[j] .= Lgrad[j-1] .* gate(x[j],s[j])
      end
      grads[j] .= R[j+1] .* Agrad[j] 
      gradients[j] .+= (1 / (localnorm[j] * prob)) .* grads[j]
    end
    Tp[N] .= prime(lpdo[N],"Link") .* dag(gate(x[N],s[N]))
    Lgrad[N-1] .= L[N-1] .* Tp[N]
    grads[N] .= Lgrad[N-1] .* gate(x[N],s[N])
    gradients[N] .+= (1 / (localnorm[N] * prob)) .* grads[N]
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
  loss += (choi ? -0.5 * length(M) * log(2) : 0.0)
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
        logZ,localnorms = lognormalize!(LPDO(model_norm))
        grads,loss = gradients(model_norm,batch,localnorm=localnorms,choi=choi)
      elseif globalnorm
        lognormalize!(LPDO(model))
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
      #if choi==true && typeof(target)==MPO
      #  F = fullfidelity(model,target)
      #else
      F = fidelity(model,target)
      #end
      @printf("Fidelity = %.3E  ",F)
    end
    @printf("Time = %.3f sec",ep_time)
    print("\n")

    tot_time += ep_time
  end
  @printf("Total Time = %.3f sec",tot_time)
  lognormalize!(model)
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

# Assume target is normalized
function fidelity(ψ::MPS,ϕ::MPS)
  log_F̃ = log(abs2(inner(ψ,ϕ)))
  log_K = 2.0 * (lognorm(ψ) + lognorm(ϕ))
  fidelity = exp(log_F̃ - log_K)
  return fidelity
end

function trace_mpo(M::MPO)
  N = length(M)
  L = M[1] * delta(dag(siteinds(M)[1]))
  if (N==1)
    return L
  end
  for j in 2:N
    trM = M[j] * delta(dag(siteinds(M)[j]))
    L = L * trM
  end
  return L[]
end

function fidelity(ρ::MPO,ψ::MPS;lpdo::Bool=true)
  if lpdo 
    A = *(ρ,ψ,method="densitymatrix",cutoff=1e-10)
    log_F̃ = log(abs(inner(A,A)))
    log_K = 2.0*(lognorm(ψ) + lognorm(ρ))
    fidelity = exp(log_F̃ - log_K)
  else
    log_F̃ = log(abs(inner(ψ,ρ,ψ)))
    log_K = 2.0*lognorm(ψ) + log(trace_mpo(ρ)) 
    fidelity = exp(log_F̃ - log_K)
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
function nll(ψ::MPS,data::Array;choi::Bool=false)
  N = length(ψ)
  @assert N==size(data)[2]
  loss = 0.0
  s = siteinds(ψ)
  
  for n in 1:size(data)[1]
    x = data[n,:]
    ψx = (choi ? dag(ψ[1]) * dag(gate(x[1],s[1])) :
                 dag(ψ[1]) * gate(x[1],s[1]))
    for j in 2:N
      ψ_r = (isodd(j) & choi ? ψ_r = dag(ψ[j]) * dag(gate(x[j],s[j])) :
                               ψ_r = dag(ψ[j]) * gate(x[j],s[j]))
      ψx = ψx * ψ_r
    end
    prob = abs2(ψx[])
    loss -= log(prob)/size(data)[1]
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
  
  for n in 1:size(data)[1]
    x = data[n,:]
    if choi
      prob = prime(lpdo[1],"Link") * gate(x[1],s[1])
      prob = prob * dag(lpdo[1]) * dag(gate(x[1],s[1]))
    else
      prob = prime(lpdo[1],"Link") * dag(gate(x[1],s[1]))
      prob = prob * dag(lpdo[1]) * gate(x[1],s[1])
    end
    for j in 2:N
      if isodd(j) & choi
        prob = prob * prime(lpdo[j],"Link") * gate(x[j],s[j])
        prob = prob * dag(lpdo[j]) * dag(gate(x[j],s[j]))
      else
        prob = prob * prime(lpdo[j],"Link") * dag(gate(x[j],s[j]))
        prob = prob * dag(lpdo[j]) * gate(x[j],s[j])
      end
    end
    loss -= log(real(prob[]))/size(data)[1]
  end
  return loss
end




""" OLD FUNCTIONS """


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
#
#
#function lognormalize!(M::Union{MPS,MPO}; choi::Bool=false)
#  
#  # PROCESS TOMOGRAPHY
#  if choi
#    choinorm = 2^(0.5*length(M))
#    #choi_local = 1/2^0.25 
#    choi_local = 1/2^0.25 
#
#    localnorms = []
#    blob = dag(M[1]) * prime(M[1],"Link")
#    localZ = norm(blob)
#    logZ = 0.5*log(localZ)
#    blob /= sqrt(localZ)
#    M[1] /= (choi_local * localZ^0.25)
#    push!(localnorms,localZ^0.25)
#    for j in 2:length(M)-1
#      blob = blob * dag(M[j]);
#      blob = blob * prime(M[j],"Link")
#      localZ = norm(blob)
#      logZ += 0.5*log(localZ)
#      blob /= sqrt(localZ)
#      M[j] /= (choi_local * localZ^0.25)
#      push!(localnorms,localZ^0.25)  
#    end
#    blob = blob * dag(M[length(M)]);
#    blob = blob * prime(M[length(M)],"Link")
#    localZ = norm(blob)
#    M[length(M)] /= (choi_local * sqrt(localZ))
#    push!(localnorms,sqrt(localZ))
#    logZ += log(real(blob[]))
#    logZ += log(choinorm)
#    localnorms .= localnorms .* choi_local
#  end
#end

