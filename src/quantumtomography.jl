"""
  function initializetomography(N::Int64,χ::Int64;
                                seed::Int64=1234,
                                σ::Float64=0.1)
Initialize a variational MPS for quantum tomography.

Arguments:
  - `N`: number of qubits
  - `χ`: bond dimension of the MPS
  - `seed`: seed of random number generator
  - `σ`: width of initial box distribution
"""
function initializetomography(N::Int64,χ::Int64;
                              seed::Int64=1234,
                              σ::Float64=0.1)
  rng = MersenneTwister(seed)
  d = 2

  sites = [Index(d; tags="Site, n=$s") for s in 1:N]
  links = [Index(χ; tags="Link, l=$l") for l in 1:N-1]

  M = ITensor[]
  # Site 1 
  rand_mat = σ * (ones(d,χ) - 2*rand(rng,d,χ))
  rand_mat += im * σ * (ones(d,χ) - 2*rand(rng,d,χ))
  push!(M,ITensor(rand_mat,sites[1],links[1]))
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
  function initializetomography(N::Int64,χ::Int64,ξ::Int64;
                                seed::Int64=1234,
                                σ::Float64=0.1)
Initialize a variational LPDO for quantum tomography.

Arguments:
  - `N`: number of qubits
  - `χ`: bond dimension of the LPDO
  - `ξ`: local purification dimension of the LPDO
  - `seed`: seed of random number generator
  - `σ`: width of initial box distribution
"""
function initializetomography(N::Int64,χ::Int64,ξ::Int64;
                              seed::Int64=1234,
                              σ::Float64=0.1)
  rng = MersenneTwister(seed)
  d = 2

  sites = [Index(d; tags="Site,n=$s") for s in 1:N]
  links = [Index(χ; tags="Link,l=$l") for l in 1:N-1]
  kraus = [Index(ξ; tags="Purifier,k=$s") for s in 1:N]

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
  function lognormalize!(M::Union{MPS,MPO})

Normalize a MPS/LPDO and store local normalizations:

- `Z = ⟨ψ|ψ⟩` for `ψ = M = MPS`
- `Z = Tr(ρ)` for `ρ = M M†` , `M = LPDO` 
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
  function gradlogZ(M::Union{MPS,MPO};localnorm=nothing)

Compute the gradients of the log-normalization with respect
to each MPS/MPO tensor component:

- `∇ᵢ = ∂ᵢlog⟨ψ|ψ⟩` for `ψ = M = MPS`
- `∇ᵢ = ∂ᵢlogTr(ρ)` for `ρ = M M†` , `M = LPDO`

If `localnorm=true`, rescale each gradient by the corresponding
local normalization.
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
  function gradnll(ψ::MPS, data::Array; localnorm=nothing, choi::Bool=false)

Compute the gradients of the cross-entropy between the MPS probability
distribution of the empirical data distribution for a set of projective 
measurements in different local bases. The probability of a single 
data-point `σ = (σ₁,σ₂,…)` is :

`P(σ) = |⟨σ|Û|ψ⟩|²`   

where `Û` is the depth-1 local circuit implementing the basis rotation.
The cross entropy function is

`nll ∝ -∑ᵢlog P(σᵢ)`

where `∑ᵢ` runs over the measurement data. Returns the gradients:

`∇ᵢ = - ∂ᵢ⟨log P(σ))⟩_data`

If `localnorm=true`, rescale each gradient by the corresponding
local normalization.

If `choi=true`, `ψ` correspodns to a Choi matrix `Λ=|ψ⟩⟨ψ|`.
The probability is then obtaining by transposing the input state, which 
is equivalent to take the conjugate of the eigenstate projector.

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
  function gradnll(lpdo::MPO, data::Array; localnorm=nothing, choi::Bool=false)

Compute the gradients of the cross-entropy between the LPDO probability 
distribution of the empirical data distribution for a set of projective 
measurements in different local bases. The probability of a single 
data-point `σ = (σ₁,σ₂,…)` is :

`P(σ) = ⟨σ|Û ρ Û†|σ⟩ = |⟨σ|Û M M† Û†|σ⟩ = |⟨σ|Û M`   

where `Û` is the depth-1 local circuit implementing the basis rotation.
The cross entropy function is

`nll ∝ -∑ᵢlog P(σᵢ)`

where `∑ᵢ` runs over the measurement data. Returns the gradients:

`∇ᵢ = - ∂ᵢ⟨log P(σ))⟩_data`

If `localnorm=true`, rescale each gradient by the corresponding
local normalization.

If `choi=true`, the probability is then obtaining by transposing the 
input state, which is equivalent to take the conjugate of the eigenstate projector.

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
    push!(kraus,firstind(lpdo[j],"Purifier"))
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
  function gradients(M::Union{MPS,MPO},data::Array;localnorm=nothing,choi::Bool=false)

Compute the gradients of the cost function:
`C = log(Z) - ⟨log P(σ)⟩_data`

If `choi=true`, add the Choi normalization `trace(Λ)=d^N` to the cost function.

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
  statetomography(model::Union{MPS,MPO},data::Array,opt::Optimizer; kwargs...)

Run quantum state tomography using a the starting state `model` on `data`.

Arguments:
  - `model`: starting MPS/LPDO state.
  - `data`: training data set of projective measurements.
  - `batchsize`: number of data-points used to compute one gradient iteration.
  - `epochs`: total number of full sweeps over the dataset.
  - `target`: target quantum state underlying the data
  - `choi`: if true, compute probability using Choi matrix
  - `observer`: keep track of measurements and fidelities.
  - `outputpath`: write observer on file 
"""
function statetomography(model::Union{MPS,MPO},data::Array,opt::Optimizer; kwargs...)
                         
  # Read arguments
  localnorm::Bool = get(kwargs,:localnorm,true)
  globalnorm::Bool = get(kwargs,:globalnorm,false)
  batchsize::Int64 = get(kwargs,:batchsize,500)
  epochs::Int64 = get(kwargs,:epochs,1000)
  target = get(kwargs,:target,nothing)
  choi::Bool = get(kwargs,:choi,false)
  observer = get(kwargs,:observer,nothing) 
  outputpath = get(kwargs,:fout,nothing)

  # Convert data to projetors
  data = "state" .* data
  
  if (localnorm && globalnorm)
    error("Both input norms are set to true")
  end
  
  model = copy(model)

  # Set up target quantum state
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
  
  # Number of training batches
  num_batches = Int(floor(size(data)[1]/batchsize))
  
  tot_time = 0.0

  # Training iterations
  for ep in 1:epochs
    ep_time = @elapsed begin
  
    data = data[shuffle(1:end),:]
    
    avg_loss = 0.0

    # Sweep over the data set
    for b in 1:num_batches
      batch = data[(b-1)*batchsize+1:b*batchsize,:]
      
      # Local normalization
      if localnorm
        model_norm = copy(model)
        logZ,localnorms = lognormalize!(model_norm) 
        grads,loss = gradients(model_norm,batch,localnorm=localnorms,choi=choi)
      # Global normalization
      elseif globalnorm
        lognormalize!(model)
        grads,loss = gradients(model,batch,choi=choi)
      # Unnormalized
      else
        grads,loss = gradients(model,batch,choi=choi)
      end
      avg_loss += loss/Float64(num_batches)
      update!(model,grads,opt)
    end

    end # end @elapsed
    
    # Measure
    if !isnothing(observer)
      measure!(observer,model;nll=avg_loss,target=target)
      # Save on file
      if !isnothing(outputpath)
        writeobserver(observer,outputpath; M = model)
      end
    end
    
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
  
  return (isnothing(observer) ? model : (model,observer))  
end


"""
  processtomography(M::Union{MPS,MPO},data_in::Array,data_out::Array,opt::Optimizer; kwargs...)

Run quantum process tomography on `(data_in,data_out)` using `model` as variational ansatz.

The data is reshuffled so it takes the format: `(input1,output1,input2,output2,…)`.
"""
function processtomography(M::Union{MPS,MPO},data_in::Array,data_out::Array,opt::Optimizer; kwargs...)
  N = size(data_in)[2]
  @assert size(data_in) == size(data_out)
  
  data = Matrix{String}(undef, size(data_in)[1],2*N)
  
  for n in 1:size(data_in)[1]
    for j in 1:N
      data[n,2*j-1] = data_in[n,j]
      data[n,2*j]   = data_out[n,j]
    end
  end
  return statetomography(M,data,opt; choi=true,kwargs...)
end

"""
  getdensityoperator(lpdo::MPO)

Contract the `purifier` indices to get the MPO
`ρ = lpdo lpdo†`

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

"""
  fidelity(ψ::MPS,ϕ::MPS)

Compute the fidelity between two MPS:

`F = |⟨ψ|ϕ⟩|²
"""
function fidelity(ψ::MPS,ϕ::MPS)
  log_F̃ = log(abs2(inner(ψ,ϕ)))
  log_K = 2.0 * (lognorm(ψ) + lognorm(ϕ))
  fidelity = exp(log_F̃ - log_K)
  return fidelity
end
"""
  fidelity(ψ::MPS,ρ::MPO)
  fidelity(ρ::MPO,ψ::MPS)

Compute the fidelity between an MPS and LPDO.

`F = ⟨ψ|ρ|ψ⟩
"""
function fidelity(ψ::MPS,ρ::MPO)
  islpdo = any(x -> any(y -> hastags(y, "Purifier"), inds(x)), ρ)
  if islpdo 
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

fidelity(ρ::MPO,ψ::MPS) = fidelity(ψ::MPS,ρ::MPO)

"""
  fullfidelity(ρ::MPO,σ::MPO;choi::Bool=false)

Compute the full quantum fidelity between two density operatos
by full enumeration.

"""
function fullfidelity(ρ::MPO,σ::MPO)
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
  nll(ψ::MPS,data::Array;choi::Bool=false)

Compute the negative log-likelihood using an MPS ansatz
over a dataset `data`:

`nll ∝ -∑ᵢlog P(σᵢ)`

If `choi=true`, the probability is then obtaining by transposing the 
input state, which is equivalent to take the conjugate of the eigenstate projector.

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
  nll(lpdo::MPO,data::Array;choi::Bool=false)
Compute the negative log-likelihood using an LPDO ansatz
over a dataset `data`:

`nll ∝ -∑ᵢlog P(σᵢ)`

If `choi=true`, the probability is then obtaining by transposing the 
input state, which is equivalent to take the conjugate of the eigenstate projector.

"""
function nll(lpdo::MPO,data::Array;choi::Bool=false)
  N = length(lpdo)
  loss = 0.0
  s = firstsiteinds(lpdo)
  
  for n in 1:size(data)[1]
    x = data[n,:]

    # Project LPDO into the measurement eigenstates
    Φdag = dag(copy(lpdo))
    for j in 1:N
      Φdag[j] = (isodd(j) & choi ? Φdag[j] = Φdag[j] * dag(gate(x[j],s[j])) :
                                   Φdag[j] = Φdag[j] * gate(x[j],s[j]))
    end
    
    # Compute overlap
    prob = inner(Φdag,Φdag)
    loss -= log(real(prob))/size(data)[1]
  end
  return loss
end

