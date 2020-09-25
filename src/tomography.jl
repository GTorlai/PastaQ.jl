"""
    initializetomography(N::Int64;kwargs...)

Initialize quantum tomography given a number of qubits
"""
function initializetomography(N::Int64;kwargs...)
  d = 2 # Dimension of the local Hilbert space
  sites = siteinds("Qubit", N)
  return initializetomography(sites; kwargs...)
end

"""
    initializetomography(sites::Vector{<:Index}; kwargs...)

Initialize quantum tomography for a set of indices. 
By default, inintializes MPS, unless `ξ` is non-trivial, 
in which case it initializes a LPDO.
"""
function initializetomography(sites::Vector{<:Index}; kwargs...)
  χ = get(kwargs,:χ,2)
  ξ = get(kwargs,:ξ,nothing)
  σ = get(kwargs,:σ,0.1)
  purifier_tag = get(kwargs,:purifier_tag,ts"Purifier")

  return (isnothing(ξ) ? initializetomography(sites,χ;σ=σ) :
                         initializetomography(sites,χ,ξ;σ=σ,purifier_tag=purifier_tag))
end

"""
    initializetomography(L::LPDO; kwargs...)


Initialize quantum tomography, given a reference state,
either in MPS or MPO (LPDO) form
"""
function initializetomography(L::LPDO; kwargs...)
  sites = firstsiteinds(L.X)
  return initializetomography(sites; kwargs...)
end

# TEMPORARY FUNCTION
function initializetomography(ψ::MPS; kwargs...)
  sites = siteinds(ψ)
  return initializetomography(sites; kwargs...)
end
# Update with the above after `firstsiteinds(ψ::MPS)` is implemented
#initializetomography(ψ::MPS; kwargs...) = initializetomography(LPDO(ψ); kwargs...)

"""
    initializetomography(sites::Vector{<:Index},χ::Int64;σ::Float64=0.1)

Initialize a variational MPS for quantum tomography.

# Arguments:
  - `N`: number of qubits
  - `χ`: bond dimension of the MPS
  - `σ`: width of initial box distribution
"""
function initializetomography(sites::Vector{<:Index},χ::Int64;σ::Float64=0.1)
  d = 2 # Dimension of the local Hilbert space
  N = length(sites)
  links = [Index(χ; tags="Link, l=$l") for l in 1:N-1]

  M = ITensor[]
  if N == 1
    rand_mat = σ * (ones(d) - 2 * rand(rng,d))
    rand_mat += im * σ * (ones(d) - 2 * rand(rng,d))
    push!(M, ITensor(rand_mat, sites[1]))
    return MPS(M)
  end

  # Site 1 
  rand_mat = σ * (ones(d,χ) - 2*rand(d,χ))
  rand_mat += im * σ * (ones(d,χ) - 2*rand(d,χ))
  push!(M,ITensor(rand_mat,sites[1],links[1]))
  for j in 2:N-1
    rand_mat = σ * (ones(χ,d,χ) - 2*rand(χ,d,χ))
    rand_mat += im * σ * (ones(χ,d,χ) - 2*rand(χ,d,χ))
    push!(M,ITensor(rand_mat,links[j-1],sites[j],links[j]))
  end
  # Site N
  rand_mat = σ * (ones(χ,d) - 2*rand(χ,d))
  rand_mat += im * σ * (ones(χ,d) - 2*rand(χ,d))
  push!(M,ITensor(rand_mat,links[N-1],sites[N]))
  
  return MPS(M)
end

"""
    initializetomography(sites::Vector{<:Index},χ::Int64,ξ::Int64;
                         σ::Float64=0.1,purifier_tag = ts"Purifier")

Initialize a variational LPDO for quantum tomography.

# Arguments:
  - `N`: number of qubits
  - `χ`: bond dimension of the LPDO
  - `ξ`: local purification dimension of the LPDO
  - `σ`: width of initial box distribution
"""
function initializetomography(sites::Vector{<:Index},χ::Int64,ξ::Int64;σ::Float64=0.1,purifier_tag = ts"Purifier")
  d = 2 # Dimension of the local Hilbert space
  N = length(sites)

  links = [Index(χ; tags="Link,l=$l") for l in 1:N-1]
  kraus = [Index(ξ; tags=addtags(purifier_tag, "k=$s")) for s in 1:N]

  M = ITensor[]
  if N == 1
    # Site 1 
    rand_mat = σ * (ones(d,ξ) - 2*rand(rng,d,ξ))
    rand_mat += im * σ * (ones(d,ξ) - 2*rand(rng,d,ξ))
    push!(M,ITensor(rand_mat,sites[1],kraus[1]))
    return LPDO(MPO(M))
  end

  # Site 1 
  rand_mat = σ * (ones(d,χ,ξ) - 2*rand(d,χ,ξ))
  rand_mat += im * σ * (ones(d,χ,ξ) - 2*rand(d,χ,ξ))
  push!(M,ITensor(rand_mat,sites[1],links[1],kraus[1]))
  # Site 2..N-1
  for j in 2:N-1
    rand_mat = σ * (ones(d,χ,ξ,χ) - 2*rand(d,χ,ξ,χ))
    rand_mat += im * σ * (ones(d,χ,ξ,χ) - 2*rand(d,χ,ξ,χ))
    push!(M,ITensor(rand_mat,sites[j],links[j-1],kraus[j],links[j]))
  end
  # Site N
  rand_mat = σ * (ones(d,χ,ξ) - 2*rand(d,χ,ξ))
  rand_mat += im * σ * (ones(d,χ,ξ) - 2*rand(d,χ,ξ))
  push!(M,ITensor(rand_mat,sites[N],links[N-1],kraus[N]))
  
  return LPDO(MPO(M), purifier_tag)
end

"""
    gradlogZ(L::LPDO; sqrt_localnorms = nothing)
    gradlogZ(ψ::MPS; localnorms = nothing)

Compute the gradients of the log-normalization with respect
to each LPDO tensor component:

- `∇ᵢ = ∂ᵢlog⟨ψ|ψ⟩` for `ψ = M = MPS`
- `∇ᵢ = ∂ᵢlogTr(ρ)` for `ρ = M M†` , `M = LPDO`
"""
function gradlogZ(lpdo::LPDO; sqrt_localnorms = nothing)
  M = lpdo.X
  N = length(M)
  L = Vector{ITensor}(undef, N-1)
  R = Vector{ITensor}(undef, N)
  
  if isnothing(sqrt_localnorms)
    sqrt_localnorms = ones(N)
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
  gradients[1] = prime(M[1],"Link") * R[2]/(sqrt_localnorms[1]*Z)
  for j in 2:N-1
    gradients[j] = (L[j-1] * prime(M[j],"Link") * R[j+1])/(sqrt_localnorms[j]*Z)
  end
  gradients[N] = (L[N-1] * prime(M[N],"Link"))/(sqrt_localnorms[N]*Z)
  
  return 2*gradients,log(Z)
end

gradlogZ(ψ::MPS; localnorms = nothing) = gradlogZ(LPDO(ψ); sqrt_localnorms = localnorms)

"""
    gradnll(L::LPDO{MPS}, data::Array; sqrt_localnorms = nothing, choi::Bool = false)
    gradnll(ψ::MPS, data::Array; localnorms = nothing, choi::Bool = false)

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

If `choi=true`, `ψ` correspodns to a Choi matrix `Λ=|ψ⟩⟨ψ|`.
The probability is then obtaining by transposing the input state, which 
is equivalent to take the conjugate of the eigenstate projector.
"""
#function gradnll(ψ::MPS,
function gradnll(L::LPDO{MPS},
                 data::Array;
                 sqrt_localnorms = nothing,
                 choi::Bool = false)
  ψ = L.X
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

  if isnothing(sqrt_localnorms)
    sqrt_localnorms = ones(N)
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
    gradients[nthread][1] .+= (1 / (sqrt_localnorms[1] * ψx)) .* grads[nthread][1]
    for j in 2:N-1
      if isodd(j) & choi
        Rpsi[nthread][j] .= L[nthread][j-1] .* dag(gate(x[j],s[j]))
      else
        Rpsi[nthread][j] .= L[nthread][j-1] .* gate(x[j],s[j])
      end
        
      # TODO: fuse into one call to mul!
      grads[nthread][j] .= Rpsi[nthread][j] .* R[nthread][j+1]
      gradients[nthread][j] .+= (1 / (sqrt_localnorms[j] * ψx)) .* grads[nthread][j]
    end
    grads[nthread][N] .= L[nthread][N-1] .* gate(x[N], s[N])
    gradients[nthread][N] .+= (1 / (sqrt_localnorms[N] * ψx)) .* grads[nthread][N]
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

gradnll(ψ::MPS, data::Array; localnorms = nothing, choi::Bool = false) = 
  gradnll(LPDO(ψ), data; sqrt_localnorms = localnorms, choi = choi)

"""
    gradnll(lpdo::LPDO{MPO}, data::Array; sqrt_localnorms = nothing, choi::Bool=false)

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

If `choi=true`, the probability is then obtaining by transposing the 
input state, which is equivalent to take the conjugate of the eigenstate projector.
"""
function gradnll(L::LPDO{MPO}, data::Array;
                 sqrt_localnorms = nothing, choi::Bool = false)
  lpdo = L.X
  N = length(lpdo)

  s = firstsiteinds(lpdo)  
  
  links = [linkind(lpdo, n) for n in 1:N-1]
  
  kraus = Index[]
  for j in 1:N
    push!(kraus,firstind(lpdo[j], "Purifier"))
  end

  ElT = eltype(lpdo[1])
  
  nthreads = Threads.nthreads()

  L     = [Vector{ITensor{2}}(undef, N) for _ in 1:nthreads]
  Llpdo = [Vector{ITensor}(undef, N) for _ in 1:nthreads]
  Lgrad = [Vector{ITensor}(undef,N) for _ in 1:nthreads]

  R     = [Vector{ITensor{2}}(undef, N) for _ in 1:nthreads]
  Rlpdo = [Vector{ITensor}(undef, N) for _ in 1:nthreads]
  
  Agrad = [Vector{ITensor}(undef, N) for _ in 1:nthreads]
  
  T  = [Vector{ITensor}(undef,N) for _ in 1:nthreads] 
  Tp = [Vector{ITensor}(undef,N) for _ in 1:nthreads]
  
  grads     = [Vector{ITensor}(undef,N) for _ in 1:nthreads] 
  gradients = [Vector{ITensor}(undef,N) for _ in 1:nthreads]
  
  for nthread in 1:nthreads

    for n in 1:N-1
      L[nthread][n] = ITensor(ElT, undef, links[n]',links[n])
    end
    for n in 2:N-1
      Llpdo[nthread][n] = ITensor(ElT, undef, kraus[n],links[n]',links[n-1])
    end
    for n in 1:N-2
      Lgrad[nthread][n] = ITensor(ElT,undef,links[n],kraus[n+1],links[n+1]')
    end
    Lgrad[nthread][N-1] = ITensor(ElT,undef,links[N-1],kraus[N])

    for n in N:-1:2
      R[nthread][n] = ITensor(ElT, undef, links[n-1]',links[n-1])
    end 
    for n in N-1:-1:2
      Rlpdo[nthread][n] = ITensor(ElT, undef, links[n-1]',kraus[n],links[n])
    end
  
    Agrad[nthread][1] = ITensor(ElT, undef, kraus[1],links[1]',s[1])
    for n in 2:N-1
      Agrad[nthread][n] = ITensor(ElT, undef, links[n-1],kraus[n],links[n]',s[n])
    end

    T[nthread][1] = ITensor(ElT, undef, kraus[1],links[1])
    Tp[nthread][1] = prime(T[nthread][1],"Link")
    for n in 2:N-1
      T[nthread][n] = ITensor(ElT, undef, kraus[n],links[n],links[n-1])
      Tp[nthread][n] = prime(T[nthread][n],"Link")
    end
    T[nthread][N] = ITensor(ElT, undef, kraus[N],links[N-1])
    Tp[nthread][N] = prime(T[nthread][N],"Link")
  
    grads[nthread][1] = ITensor(ElT, undef,links[1],kraus[1],s[1])
    gradients[nthread][1] = ITensor(ElT,links[1],kraus[1],s[1])
    for n in 2:N-1
      grads[nthread][n] = ITensor(ElT, undef,links[n],links[n-1],kraus[n],s[n])
      gradients[nthread][n] = ITensor(ElT,links[n],links[n-1],kraus[n],s[n])
    end
    grads[nthread][N] = ITensor(ElT, undef,links[N-1],kraus[N],s[N])
    gradients[nthread][N] = ITensor(ElT, links[N-1],kraus[N],s[N])
  end
  
  if isnothing(sqrt_localnorms)
    sqrt_localnorms = ones(N)
  end
  
  loss = zeros(nthreads)

  Threads.@threads for n in 1:size(data)[1]

    nthread = Threads.threadid()

    x = data[n,:]
    
    """ LEFT ENVIRONMENTS """
    if choi
      T[nthread][1] .= lpdo[1] .* gate(x[1],s[1])
      L[nthread][1] .= prime(T[nthread][1],"Link") .* dag(T[nthread][1])
    else
      T[nthread][1] .= lpdo[1] .* dag(gate(x[1],s[1]))
      L[nthread][1] .= prime(T[nthread][1],"Link") .* dag(T[nthread][1])
    end
    for j in 2:N-1
      if isodd(j) & choi
        T[nthread][j] .= lpdo[j] .* gate(x[j],s[j])
      else
        T[nthread][j] .= lpdo[j] .* dag(gate(x[j],s[j]))
      end
      Llpdo[nthread][j] .= prime(T[nthread][j],"Link") .* L[nthread][j-1]
      L[nthread][j] .= Llpdo[nthread][j] .* dag(T[nthread][j])
    end
    T[nthread][N] .= lpdo[N] .* dag(gate(x[N],s[N]))
    prob = L[nthread][N-1] * prime(T[nthread][N],"Link")
    prob = prob * dag(T[nthread][N])
    prob = real(prob[])
    loss[nthread] -= log(prob)/size(data)[1]
    
    """ RIGHT ENVIRONMENTS """
    R[nthread][N] .= prime(T[nthread][N],"Link") .* dag(T[nthread][N])
    for j in reverse(2:N-1)
      Rlpdo[nthread][j] .= prime(T[nthread][j],"Link") .* R[nthread][j+1] 
      R[nthread][j] .= Rlpdo[nthread][j] .* dag(T[nthread][j])
    end
    
    """ GRADIENTS """
    if choi
      Tp[nthread][1] .= prime(lpdo[1],"Link") .* gate(x[1],s[1])
      Agrad[nthread][1] .=  Tp[nthread][1] .* dag(gate(x[1],s[1]))
    else
      Tp[nthread][1] .= prime(lpdo[1],"Link") .* dag(gate(x[1],s[1]))
      Agrad[nthread][1] .=  Tp[nthread][1] .* gate(x[1],s[1])
    end
    grads[nthread][1] .= R[nthread][2] .* Agrad[nthread][1]
    gradients[nthread][1] .+= (1 / (sqrt_localnorms[1] * prob)) .* grads[nthread][1]
    for j in 2:N-1
      if isodd(j) & choi
        Tp[nthread][j] .= prime(lpdo[j],"Link") .* gate(x[j],s[j])
        Lgrad[nthread][j-1] .= L[nthread][j-1] .* Tp[nthread][j]
        Agrad[nthread][j] .= Lgrad[nthread][j-1] .* dag(gate(x[j],s[j]))
      else
        Tp[nthread][j] .= prime(lpdo[j],"Link") .* dag(gate(x[j],s[j]))
        Lgrad[nthread][j-1] .= L[nthread][j-1] .* Tp[nthread][j]
        Agrad[nthread][j] .= Lgrad[nthread][j-1] .* gate(x[j],s[j])
      end
      grads[nthread][j] .= R[nthread][j+1] .* Agrad[nthread][j] 
      gradients[nthread][j] .+= (1 / (sqrt_localnorms[j] * prob)) .* grads[nthread][j]
    end
    Tp[nthread][N] .= prime(lpdo[N],"Link") .* dag(gate(x[N],s[N]))
    Lgrad[nthread][N-1] .= L[nthread][N-1] .* Tp[nthread][N]
    grads[nthread][N] .= Lgrad[nthread][N-1] .* gate(x[N],s[N])
    gradients[nthread][N] .+= (1 / (sqrt_localnorms[N] * prob)) .* grads[nthread][N]
  end
  
  for nthread in 1:nthreads
    for g in gradients[nthread]
      g .= (-2/size(data)[1]) .* g
    end
  end
  
  gradients_tot = Vector{ITensor}(undef,N) 
  gradients_tot[1] = ITensor(ElT,links[1],kraus[1],s[1])
  for n in 2:N-1
    gradients_tot[n] = ITensor(ElT,links[n],links[n-1],kraus[n],s[n])
  end
  gradients_tot[N] = ITensor(ElT, links[N-1],kraus[N],s[N])
  
  loss_tot = 0.0
  for nthread in 1:nthreads
    gradients_tot .+= gradients[nthread]
    loss_tot += loss[nthread]
  end
  
  return gradients_tot, loss_tot
end


"""
    gradients(L::LPDO, data::Array; sqrt_localnorms = nothing, choi::Bool = false)
    gradients(ψ::MPS, data::Array; localnorms = nothing, choi::Bool = false)

Compute the gradients of the cost function:
`C = log(Z) - ⟨log P(σ)⟩_data`

If `choi=true`, add the Choi normalization `trace(Λ)=d^N` to the cost function.
"""
function gradients(L::LPDO, data::Array;
                   sqrt_localnorms = nothing, choi::Bool = false)
  g_logZ,logZ = gradlogZ(L; sqrt_localnorms = sqrt_localnorms)
  g_nll, nll  = gradnll(L, data; sqrt_localnorms = sqrt_localnorms, choi = choi)
  
  grads = g_logZ + g_nll
  loss = logZ + nll
  loss += (choi ? -0.5 * length(L) * log(2) : 0.0)
  return grads,loss
end

gradients(ψ::MPS, data::Array; localnorms = nothing, choi::Bool = false) = 
  gradients(LPDO(ψ), data; sqrt_localnorms = localnorms, choi = choi)

"""
    tomography(L::LPDO, data::Array, opt::Optimizer; kwargs...)
    tomography(ψ::MPS, data::Array, opt::Optimizer; kwargs...)

Run quantum state tomography using a the starting state `model` on `data`.

# Arguments:
  - `model`: starting LPDO state.
  - `data`: training data set of projective measurements.
  - `batchsize`: number of data-points used to compute one gradient iteration.
  - `epochs`: total number of full sweeps over the dataset.
  - `target`: target quantum state underlying the data
  - `choi`: if true, compute probability using Choi matrix
  - `observer`: keep track of measurements and fidelities.
  - `outputpath`: write observer on file 
"""
function tomography(L::LPDO,
                    data::Array,
                    opt::Optimizer;
                    kwargs...)
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
  
  model = copy(L)
  
  # Set up target quantum state
  if !isnothing(target)
    target = copy(target)
    if typeof(target)==MPS
      for j in 1:length(model)
        replaceind!(target[j],firstind(target[j],"Site"),firstind(model.X[j],"Site"))
      end
    else
      for j in 1:length(model)
        replaceind!(target[j],inds(target[j],"Site")[1],firstind(model.X[j],"Site"))
        replaceind!(target[j],inds(target[j],"Site")[2],prime(firstind(model.X[j],"Site")))
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
        modelcopy = copy(model)
        sqrt_localnorms = []
        #logZ,localnorms = normalize!(modelcopy)
        normalize!(modelcopy; sqrt_localnorms! = sqrt_localnorms)
        grads,loss = gradients(modelcopy, batch, sqrt_localnorms = sqrt_localnorms, choi = choi)
      # Global normalization
      elseif globalnorm
        normalize!(model)
        grads,loss = gradients(model,batch,choi=choi)
      # Unnormalized
      else
        grads,loss = gradients(model,batch,choi=choi)
      end

      nupdate = ep * num_batches + b
      avg_loss += loss/Float64(num_batches)
      update!(model,grads,opt;step=nupdate)
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
      if ((model.X isa MPO) & (target isa MPO)) 
        frob_dist = frobenius_distance(model,target)
        fbound = fidelity_bound(model,target)
        @printf("Trace distance = %.3E  ",frob_dist)
        @printf("Fidelity bound = %.3E  ",fbound)
        if (length(model) <= 8)
          disable_warn_order!()
          fid = fullfidelity(model,target)
          reset_warn_order!()
          @printf("Fidelity = %.3E  ",fid)
        end
      else
        F = fidelity(model,target)
        @printf("Fidelity = %.3E  ",F)
      end
    end
    @printf("Time = %.3f sec",ep_time)
    print("\n")

    tot_time += ep_time
  end
  @printf("Total Time = %.3f sec\n",tot_time)
  normalize!(model)
  
  return (isnothing(observer) ? model : (model,observer))  
end

tomography(ψ::MPS, data::Array,opt::Optimizer; kwargs...) =
  tomography(LPDO(ψ),data,opt; kwargs...)

"""
    tomography(L::LPDO,data_in::Array,data_out::Array,opt::Optimizer; kwargs...)
    tomography(ψ::MPS,data_in::Array,data_out::Array,opt::Optimizer; kwargs...)

Run quantum process tomography on `(data_in,data_out)` using `model` as variational ansatz.

The data is reshuffled so it takes the format: `(input1,output1,input2,output2,…)`.
"""
function tomography(L::LPDO,
                    data_in::Array,
                    data_out::Array,
                    opt::Optimizer;
                    kwargs...)
  N = size(data_in)[2]
  @assert size(data_in) == size(data_out)
  
  data = Matrix{String}(undef, size(data_in)[1],2*N)
  
  for n in 1:size(data_in)[1]
    for j in 1:N
      data[n,2*j-1] = data_in[n,j]
      data[n,2*j]   = data_out[n,j]
    end
  end
  return tomography(L,data,opt; choi=true,kwargs...)
end

tomography(ψ::MPS,data_in::Array, data_out::Array,opt::Optimizer; kwargs...) =
  tomography(LPDO(ψ),data_in,data_out,opt; kwargs...)


"""
    nll(ψ::MPS,data::Array;choi::Bool=false)

Compute the negative log-likelihood using an MPS ansatz
over a dataset `data`:

`nll ∝ -∑ᵢlog P(σᵢ)`

If `choi=true`, the probability is then obtaining by transposing the 
input state, which is equivalent to take the conjugate of the eigenstate projector.
"""
function nll(L::LPDO{MPS}, data::Array; choi::Bool = false)
  ψ = L.X
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

nll(ψ::MPS, args...; kwargs...) = nll(LPDO(ψ), args...; kwargs...)

"""
    nll(lpdo::LPDO, data::Array; choi::Bool = false)

Compute the negative log-likelihood using an LPDO ansatz
over a dataset `data`:

`nll ∝ -∑ᵢlog P(σᵢ)`

If `choi=true`, the probability is then obtaining by transposing the 
input state, which is equivalent to take the conjugate of the eigenstate projector.
"""
function nll(L::LPDO{MPO}, data::Array; choi::Bool = false)
  lpdo = L.X
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

"""
    MPO(L::LPDO)

Contract the purifier indices to get the MPO
`ρ = L.X L.X†`
"""
function ITensors.MPO(lpdo0::LPDO)
  lpdo = copy(lpdo0.X)
  noprime!(lpdo)
  N = length(lpdo)
  M = ITensor[]
  prime!(lpdo[1]; tags = "Site")
  prime!(lpdo[1]; tags = "Link")
  tmp = dag(lpdo[1]) * noprime(lpdo[1])
  Cdn = combiner(inds(tmp, tags = "Link"), tags = "Link,l=1")
  push!(M, tmp * Cdn)
  
  for j in 2:N-1
    prime!(lpdo[j]; tags = "Site")
    prime!(lpdo[j]; tags = "Link")
    tmp = dag(lpdo[j]) * noprime(lpdo[j]) 
    Cup = Cdn
    Cdn = combiner(inds(tmp,tags="Link,l=$j"),tags="Link,l=$j")
    push!(M, tmp * Cup * Cdn)
  end
  prime!(lpdo[N]; tags = "Site")
  prime!(lpdo[N]; tags = "Link")
  tmp = dag(lpdo[N]) * noprime(lpdo[N])
  Cup = Cdn
  push!(M, tmp * Cdn)
  rho = MPO(M)
  noprime!(lpdo)
  return rho
end

