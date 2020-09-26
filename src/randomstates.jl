"""
    randomstate(N::Int64; mixed::Bool=false, kwargs...)

Generate MPS (`mixed=false`) or LPDO ('mixed=true`) with 
random parameters
"""
function randomstate(N::Int64; mixed::Bool=false, kwargs...)
  return (mixed ? randomstate(N,LPDO; kwargs...) : randomstate(N,MPS; kwargs...)) 
end

function randomstate(N::Int64,T::Type; kwargs...)
  sites = siteinds("Qubit", N)
  return randomstate(sites,T; kwargs...)
end

function randomstate(sites::Vector{<:Index},T::Type; kwargs...)
  χ::Int64 = get(kwargs,:χ,2)
  ξ::Int64 = get(kwargs,:ξ,2)
  σ::Float64 = get(kwargs,:σ,0.1)
  purifier_tag = get(kwargs,:purifier_tag,ts"Purifier")

  if T == MPS
    return randomstate(sites,χ;σ=σ)
  elseif T == LPDO
    return randomstate(sites,χ,ξ;σ=σ,purifier_tag=purifier_tag)
  else
    error("ansatz type not recognized")
  end
end

function randomstate(L::LPDO,T::Type; kwargs...)
  sites = firstsiteinds(L.X)
  return randomstate(sites,T; kwargs...)
end

randomstate(M::MPO,T::Type, kwargs...) = randomstate(LPDO(M),T; kwargs...)

#randomstate(ψ::MPS; kwargs...) = randomstate(LPDO(ψ); kwargs...)
# TODO: remove when `firstsiteinds(ψ::MPS)` is implemented 
function randomstate(ψ::MPS,T::Type; kwargs...)
  sites = siteinds(ψ)
  return randomstate(sites,T; kwargs...)
end


"""
    randomstate(sites::Vector{<:Index},χ::Int64;σ::Float64=0.1)

Generates an MPS with random parameters

# Arguments:
  - `sites`: a set of site indices (local Hilbert spaces)
  - `χ`: bond dimension of the MPS
  - `σ`: width of initial box distribution
"""
function randomstate(sites::Vector{<: Index},χ::Int64;σ::Float64 = 0.1)
  d = 2 # Dimension of the local Hilbert space
  N = length(sites)
  links = [Index(χ; tags="Link, l=$l") for l in 1:N-1]
  M = ITensor[]
  if N == 1
    rand_mat = σ * (ones(d) - 2 * rand(d))
    rand_mat += im * σ * (ones(d) - 2 * rand(d))
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
    randomstate(sites::Vector{<: Index},χ::Int64,ξ::Int64;
                σ::Float64 = 0.1,purifier_tag = ts"Purifier")

Generates an LPDO for a density operator with random parameters

# Arguments:
  - `sites`: a set of site indices (local Hilbert spaces)
  - `ξ`: kraus dimension of the LPDO 
  - `χ`: bond dimension of the LPDO
  - `σ`: width of initial box distribution
"""
function randomstate(sites::Vector{<: Index},χ::Int64,ξ::Int64;
                     σ::Float64 = 0.1,purifier_tag = ts"Purifier")
  d = 2 # Dimension of the local Hilbert space
  N = length(sites)
  links = [Index(χ; tags="Link, l=$l") for l in 1:N-1]
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
    randomprocess(N::Int64; mixed::Bool=false, kwargs...)

Generate MPO (`mixed=false`) or LPDO ('mixed=true`) with 
random parameters.
"""
function randomprocess(N::Int64;mixed::Bool=false, kwargs...)
  return (mixed ? randomprocess(N,LPDO; kwargs...) : randomprocess(N,MPO; kwargs...))
end


function randomprocess(N::Int64,T::Type; kwargs...)
  sites = siteinds("Qubit", N)
  return randomprocess(sites,T; kwargs...)
end

# TODO remove the split flag after implementing unsplit choi matrix
function randomprocess(sites::Vector{<:Index},T::Type; kwargs...)
  χ::Int64 = get(kwargs,:χ,2)
  ξ::Int64 = get(kwargs,:ξ,2)
  σ::Float64 = get(kwargs,:σ,0.1)
  split = get(kwargs,:split,true)
  purifier_tag = get(kwargs,:purifier_tag,ts"Purifier")
  if T == MPO
    return randomprocess(sites,χ;σ=σ)
  elseif T == LPDO
    if split
      return randomstate(sites,χ,ξ;σ=σ,purifier_tag=purifier_tag)
    else
      return randomprocess(sites,χ,ξ;σ=σ,purifier_tag=purifier_tag)
    end
  else
    error("ansatz type not recognized")
  end
end

function randomprocess(M::Union{MPO,LPDO},T::Type; kwargs...)
  sites = firstsiteinds(L.X)
  return randomprocess(sites,T; kwargs...)
end

# TODO: update when `firstsiteinds(ψ::MPS)` is implemented 
function randomprocess(ψ::MPS,T::Type; kwargs...)
  sites = siteinds(ψ)
  return randomprocess(sites,T; kwargs...)
end
#randomstate(ψ::MPS; kwargs...) = randomstate(LPDO(ψ); kwargs...)


function randomprocess(sites::Vector{<: Index},χ::Int64;σ::Float64 = 0.1)
  d = 2 # Dimension of the local Hilbert space
  N = length(sites)
  links = [Index(χ; tags="Link, l=$l") for l in 1:N-1]
  
  M = ITensor[]
  if N == 1
    rand_mat = σ * (ones(d,d) - 2 * rand(d,d))
    rand_mat += im * σ * (ones(d,d) - 2 * rand(d,d))
    push!(M, ITensor(rand_mat, sites[1]',sites[1]))
    return MPO(M)
  end

  # Site 1 
  rand_mat = σ * (ones(d,χ,d) - 2*rand(d,χ,d))
  rand_mat += im * σ * (ones(d,χ,d) - 2*rand(d,χ,d))
  push!(M,ITensor(rand_mat,sites[1]',links[1],sites[1]))
  for j in 2:N-1
    rand_mat = σ * (ones(d,χ,d,χ) - 2*rand(d,χ,d,χ))
    rand_mat += im * σ * (ones(d,χ,d,χ) - 2*rand(d,χ,d,χ))
    push!(M,ITensor(rand_mat,sites[j]',links[j-1],sites[j],links[j]))
  end
  # Site N
  rand_mat = σ * (ones(d,χ,d) - 2*rand(d,χ,d))
  rand_mat += im * σ * (ones(d,χ,d) - 2*rand(d,χ,d))
  push!(M,ITensor(rand_mat,sites[N]',links[N-1],sites[N]))
  
  U = MPO(M)
  addtags!(U, "Input", plev = 0, tags = "Qubit")
  addtags!(U, "Output", plev = 1, tags = "Qubit")
  
  return U
end
  
function randomprocess(sites::Vector{<: Index},χ::Int64,ξ::Int64;
                     σ::Float64 = 0.1,purifier_tag = ts"Purifier")
  d = 2 # Dimension of the local Hilbert space
  N = length(sites)
  links = [Index(χ; tags="Link, l=$l") for l in 1:N-1]
  kraus = [Index(ξ; tags=addtags(purifier_tag, "k=$s")) for s in 1:N]

  M = ITensor[]
  if N == 1
    # Site 1 
    rand_mat = σ * (ones(d,d,ξ) - 2*rand(d,d,ξ))
    rand_mat += im * σ * (ones(d,ξ) - 2*rand(d,d,ξ))
    push!(M,ITensor(rand_mat,sites[1],sites[1]',kraus[1]))
    return LPDO(MPO(M))
  end

  # Site 1 
  rand_mat = σ * (ones(d,d,χ,ξ) - 2*rand(d,d,χ,ξ))
  rand_mat += im * σ * (ones(d,d,χ,ξ) - 2*rand(d,d,χ,ξ))
  push!(M,ITensor(rand_mat,sites[1],sites[1]',links[1],kraus[1]))
  # Site 2..N-1
  for j in 2:N-1
    rand_mat = σ * (ones(d,d,χ,ξ,χ) - 2*rand(d,d,χ,ξ,χ))
    rand_mat += im * σ * (ones(d,d,χ,ξ,χ) - 2*rand(d,d,χ,ξ,χ))
    push!(M,ITensor(rand_mat,sites[j],sites[j]',links[j-1],kraus[j],links[j]))
  end
  # Site N
  rand_mat = σ * (ones(d,d,χ,ξ) - 2*rand(d,d,χ,ξ))
  rand_mat += im * σ * (ones(d,d,χ,ξ) - 2*rand(d,d,χ,ξ))
  push!(M,ITensor(rand_mat,sites[N],sites[N]',links[N-1],kraus[N]))
  
  Λ = MPO(M)
  addtags!(Λ, "Input", plev = 0, tags = "Qubit")
  addtags!(Λ, "Output", plev = 1, tags = "Qubit")
  return LPDO(MPO(M), purifier_tag)
end


