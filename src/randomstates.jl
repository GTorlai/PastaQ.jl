"""
    random_mps(ElT::Type{<:Number},sites::Vector{<: Index},χ::Int64,σ::Float64)

Make a random MPS with bond dimension `χ` using the Hilbert spacee `sites`.
Each bulk tensor has one site index, and two link indices. The components of each 
tensor, with type `ElT`, are randomly drawn from a uniform distribution centered 
around zero, with width `σ`.
"""
function random_mps(ElT::Type{<:Number},sites::Vector{<: Index},χ::Int64,σ::Float64) 
  d = 2 # Dimension of the local Hilbert space
  N = length(sites)
  links = [Index(χ; tags="Link, l=$l") for l in 1:N-1]
  M = ITensor[]
  if N == 1
    rand_mat = σ * (ones(d) - 2 * rand(d))
    if (ElT == Complex{Float64})
      rand_mat += im * σ * (ones(d) - 2 * rand(d))
    end
    push!(M, ITensor(rand_mat, sites[1]))
    return MPS(M)
  end
  # Site 1 
  rand_mat = σ * (ones(d,χ) - 2*rand(d,χ))
  if (ElT == Complex{Float64})
    rand_mat += im * σ * (ones(d,χ) - 2*rand(d,χ)) 
  end
  push!(M,ITensor(rand_mat,sites[1],links[1]))
  for j in 2:N-1
    rand_mat = σ * (ones(χ,d,χ) - 2*rand(χ,d,χ))
    if (ElT == Complex{Float64})
      rand_mat += im * σ * (ones(χ,d,χ) - 2*rand(χ,d,χ))
    end
    push!(M,ITensor(rand_mat,links[j-1],sites[j],links[j]))
  end
  # Site N
  rand_mat = σ * (ones(χ,d) - 2*rand(χ,d))
  if (ElT == Complex{Float64})
    rand_mat += im * σ * (ones(χ,d) - 2*rand(χ,d))
  end
  push!(M,ITensor(rand_mat,links[N-1],sites[N]))
  return MPS(M)
end


"""
    random_mpo(ElT::Type{<:Number},sites::Vector{<: Index},χ::Int64,σ::Float64;
               processtags::Bool=false)

Make a random MPO with bond dimension `χ` using the Hilbert spacee `sites`.
Each bulk tensor has two site indices, and two link indices. The components of each 
tensor, with type `ElT`, are randomly drawn from a uniform distribution centered 
around zero, with width `σ`.
If `processtags=true`, add the tag `input` to the bra, and the tag `output` 
to the ket.
"""
function random_mpo(ElT::Type{<:Number},sites::Vector{<: Index},χ::Int64,σ::Float64;processtags::Bool=false)
  d = 2 # Dimension of the local Hilbert space
  #@show sites
  N = length(sites)
  links = [Index(χ; tags="Link, l=$l") for l in 1:N-1]
  
  M = ITensor[]
  if N == 1
    rand_mat = σ * (ones(d,d) - 2 * rand(d,d))
    if (ElT == Complex{Float64})
      rand_mat += im * σ * (ones(d,d) - 2 * rand(d,d))
    end
    push!(M, ITensor(rand_mat, sites[1]',sites[1]))
    return MPO(M)
  end

  # Site 1 
  rand_mat = σ * (ones(d,χ,d) - 2*rand(d,χ,d))
  if (ElT == Complex{Float64})
    rand_mat += im * σ * (ones(d,χ,d) - 2*rand(d,χ,d))
  end
  push!(M,ITensor(rand_mat,sites[1]',links[1],sites[1]))
  for j in 2:N-1
    rand_mat = σ * (ones(d,χ,d,χ) - 2*rand(d,χ,d,χ))
    if (ElT == Complex{Float64})
      rand_mat += im * σ * (ones(d,χ,d,χ) - 2*rand(d,χ,d,χ))
    end
    push!(M,ITensor(rand_mat,sites[j]',links[j-1],sites[j],links[j]))
  end
  # Site N
  rand_mat = σ * (ones(d,χ,d) - 2*rand(d,χ,d))
  if (ElT == Complex{Float64})
    rand_mat += im * σ * (ones(d,χ,d) - 2*rand(d,χ,d))
  end
  push!(M,ITensor(rand_mat,sites[N]',links[N-1],sites[N]))
 
  proc_tagcheck = (any(x -> hastags(x,"Input") , sites))
  if processtags & !proc_tagcheck
    U = MPO(M)
    addtags!(U, "Input", plev = 0, tags = "Qubit")
    addtags!(U, "Output", plev = 1, tags = "Qubit")
    return U
  else
    return MPO(M)
  end
end


"""
    random_lpdo(ElT::Type{<:Number},sites::Vector{<: Index},χ::Int64,ξ::Int64,σ::Float64)

Make a random LPDO with bond dimension `χ`, kraus dimension `ξ` ,and using 
the Hilbert spacee `sites`. Each bulk tensor has one site index, one kraus index,
and two link indices. The components of each tensor, with type `ElT`, are 
randomly drawn from a uniform distribution centered around zero, with width `σ`.
"""
function random_lpdo(ElT::Type{<:Number},sites::Vector{<: Index},χ::Int64,ξ::Int64,σ::Float64;
                    purifier_tag = ts"Purifier")
  d = 2 # Dimension of the local Hilbert space
  N = length(sites)
  links = [Index(χ; tags="Link, l=$l") for l in 1:N-1]
  kraus = [Index(ξ; tags=addtags(purifier_tag, "k=$s")) for s in 1:N]

  M = ITensor[]
  if N == 1
    # Site 1 
    rand_mat = σ * (ones(d,ξ) - 2*rand(rng,d,ξ))
    if (ElT == Complex{Float64})
      rand_mat += im * σ * (ones(d,ξ) - 2*rand(rng,d,ξ))
    end
    push!(M,ITensor(rand_mat,sites[1],kraus[1]))
    return LPDO(MPO(M))
  end

  # Site 1 
  rand_mat = σ * (ones(d,χ,ξ) - 2*rand(d,χ,ξ))
  if (ElT == Complex{Float64})
    rand_mat += im * σ * (ones(d,χ,ξ) - 2*rand(d,χ,ξ))
  end
  push!(M,ITensor(rand_mat,sites[1],links[1],kraus[1]))
  # Site 2..N-1
  for j in 2:N-1
    rand_mat = σ * (ones(d,χ,ξ,χ) - 2*rand(d,χ,ξ,χ))
    if (ElT == Complex{Float64})
      rand_mat += im * σ * (ones(d,χ,ξ,χ) - 2*rand(d,χ,ξ,χ))
    end
    push!(M,ITensor(rand_mat,sites[j],links[j-1],kraus[j],links[j]))
  end
  # Site N
  rand_mat = σ * (ones(d,χ,ξ) - 2*rand(d,χ,ξ))
  if (ElT == Complex{Float64})
    rand_mat += im * σ * (ones(d,χ,ξ) - 2*rand(d,χ,ξ))
  end
  push!(M,ITensor(rand_mat,sites[N],links[N-1],kraus[N]))
  
  return LPDO(MPO(M), purifier_tag)
end

"""
    random_choi(ElT::Type{<:Number},sites::Vector{<: Index},χ::Int64,ξ::Int64,σ::Float64;
                purifier_tag = ts"Purifier")

Make a random Choi matrix with bond dimension `χ`, kraus dimension `ξ` ,and using 
the Hilbert spacee `sites`. Each bulk tensor has two site indices (corresponding 
to input and output indices, one kraus index,and two link indices. The components 
of each tensor, with type `ElT`, are randomly drawn from a uniform distribution 
centered around zero, with width `σ`.
"""
function random_choi(ElT::Type{<:Number},sites::Vector{<: Index},χ::Int64,ξ::Int64,σ::Float64;
                    purifier_tag = ts"Purifier")
  d = 2 # Dimension of the local Hilbert space
  N = length(sites)
  links = [Index(χ; tags="Link, l=$l") for l in 1:N-1]
  kraus = [Index(ξ; tags=addtags(purifier_tag, "k=$s")) for s in 1:N]

  M = ITensor[]
  if N == 1
    # Site 1 
    rand_mat = σ * (ones(d,d,ξ) - 2*rand(d,d,ξ))
    if (ElT == Complex{Float64})
      rand_mat += im * σ * (ones(d,d,ξ) - 2*rand(d,d,ξ))
    end
    push!(M,ITensor(rand_mat,sites[1],sites[1]',kraus[1]))
    return LPDO(MPO(M))
  end

  # Site 1 
  rand_mat = σ * (ones(d,d,χ,ξ) - 2*rand(d,d,χ,ξ))
  if (ElT == Complex{Float64})
    rand_mat += im * σ * (ones(d,d,χ,ξ) - 2*rand(d,d,χ,ξ))
  end
  push!(M,ITensor(rand_mat,sites[1],sites[1]',links[1],kraus[1]))
  # Site 2..N-1
  for j in 2:N-1
    rand_mat = σ * (ones(d,d,χ,ξ,χ) - 2*rand(d,d,χ,ξ,χ))
    if (ElT == Complex{Float64})
      rand_mat += im * σ * (ones(d,d,χ,ξ,χ) - 2*rand(d,d,χ,ξ,χ))
    end
    push!(M,ITensor(rand_mat,sites[j],sites[j]',links[j-1],kraus[j],links[j]))
  end
  # Site N
  rand_mat = σ * (ones(d,d,χ,ξ) - 2*rand(d,d,χ,ξ))
  if (ElT == Complex{Float64})
    rand_mat += im * σ * (ones(d,d,χ,ξ) - 2*rand(d,d,χ,ξ))
  end
  push!(M,ITensor(rand_mat,sites[N],sites[N]',links[N-1],kraus[N]))
  
  Λ = MPO(M)
  # Add process tags
  addtags!(Λ, "Input", plev = 0, tags = "Qubit")
  addtags!(Λ, "Output", plev = 1, tags = "Qubit")
  noprime!(Λ)
  return LPDO(Λ)#, purifier_tag)
end



function randomstate(N::Int64; kwargs...)
  sites = siteinds("Qubit", N)
  return randomstate(sites; kwargs...)
end

function randomstate(sites::Vector{<:Index}; kwargs...)
  mixed::Bool = get(kwargs,:mixed,false)
  lpdo::Bool  = get(kwargs,:lpdo,false)
  if lpdo
    return randomstate(sites,LPDO; kwargs...)
  else
    if !mixed
      return randomstate(sites,MPS; kwargs...)
    else
      return randomstate(sites,MPO; kwargs...)
    end
  end
end

function randomstate(sites::Vector{<:Index},T::Type; kwargs...)
  χ::Int64 = get(kwargs,:χ,2)
  ξ::Int64 = get(kwargs,:ξ,2)
  σ::Float64 = get(kwargs,:σ,0.1)
  purifier_tag = get(kwargs,:purifier_tag,ts"Purifier")
  init::String = get(kwargs,:init,"randpars")
  cplx::Bool = get(kwargs,:complex,true)
  normalize::Bool = get(kwargs,:normalize,false)

  ElT = (cplx ? Complex{Float64} : Float64)

  if T == MPS
    # Build MPS by random parameter initialization
    if init == "randpars"
      M =  random_mps(ElT,sites,χ,σ)
    # Build MPS using a quantum circuit
    elseif init=="circuit"
      M = randomMPS(ElT,sites,χ)
    end
  elseif T == MPO
    if init == "randpars"
      M = random_mpo(ElT,sites,χ,σ)
    else
      error("randomMPO with circuit initialization not implemented yet")
    end
  elseif T == LPDO
    M = random_lpdo(ElT,sites,χ,ξ,σ;purifier_tag=purifier_tag)
  else
    error("ansatz type not recognized")
  end
  return (normalize ? normalize!(M) : M)
end

function randomstate(M::Union{MPS,MPO,LPDO}; kwargs...)
  N = length(M)
  state = randomstate(N;kwargs...)
  replacehilbertspace!(state,M)
  return state
end


function randomprocess(N::Int64; kwargs...)
  sites = siteinds("Qubit", N)
  return randomprocess(sites; kwargs...)
end

function randomprocess(sites::Vector{<:Index}; kwargs...)
  mixed::Bool = get(kwargs,:mixed,false)
  lpdo::Bool = get(kwargs,:lpdo,false)
  return ((mixed | lpdo) ? randomprocess(sites,LPDO; kwargs...) : randomprocess(sites,MPO; kwargs...))
end

function randomprocess(sites::Vector{<:Index},T::Type; kwargs...)
  χ::Int64 = get(kwargs,:χ,2)
  ξ::Int64 = get(kwargs,:ξ,2)
  σ::Float64 = get(kwargs,:σ,0.1)
  purifier_tag = get(kwargs,:purifier_tag,ts"Purifier")
  init::String = get(kwargs,:init,"randpars")
  cplx::Bool = get(kwargs,:complex,true)
  
  ElT = (cplx ? Complex{Float64} : Float64)
  
  processtags = !(any(x -> hastags(x,"Input") , sites))
  if T == MPO
    if init == "randpars"
      M = random_mpo(ElT,sites,χ,σ;processtags=processtags)
    else
      error("randomMPO with circuit initialization not implemented yet")
    end
  elseif T == LPDO
    M = random_choi(ElT,sites,χ,ξ,σ;purifier_tag=purifier_tag)
  else
    error("ansatz type not recognized")
  end
  return M
end

function randomprocess(M::Union{MPS,MPO,LPDO}; kwargs...)
  N = length(M)
  proc = randomprocess(N;kwargs...)
  replacehilbertspace!(proc,M)
  return proc
end


