"""
    random_mps(ElT::Type{<:Number},sites::Vector{<: Index},χ::Int64,σ::Float64)

Make a random MPS with bond dimension `χ` using the Hilbert spacee `sites`.
Each bulk tensor has one site index, and two link indices. The components of each 
tensor, with type `ElT`, are randomly drawn from a uniform distribution centered 
around zero, with width `σ`.
"""
function random_mps(ElT::Type{<:Number}, sites::Vector{<:Index}, χ::Int64, σ::Float64)
  d = dim(sites[1]) # Dimension of the local Hilbert space
  N = length(sites)
  links = [Index(χ; tags="Link, l=$l") for l in 1:(N - 1)]
  M = ITensor[]
  if N == 1
    rand_mat = σ * (ones(d) - 2 * rand(d))
    if ElT <: Complex
      rand_mat += im * σ * (ones(d) - 2 * rand(d))
    end
    push!(M, ITensor(rand_mat, sites[1]))
    return MPS(M)
  end
  # Site 1 
  rand_mat = σ * (ones(d, χ) - 2 * rand(d, χ))
  if ElT <: Complex
    rand_mat += im * σ * (ones(d, χ) - 2 * rand(d, χ))
  end
  push!(M, ITensor(rand_mat, sites[1], links[1]))
  for j in 2:(N - 1)
    rand_mat = σ * (ones(χ, d, χ) - 2 * rand(χ, d, χ))
    if ElT <: Complex
      rand_mat += im * σ * (ones(χ, d, χ) - 2 * rand(χ, d, χ))
    end
    push!(M, ITensor(rand_mat, links[j - 1], sites[j], links[j]))
  end
  # Site N
  rand_mat = σ * (ones(χ, d) - 2 * rand(χ, d))
  if ElT <: Complex
    rand_mat += im * σ * (ones(χ, d) - 2 * rand(χ, d))
  end
  push!(M, ITensor(rand_mat, links[N - 1], sites[N]))
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
function random_mpo(
  ElT::Type{<:Number}, sites::Vector{<:Index}, χ::Int64, σ::Float64; processtags::Bool=false
)
  d = dim(sites[1]) # Dimension of the local Hilbert space
  #@show sites
  N = length(sites)
  links = [Index(χ; tags="Link, l=$l") for l in 1:(N - 1)]

  M = ITensor[]
  if N == 1
    rand_mat = σ * (ones(d, d) - 2 * rand(d, d))
    if ElT <: Complex
      rand_mat += im * σ * (ones(d, d) - 2 * rand(d, d))
    end
    push!(M, ITensor(rand_mat, sites[1]', sites[1]))
    return MPO(M)
  end

  # Site 1 
  rand_mat = σ * (ones(d, χ, d) - 2 * rand(d, χ, d))
  if ElT <: Complex
    rand_mat += im * σ * (ones(d, χ, d) - 2 * rand(d, χ, d))
  end
  push!(M, ITensor(rand_mat, sites[1]', links[1], sites[1]))
  for j in 2:(N - 1)
    rand_mat = σ * (ones(d, χ, d, χ) - 2 * rand(d, χ, d, χ))
    if ElT <: Complex
      rand_mat += im * σ * (ones(d, χ, d, χ) - 2 * rand(d, χ, d, χ))
    end
    push!(M, ITensor(rand_mat, sites[j]', links[j - 1], sites[j], links[j]))
  end
  # Site N
  rand_mat = σ * (ones(d, χ, d) - 2 * rand(d, χ, d))
  if ElT <: Complex
    rand_mat += im * σ * (ones(d, χ, d) - 2 * rand(d, χ, d))
  end
  push!(M, ITensor(rand_mat, sites[N]', links[N - 1], sites[N]))

  U = MPO(M)

  proc_tagcheck = (any(x -> hastags(x, "Input"), sites))
  if proc_tagcheck
    removetags!(U, "Input")
  end
  return U
end

"""
    random_lpdo(ElT::Type{<:Number},sites::Vector{<: Index},χ::Int64,ξ::Int64,σ::Float64)

Make a random LPDO with bond dimension `χ`, kraus dimension `ξ` ,and using 
the Hilbert spacee `sites`. Each bulk tensor has one site index, one kraus index,
and two link indices. The components of each tensor, with type `ElT`, are 
randomly drawn from a uniform distribution centered around zero, with width `σ`.
"""
function random_lpdo(
  ElT::Type{<:Number},
  sites::Vector{<:Index},
  χ::Int64,
  ξ::Int64,
  σ::Float64;
  purifier_tags=default_purifier_tags,
)
  d = dim(sites[1]) # Dimension of the local Hilbert space
  N = length(sites)
  links = [Index(χ; tags="Link, l=$l") for l in 1:(N - 1)]
  kraus = [Index(ξ; tags=addtags(purifier_tags, "k=$s")) for s in 1:N]

  M = ITensor[]
  if N == 1
    # Site 1 
    rand_mat = σ * (ones(d, ξ) - 2 * rand(rng, d, ξ))
    if ElT <: Complex
      rand_mat += im * σ * (ones(d, ξ) - 2 * rand(rng, d, ξ))
    end
    push!(M, ITensor(rand_mat, sites[1], kraus[1]))
    return LPDO(MPO(M))
  end

  # Site 1 
  rand_mat = σ * (ones(d, χ, ξ) - 2 * rand(d, χ, ξ))
  if ElT <: Complex
    rand_mat += im * σ * (ones(d, χ, ξ) - 2 * rand(d, χ, ξ))
  end
  push!(M, ITensor(rand_mat, sites[1], links[1], kraus[1]))
  # Site 2..N-1
  for j in 2:(N - 1)
    rand_mat = σ * (ones(d, χ, ξ, χ) - 2 * rand(d, χ, ξ, χ))
    if ElT <: Complex
      rand_mat += im * σ * (ones(d, χ, ξ, χ) - 2 * rand(d, χ, ξ, χ))
    end
    push!(M, ITensor(rand_mat, sites[j], links[j - 1], kraus[j], links[j]))
  end
  # Site N
  rand_mat = σ * (ones(d, χ, ξ) - 2 * rand(d, χ, ξ))
  if ElT <: Complex
    rand_mat += im * σ * (ones(d, χ, ξ) - 2 * rand(d, χ, ξ))
  end
  push!(M, ITensor(rand_mat, sites[N], links[N - 1], kraus[N]))

  return LPDO(MPO(M), purifier_tags)
end

"""
    random_choi(ElT::Type{<:Number},sites::Vector{<: Index},χ::Int64,ξ::Int64,σ::Float64;
                purifier_tags = default_purifier_tags)

Make a random Choi matrix with bond dimension `χ`, kraus dimension `ξ` ,and using 
the Hilbert spacee `sites`. Each bulk tensor has two site indices (corresponding 
to input and output indices, one kraus index,and two link indices. The components 
of each tensor, with type `ElT`, are randomly drawn from a uniform distribution 
centered around zero, with width `σ`.
"""
function random_choi(
  ElT::Type{<:Number},
  sites::Vector{<:Index},
  χ::Int64,
  ξ::Int64,
  σ::Float64;
  purifier_tags=default_purifier_tags,
)
  d = dim(sites[1]) # Dimension of the local Hilbert space
  N = length(sites)
  links = [Index(χ; tags="Link, l=$l") for l in 1:(N - 1)]
  kraus = [Index(ξ; tags=addtags(purifier_tags, "k=$s")) for s in 1:N]

  M = ITensor[]
  if N == 1
    # Site 1 
    rand_mat = σ * (ones(d, d, ξ) - 2 * rand(d, d, ξ))
    if ElT <: Complex
      rand_mat += im * σ * (ones(d, d, ξ) - 2 * rand(d, d, ξ))
    end
    push!(M, ITensor(rand_mat, sites[1], sites[1]', kraus[1]))
    return LPDO(choitags(MPO(M)), purifier_tags)
  end

  # Site 1 
  rand_mat = σ * (ones(d, d, χ, ξ) - 2 * rand(d, d, χ, ξ))
  if ElT <: Complex
    rand_mat += im * σ * (ones(d, d, χ, ξ) - 2 * rand(d, d, χ, ξ))
  end
  push!(M, ITensor(rand_mat, sites[1], sites[1]', links[1], kraus[1]))
  # Site 2..N-1
  for j in 2:(N - 1)
    rand_mat = σ * (ones(d, d, χ, ξ, χ) - 2 * rand(d, d, χ, ξ, χ))
    if ElT <: Complex
      rand_mat += im * σ * (ones(d, d, χ, ξ, χ) - 2 * rand(d, d, χ, ξ, χ))
    end
    push!(M, ITensor(rand_mat, sites[j], sites[j]', links[j - 1], kraus[j], links[j]))
  end
  # Site N
  rand_mat = σ * (ones(d, d, χ, ξ) - 2 * rand(d, d, χ, ξ))
  if ElT <: Complex
    rand_mat += im * σ * (ones(d, d, χ, ξ) - 2 * rand(d, d, χ, ξ))
  end
  push!(M, ITensor(rand_mat, sites[N], sites[N]', links[N - 1], kraus[N]))

  Λ = MPO(M)
  has_inputtags = (any(x -> hastags(x, "Input"), sites))
  has_outputtags = (any(x -> hastags(x, "Output"), sites))
  if !has_inputtags && !has_outputtags
    addtags!(Λ, "Input"; plev=0, tags="Qubit")
    addtags!(Λ, "Output"; plev=1, tags="Qubit")
  elseif !has_outputtags
    for j in 1:N
      replacetags!(Λ, "Input", "Output"; plev=1)
    end
  end
  noprime!(Λ)
  return LPDO(Λ, purifier_tags)
end

"""
    randomstate(N::Int64; kwargs...)

    randomstate(ElT::Type{<: Number}, N::Int64; kwargs...)

Generates a random quantum state of N qubits.

Optionally, specify an element type, such as `randomstate(Float64, 10)` for a random real state (by default it is complex).

# Arguments
  - `N`: number of qubits
  - `mixed`: if false (default), generate a random MPS; if true, generates a random LPDO
  - `alg`: algorithm used for initialization: `"rand"` initializes random tensor elements; 
    `"circuit"` initializes with a random quantum circuit (MPS only).
  - `σ`: size of the 0-centered uniform distribution in `alg="rand"`. 
  - `χ`: bond dimension of the MPS/LPDO
  - 'ξ`: kraus dimension (LPDO)
  - `normalize`: if true, return normalized state
"""
function randomstate(ElT::Type{<:Number}, N::Int64; kwargs...)
  sites = siteinds("Qubit", N)
  return randomstate(ElT, sites; kwargs...)
end

randomstate(N::Int64; kwargs...) = randomstate(ComplexF64, N; kwargs...)

function randomstate(ElT::Type{<:Number}, sites::Vector{<:Index}; kwargs...)
  mixed::Bool = get(kwargs, :mixed, false)
  lpdo::Bool = get(kwargs, :lpdo, true)
  if !mixed
    return randomstate(ElT, MPS, sites; kwargs...)
  else
    if lpdo
      return randomstate(ElT, LPDO, sites; kwargs...)
    else
      return randomstate(ElT, MPO, sites; kwargs...)
    end
  end
end

randomstate(sites::Vector{<:Index}; kwargs...) = randomstate(ComplexF64, sites; kwargs...)

function randomstate(ElT::Type{<:Number}, T::Type, sites::Vector{<:Index}; kwargs...)
  χ::Int64 = get(kwargs, :χ, 1)
  ξ::Int64 = get(kwargs, :ξ, 1)
  σ::Float64 = get(kwargs, :σ, 0.1)
  purifier_tags = get(kwargs, :purifier_tags, default_purifier_tags)
  alg::String = get(kwargs, :alg, "rand")
  normalize::Bool = get(kwargs, :normalize, false)

  if ξ > 1
    M = random_lpdo(ElT, sites, χ, ξ, σ; purifier_tags=purifier_tags)
  else
    if T == MPS
      # Build MPS by random parameter initialization
      alg == "rand" && (M = random_mps(ElT, sites, χ, σ))
      alg == "circuit" && (M = randomMPS(ElT, sites, χ))
    elseif T == MPO
      error("initialization of random MPO density matrix not yet implemented.")
    elseif T == LPDO
      M = random_lpdo(ElT, sites, χ, ξ, σ; purifier_tags=purifier_tags)
    else
      error("ansatz type not recognized")
    end
  end
  normalize && normalize!(M)
  return M
end

"""
    randomstate(M::Union{MPS,MPO,LPDO}; kwargs...)

Generate a random state with same Hilbert space (i.e. site indices)
of a reference state `M`.
"""
function randomstate(ElT::Type{<:Number}, M::Union{MPS,MPO,LPDO}; kwargs...)
  hM = hilbertspace(M)
  return randomstate(ElT, hM; kwargs...)
end

randomstate(M::Union{MPS,MPO,LPDO}; kwargs...) = randomstate(ComplexF64, M; kwargs...)

"""
    randomprocess(N::Int64; kwargs...)

    randomprocess(ElT::Type{<: Number}, N::Int64; kwargs...)

Generates a random quantum procecss of N qubits.

Optionally choose the element type with calls like `randomprocess(Float64, 10)` (by default it is complex).

# Arguments
  - `N`: number of qubits
  - `mixed`: if false (default), generates a random MPO; if true, generates a random LPDO.
  - `alg`: initialization criteria, set to `"randompars"` (see `randomstate`).
  - `σ`: size of the 0-centered uniform distribution in `alg="rand"`. 
  - `χ`: bond dimension of the MPO/LPDO.
  - 'ξ`: kraus dimension (LPDO).
"""
function randomprocess(ElT::Type{<:Number}, N::Int64; kwargs...)
  sites = siteinds("Qubit", N)
  return randomprocess(ElT, sites; kwargs...)
end

randomprocess(N::Int64; kwargs...) = randomprocess(ComplexF64, N; kwargs...)

function randomprocess(ElT::Type{<:Number}, sites::Vector{<:Index}; kwargs...)
  mixed::Bool = get(kwargs, :mixed, false)
  if mixed
    return randomprocess(ElT, LPDO, sites; kwargs...)
  end
  return randomprocess(ElT, MPO, sites; kwargs...)
end

function randomprocess(sites::Vector{<:Index}; kwargs...)
  return randomprocess(ComplexF64, sites; kwargs...)
end

function randomprocess(ElT::Type{<:Number}, T::Type, sites::Vector{<:Index}; kwargs...)
  χ::Int64 = get(kwargs, :χ, 1)
  ξ::Int64 = get(kwargs, :ξ, 1)
  σ::Float64 = get(kwargs, :σ, 0.1)
  purifier_tags = get(kwargs, :purifier_tags, default_purifier_tags)
  alg::String = get(kwargs, :alg, "rand")
  normalize::Bool = get(kwargs, :normalize, false)
  processtags = !(any(x -> hastags(x, "Input"), sites))

  if ξ > 1
    M = random_choi(ElT, sites, χ, ξ, σ; purifier_tags=purifier_tags)
  else
    if T == MPO
      if alg == "rand"
        M = random_mpo(ElT, sites, χ, σ; processtags=processtags)
      else
        error("randomMPO with circuit initialization not implemented yet")
      end
    elseif T == LPDO
      M = random_choi(ElT, sites, χ, ξ, σ; purifier_tags=purifier_tags)
    else
      error("ansatz type not recognized")
    end
  end
  if normalize
    if M isa MPO
      Φ = normalize!(unitary_mpo_to_choi_mps(M))
      Φ = Φ * √2^length(M)
      M = choi_mps_to_unitary_mpo(Φ)
    else
      normalize!(M; localnorm=2)
    end
  end
  return M
end

function randomprocess(T::Type, sites::Vector{<:Index}; kwargs...)
  return randomprocess(ComplexF64, T, sites; kwargs...)
end

"""
    randomprocess(M::Union{MPS,MPO}; kwargs...)

Generate a random process with same Hilbert space (i.e. input
and output indices)of a reference process `M`.
"""
function randomprocess(ElT::Type{<:Number}, M::Union{MPS,MPO}; kwargs...)
  mixed = get(kwargs, :mixed, false)
  N = length(M)
  s = Index[]
  for j in 1:N
    push!(s, firstind(M[j]; tags="Site", plev=0))
  end
  proc = randomprocess(ElT, s; mixed=mixed, kwargs...)
  return proc
end

randomprocess(M::Union{MPS,MPO}; kwargs...) = randomprocess(ComplexF64, M; kwargs...)

randomprocess(ElT::Type{<:Number}, L::LPDO; kwargs...) = randomprocess(ElT, L.X; kwargs...)

randomprocess(L::LPDO; kwargs...) = randomprocess(ComplexF64, L; kwargs...)
