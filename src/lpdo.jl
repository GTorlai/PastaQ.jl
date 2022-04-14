# The default purifier tags for LPDOs
const default_purifier_tags = ts"Purifier"

# Locally purified density operator
# L = prime(X, !purifier_tags(X)) * dag(X)
struct LPDO{XT<:Union{MPS,MPO}}
  X::XT
  purifier_tags::TagSet
  function LPDO(M::Union{MPO,MPS}, purifier_tags)
    sites = siteinds(all, M)
    if any(s -> has_indpairs(s, 0 => 1), sites)
      error(
        "In `LPDO(X::MPO, purifier_tags)`, `MPO` `X` must not have pairs of primed and unprimed site indices, since the `LPDO` is interpreted as `prime(X; tags = !purifier_tags) * dag(X)`.",
      )
    end
    return new{typeof(M)}(M, TagSet(purifier_tags))
  end
end

# MPS defaults to having no purifier tag
LPDO(M::MPS) = LPDO(M, random_tags())

length(L::LPDO) = length(L.X)

copy(L::LPDO) = LPDO(copy(L.X), L.purifier_tags)

# TODO: define this (not defined for MPS/MPO yet)
#Base.lastindex(L::LPDO) = lastindex(L.X)

function getindex(L::LPDO, args...)
  return error(
    "`getindex(L::LPDO, args...)` is purposefully not implemented yet. For the LPDO `L = X X†`, you can get the jth tensor `X[j]` with `L.X[j]`.",
  )
end

function setindex!(L::LPDO, args...)
  return error(
    "`setindex!(L::LPDO, args...)` is purposefully not implemented yet. For the LPDO `L = X X†`, you can set the jth tensor `X[j]` with `L.X[j] = A`.",
  )
end

purifier_tags(L::LPDO) = L.purifier_tags

ket(L::LPDO) = prime(L.X, !purifier_tags(L))
ket(L::LPDO, j::Int) = prime(L.X[j], !purifier_tags(L))
bra(L::LPDO) = dag(L.X)
bra(L::LPDO, j::Int) = dag(L.X[j])

function Base.iterate(L::LPDO, state=1)
  if state > 2 * length(L)
    return nothing
  elseif isodd(state)
    T = bra(L, (state + 1) ÷ 2)
  else
    T = ket(L, state ÷ 2)
  end
  return T, state + 1
end

ket(L::LPDO{MPS}) = prime(L.X)
ket(L::LPDO{MPS}, j::Int) = prime(L.X[j])
bra(L::LPDO{MPS}) = dag(L.X)
bra(L::LPDO{MPS}, j::Int) = dag(L.X[j])

tr(L::LPDO) = inner(L.X, L.X)

logtr(L::LPDO) = loginner(L.X, L.X)

norm(L::LPDO{MPS}) = norm(L.X)

norm(L::LPDO{MPO}) = norm(MPO(L))

# TODO: define siteinds, firstsiteind, allsiteind, etc.

"""
    normalize!(ψ::MPS; localnorms! = [])
    normalize!(M::MPO; localnorms! = [])
    normalize!(L::LPDO; sqrt_localnorms! = [])

Normalize the MPS/MPO/LPDO and returns the log of the norm and a vector of the local norms of each site.

An MPS `|ψ⟩` is normalized by `√⟨ψ|ψ⟩`, and the resulting MPS will have the property `norm(ψ) ≈ 1`.

An MPO `M` is normalized by `tr(M)`, and the resulting MPO will have the property `tr(M) ≈ 1`.

An LPDO `L = X X†` is normalized by `tr(L) = tr(X X†)`, so each `X` is normalized by `√tr(L) = √tr(X X†)`. The resulting LPDO will have the property `tr(L) ≈ 1`.

Passing a vector `v` as the keyword arguments `localnorms!` (`sqrt_localnorms!`) will fill the vector with the (square root) of the normalization factor per site. For an MPS `ψ`, `prod(v) ≈ norm(ψ)`. For an MPO `M`, `prod(v) ≈ tr(M)`. For an LPDO `L`, `prod(v)^2 ≈ tr(L)`.
"""
function normalize!(M::MPO; plev=0 => 1, tags=ts"" => ts"", (localnorms!)=[])
  N = length(M)
  resize!(localnorms!, N)
  blob = tr(M[1]; plev=plev, tags=tags)
  localZ = norm(blob)
  blob /= localZ
  M[1] /= localZ
  localnorms![1] = localZ
  for j in 2:(N - 1)
    blob *= M[j]
    blob = tr(blob; plev=plev, tags=tags)
    localZ = norm(blob)
    blob /= localZ
    M[j] /= localZ
    localnorms![j] = localZ
  end
  blob *= M[N]
  localZ = tr(blob; plev=plev, tags=tags)
  M[N] /= localZ
  localnorms![N] = localZ
  return M
end

function normalize!(L::LPDO; (sqrt_localnorms!)=[], localnorm=1.0)
  N = length(L)
  resize!(sqrt_localnorms!, N)
  # TODO: replace with:
  #blob = noprime(ket(L, 1) * siteind(L, 1)) * bra(L, 1)
  blob = noprime(ket(L, 1), "Site") * bra(L, 1)
  localZ = norm(blob)
  blob /= localZ
  L.X[1] /= sqrt(localZ / localnorm)
  sqrt_localnorms![1] = sqrt(localZ / localnorm)
  for j in 2:(length(L) - 1)
    # TODO: replace with:
    # noprime(ket(L, j), siteind(L, j))
    blob = blob * noprime(ket(L, j), "Site")
    blob = blob * bra(L, j)
    localZ = norm(blob)
    blob /= localZ
    L.X[j] /= sqrt(localZ / localnorm)
    sqrt_localnorms![j] = sqrt(localZ / localnorm)
  end
  # TODO: replace with:
  # noprime(ket(L, N), siteind(L, N))
  blob = blob * noprime(ket(L, N), "Site")
  blob = blob * bra(L, N)
  localZ = norm(blob)
  L.X[N] /= sqrt(localZ / localnorm)
  sqrt_localnorms![N] = sqrt(localZ / localnorm)

  return L
end

function normalize!(ψ::MPS; (localnorms!)=[])
  normalize!(LPDO(ψ); (sqrt_localnorms!)=localnorms!)
  return ψ
end

"""
    MPO(L::LPDO)

Contract the purifier indices to get the MPO
`ρ = L.X L.X†`. This contraction is performed exactly,
in the future we will support approximate contraction.
"""
function MPO(L::LPDO{MPO}; kwargs...)
  X = L.X
  X′ = prime(X; tags=!purifier_tags(L))
  return *(X′, dag(X); kwargs...)
end

# TODO: implement in terms of `outer(L.X', L.X; kwargs...)
# It could also maybe just call `MPO(::LPDO{MPO})` if it
# is generic enough.
ITensors.MPO(L::LPDO{MPS}; kwargs...) = MPO(L.X; kwargs...)

"""
    tr(L::LPDO, tag::String)

Trace `"Input"` or `"Output"` qubits.
"""
function tr(L::LPDO, tag::String)
  N = length(L)

  Φ = ITensor[]

  tmp = noprime(ket(L, 1); tags=tag) * bra(L, 1)
  Cdn = combiner(commonind(tmp, L.X[2]), commonind(tmp, L.X[2]'))
  push!(Φ, tmp * Cdn)

  for j in 2:(N - 1)
    tmp = noprime(ket(L, j); tags=tag) * bra(L, j)
    Cup = Cdn
    Cdn = combiner(commonind(tmp, L.X[j + 1]), commonind(tmp, L.X[j + 1]'))
    push!(Φ, tmp * Cup * Cdn)
  end
  tmp = noprime(ket(L, N); tags=tag) * bra(L, N)
  Cup = Cdn
  push!(Φ, tmp * Cup)
  return MPO(Φ)
end

function HDF5.write(parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, L::LPDO)
  g = create_group(parent, name)
  attributes(g)["type"] = String(Symbol(typeof(L)))
  return write(parent, "state/X", L.X)
end

function HDF5.read(
  parent::Union{HDF5.File,HDF5.Group}, name::AbstractString, ::Type{LPDO{XT}}
) where {XT}
  g = open_group(parent, name)
  X = read(g, "X", XT)
  return LPDO(X, ts"Purifier")
end
