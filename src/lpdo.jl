
# Locally purified density operator
# L = prime(X, !purifier_tag(X)) * X†
struct LPDO{XT <: Union{MPS, MPO}}
  X::XT
  purifier_tag::TagSet
end

#LPDO(X::MPO) = LPDO(X, ts"Purifier")
LPDO(M::MPS) = LPDO(M, ts"")

function LPDO(M::MPO)
  if length(inds(M[1])) == 3
    # unitary MPO and density matrix MPO
    (length(inds(M[1], tags="Qubit")) == 2) && return LPDO(M, ts"")
    # density matrix LPDO
    (length(inds(M[1], tags="Qubit")) == 1) && return LPDO(M, ts"Purifier")
  elseif length(inds(M[1])) == 4
    # Choi matrix LPDO
    (length(inds(M[1], tags="Qubit")) == 2) && return LPDO(M, ts"Purifier")
  elseif length(inds(M[1])) == 5
    # Choi matrix MPO
    (length(inds(M[1], tags="Qubit")) == 4) && return LPDO(M, ts"")
  else
    error("Input not recognized")
  end
end

Base.length(L::LPDO) = length(L.X)

Base.copy(L::LPDO) = LPDO(copy(L.X), L.purifier_tag)

# TODO: define this (not defined for MPS/MPO yet)
#Base.lastindex(L::LPDO) = lastindex(L.X)

function Base.getindex(L::LPDO, args...)
  error("getindex(L::LPDO, args...) is purposefully not implemented yet. For the LPDO L = X X†, you can get the jth tensor X[j] with L.X[j].")
end

function Base.setindex!(L::LPDO, args...)
  error("setindex!(L::LPDO, args...) is purposefully not implemented yet. For the LPDO L = X X†, you can set the jth tensor X[j] with L.X[j] = A.")
end

purifier_tag(L::LPDO) = L.purifier_tag

ket(L::LPDO) = prime(L.X, !purifier_tag(L))
ket(L::LPDO, j::Int) = prime(L.X[j], !purifier_tag(L))
bra(L::LPDO) = dag(L.X)
bra(L::LPDO, j::Int) = dag(L.X[j])

function Base.iterate(L::LPDO, state = 1) 
  if state > 2*length(L)
    return nothing
  elseif isodd(state)
    T = bra(L, (state+1)÷2)
  else
    T = ket(L, state÷2)
  end
  return T, state+1
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

Passing a vector `v` as the keyword arguments `localnorms!` (`sqrt_localnorms!`) will fill the vector with the (square root) of the normalization factor per site. For an MPS `ψ`, `prod(v) ≈ norm(ψ)`. For an MPO `M`, `prod(v) ≈ tr(M). For an LPDO `L`, `prod(v)^2 ≈ tr(L)`.
"""
function normalize!(M::MPO;
                    plev = 0 => 1,
                    tags = ts"" => ts"",
                    localnorms! = [])
  N = length(M)
  resize!(localnorms!, N)
  blob = tr(M[1]; plev = plev, tags = tags)
  localZ = norm(blob)
  blob /= localZ
  M[1] /= localZ
  localnorms![1] = localZ
  for j in 2:N-1
    blob *= M[j]
    blob = tr(blob; plev = plev, tags = tags)
    localZ = norm(blob)
    blob /= localZ
    M[j] /= localZ
    localnorms![j] = localZ
  end
  blob *= M[N]
  localZ = tr(blob; plev = plev, tags = tags)
  M[N] /= localZ
  localnorms![N] = localZ
  return M
end

function normalize!(L::LPDO; sqrt_localnorms! = [], localnorm=1.0)
  N = length(L)
  resize!(sqrt_localnorms!, N)
  # TODO: replace with:
  #blob = noprime(ket(L, 1) * siteind(L, 1)) * bra(L, 1)
  blob = noprime(ket(L, 1), "Site") * bra(L, 1)
  localZ = norm(blob)
  blob /= localZ
  L.X[1] /= sqrt(localZ / localnorm)
  sqrt_localnorms![1] = sqrt(localZ / localnorm)
  for j in 2:length(L)-1
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

function normalize!(ψ::MPS; localnorms! = [])
  normalize!(LPDO(ψ); sqrt_localnorms! = localnorms!)
  return ψ
end

"""
    MPO(L::LPDO)

Contract the purifier indices to get the MPO
`ρ = L.X L.X†`. This contraction is performed exactly,
in the future we will support approximate contraction.
"""
function ITensors.MPO(lpdo0::LPDO)
  lpdo = copy(lpdo0.X)
  noprime!(lpdo)
  N = length(lpdo)
  M = ITensor[]
  prime!(lpdo[1]; tags = "Site")
  prime!(lpdo[1]; tags = "Link")
  tmp = lpdo[1] * noprime(dag(lpdo[1])) 
  Cdn = combiner(commonind(tmp,lpdo[2]),commonind(tmp,lpdo[2])')
  push!(M, tmp * Cdn)

  for j in 2:N-1
    prime!(lpdo[j]; tags = "Site")
    prime!(lpdo[j]; tags = "Link")
    tmp = lpdo[j] * noprime(dag(lpdo[j]))
    Cup = Cdn
    Cdn = combiner(commonind(tmp,lpdo[j+1]),commonind(tmp,lpdo[j+1])')
    push!(M, tmp * Cup * Cdn)
  end
  prime!(lpdo[N]; tags = "Site")
  prime!(lpdo[N]; tags = "Link")
  tmp = lpdo[N] * noprime(dag(lpdo[N])) 
  Cup = Cdn
  push!(M, tmp * Cdn)
  rho = MPO(M)
  
  noprime!(lpdo)
  return rho
end

function HDF5.write(parent::Union{HDF5File,HDF5Group},
                    name::AbstractString,
                    L::LPDO)
  g = g_create(parent, name)
  attrs(g)["type"] = String(Symbol(typeof(L)))
  write(parent, "X", L.X)
end

function HDF5.read(parent::Union{HDF5File, HDF5Group},
                   name::AbstractString,
                   ::Type{LPDO{XT}}) where {XT}
  g = g_open(parent, name)
  X = read(g, "X", XT)
  return LPDO(X, ts"Purifier")
end

