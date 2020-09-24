
# Locally purified density operator
# L = prime(X, !purifier_tag(X)) * X†
struct LPDO{MPOT <: Union{MPS, MPO}}
  X::MPOT
  purifier_tag::TagSet
end

LPDO(X::Union{MPS, MPO}) = LPDO(X, ts"Purifier")

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

LinearAlgebra.tr(L::LPDO) = inner(L.X, L.X)

logtr(L::LPDO) = loginner(L.X, L.X)

# TODO: define siteinds, firstsiteind, allsiteind, etc.

"""
    normalize!(L::LPDO; sqrt_localnorms! = [])
    normalize!(ψ::MPS; localnorms! = [])

Normalize the MPS/LPDO and returns the log of the norm and a vector of the local norms of each site.

An MPS `|ψ⟩` is normalized by `√⟨ψ|ψ⟩`, and the resulting MPS will have the property `norm(ψ) ≈ 1`.

An LPDO `L = X X†` is normalized by `tr(L) = tr(X X†)`, so each `X` is normalized by `√tr(L) = √tr(X X†)`. The resulting LPDO will have the property `tr(L) ≈ 1`.

Passing a vector `v` as the keyword arguments `localnorms!` (`sqrt_localnorms!`) will fill the vector with the (square root) of the normalization factor per site. For an MPS `ψ`, `prod(v) ≈ norm(ψ)`, while for an LPDO `L`, `prod(v)^2 ≈ tr(L)`.
"""
function LinearAlgebra.normalize!(L::LPDO; sqrt_localnorms! = [])
  N = length(L)
  resize!(sqrt_localnorms!, N)
  # TODO: replace with:
  #blob = ket(L, 1) * δ(siteinds(L, 1)) * bra(L, 1)
  blob = noprime(ket(L, 1), "Site") * bra(L, 1)
  localZ = norm(blob)
  blob /= sqrt(localZ)
  L.X[1] /= (localZ^0.25)
  sqrt_localnorms![1] = localZ^0.25
  for j in 2:length(L)-1
    # TODO: replace with:
    # noprime(ket(L, j), siteind(L, j))
    blob = blob * noprime(ket(L, j), "Site")
    blob = blob * bra(L, j)
    localZ = norm(blob)
    blob /= sqrt(localZ)
    L.X[j] /= (localZ^0.25)
    sqrt_localnorms![j] = localZ^0.25
  end
  # TODO: replace with:
  # noprime(ket(L, N), siteind(L, N))
  blob = blob * noprime(ket(L, N), "Site")
  blob = blob * bra(L, N)
  localZ = norm(blob)
  L.X[N] /= sqrt(localZ)
  sqrt_localnorms![N] = sqrt(localZ)
  return L
end

function LinearAlgebra.normalize!(ψ::MPS; localnorms! = [])
  normalize!(LPDO(ψ); sqrt_localnorms! = localnorms!)
  return ψ
end

