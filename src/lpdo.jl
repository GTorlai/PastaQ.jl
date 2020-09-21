
# Locally purified density operator
# L = prime(X, !purifier_tag(X)) * X†
struct LPDO{MPOT <: Union{MPS, MPO}}
  X::MPOT
  purifier_tag::TagSet
end

LPDO(X::Union{MPS, MPO}) = LPDO(X, ts"purifier")

Base.length(L::LPDO) = length(L.X)
Base.setindex!(L::LPDO, args...) = setindex!(L.X, args...)

# TODO: define this (not defined for MPS/MPO yet)
#Base.lastindex(L::LPDO) = lastindex(L.X)

# TODO: decide if this is a good definition
#Base.getindex(L::LPDO, args...) = getindex(L.X, args...)

purifier_tag(L::LPDO) = L.purifier_tag

ket(L::LPDO) = prime(L.X, !purifier_tag(L))
ket(L::LPDO, j::Int) = prime(L.X[j], !purifier_tag(L))
bra(L::LPDO) = dag(L.X)
bra(L::LPDO, j::Int) = dag(L.X[j])

# TODO: define siteinds, firstsiteind, allsiteind, etc.

