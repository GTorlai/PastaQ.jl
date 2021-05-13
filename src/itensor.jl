#
# Functions defined for ITensors.jl objects
# that may be moved to ITensors.jl
#

#
# imports.jl
#

import Base: isapprox, eltype

using ITensors: AbstractMPS

######################################################
# TagSet
#

function random_tags()
  ts = TagSet()
  ntags = length(ts.data)
  tagtype = eltype(ts.data)
  for n in 1:length(ts.data)
    ts = addtags(ts, rand(tagtype))
  end
  return ts
end

######################################################
# IndexSet
#

function has_indpairs(is, plevs::Pair{Int,Int})
  return any(i -> hasplev(i, first(plevs)) && setprime(i, last(plevs)) in is, is)
end

######################################################
# ITensor
#

function sqrt(ρ::ITensor)
  D, U = eigen(ρ)
  sqrtD = D
  sqrtD .= sqrt.(D)
  return U' * sqrtD * dag(U)
end

######################################################
# MPS
#

## # For |ψ⟩ and |ϕ⟩, return |ψ⟩⊗⟨ϕ|
## function ITensors.outer(ψ::MPS, ϕ::MPS; kwargs...)
##   # XXX: implement by converting to MPOs and
##   # contracting the MPOs?
##   @assert ψ == ϕ'
##   return MPO(ϕ; kwargs...)
## end

eltype(ψ::MPS) = ITensor
eltype(ψ::MPO) = ITensor

function promote_leaf_eltypes(ψ::AbstractMPS)::Type{<:Number}
  eltypeψ = eltype(ψ[1])
  for n in 2:length(ψ)
    eltypeψ = promote_type(eltypeψ, eltype(ψ[n]))
  end
  return eltypeψ
end

function isapprox(
  x::AbstractMPS,
  y::AbstractMPS;
  atol::Real=0,
  rtol::Real=Base.rtoldefault(promote_leaf_eltypes(x), promote_leaf_eltypes(y), atol),
)
  #d = norm(x - y)
  normx² = inner(x, x)
  normy² = inner(y, y)
  d² = normx² + normy² - 2 * real(inner(x, y))
  @assert imag(d²) < 1e-15
  d = sqrt(abs(d²))
  normx = sqrt(abs(normx²))
  normy = sqrt(abs(normy²))
  return d <= max(atol, rtol * max(normx, normy))
end

isapprox(x::AbstractMPS, y::ITensor; kwargs...) = isapprox(prod(x), y; kwargs...)
isapprox(x::ITensor, y::AbstractMPS; kwargs...) = isapprox(y, x; kwargs...)
