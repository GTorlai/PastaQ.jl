
#
# imports.jl
#

import Base:
  isapprox,
  eltype

#
# Functions defined for ITensors.jl objects
# that may be moved to ITensors.jl
#

using ITensors: AbstractMPS

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

eltype(ψ::MPS) = ITensor
eltype(ψ::MPO) = ITensor

function promote_leaf_eltypes(ψ::AbstractMPS)::Type{<:Number}
  eltypeψ = eltype(ψ[1])
  for n in 2:length(ψ)
    eltypeψ = promote_type(eltypeψ, eltype(ψ[n]))
  end 
  return eltypeψ
end

function isapprox(x::AbstractMPS, y::AbstractMPS;
                  atol::Real = 0,
                  rtol::Real = Base.rtoldefault(promote_leaf_eltypes(x), promote_leaf_eltypes(y), atol))
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

