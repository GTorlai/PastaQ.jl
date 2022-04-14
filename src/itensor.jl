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
    rand_tag = rand(tagtype)
    # Before ITensors v0.2, addtags expects a Tag.
    # Afterward, just needs an integer.
    rand_tag = ITensors.version() < v"0.2" ? ITensors.Tag(rand_tag) : rand_tag
    ts = addtags(ts, rand_tag)
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

function sqrt_hermitian(ρ::ITensor; cutoff::Float64=1e-15)
  if !isapprox(swapprime(dag(ρ), 0 => 1), ρ)
    error("matrix is not hermitian")
  end
  D, U = eigen(ρ; ishermitian=true, cutoff=cutoff)
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

isapprox(x::AbstractMPS, y::ITensor; kwargs...) = isapprox(prod(x), y; kwargs...)
isapprox(x::ITensor, y::AbstractMPS; kwargs...) = isapprox(y, x; kwargs...)

function expect(T₀::ITensor, ops::AbstractString...; kwargs...)
  T = copy(T₀)
  N = nsites(T)
  ElT = real(ITensors.promote_itensor_eltype([T]))
  Nops = length(ops)

  site_range::UnitRange{Int} = get(kwargs, :site_range, 1:N)
  Ns = length(site_range)
  start_site = first(site_range)
  offset = start_site - 1

  normalization = is_operator(T) ? tr(T) : norm(T)^2

  ex = ntuple(n -> zeros(ElT, Ns), Nops)
  for j in site_range
    for n in 1:Nops
      s = firstind(T; tags="Site, n=$j", plev=0)
      if is_operator(T)
        Top = replaceprime(T * op(ops[n], s'), 2 => 1; tags="Site, n=$j")
        ex[n][j - offset] = real(tr(Top) / normalization)
      else
        ex[n][j - offset] =
          real(scalar(dag(T) * noprime(op(ops[n], s) * T))) / normalization
      end
    end
  end

  if Nops == 1
    return Ns == 1 ? ex[1][1] : ex[1]
  else
    return Ns == 1 ? [x[1] for x in ex] : ex
  end
end

@non_differentiable ITensors.name(::Any)
