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

function sqrt_hermitian(ρ::ITensor; cutoff::Float64 = 1e-15)
  if !isapprox(swapprime(dag(ρ), 0 => 1), ρ)
    error("matrix is not hermitian")
  end
  D, U = eigen(ρ; ishermitian = true, cutoff = cutoff)
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

function expect(T₀::ITensor, ops; kwargs...)
  T = copy(T₀)
  s = inds(T, plev = 0)
  N = length(s)
  
  ElT = ITensors.promote_itensor_eltype([T])
  
  is_operator = !isempty(inds(T, plev = 1))

  if haskey(kwargs, :site_range)
    @warn "The `site_range` keyword arg. to `expect` is deprecated: use the keyword `sites` instead"
    sites = kwargs[:site_range]
  else
    sites = get(kwargs, :sites, 1:N)
  end
  
  site_range = (sites isa AbstractRange) ? sites : collect(sites)
  Ns = length(site_range)
  start_site = first(site_range)
  
  el_types = map(o -> ishermitian(op(o, s[start_site])) ? real(ElT) : ElT, ops)
   
  normalization = is_operator ? tr(T) : norm(T)^2

  ex = map((o, el_t) -> zeros(el_t, Ns), ops, el_types)
  for (entry, j) in enumerate(site_range)
    for (n, opname) in enumerate(ops)
      if is_operator
        val = replaceprime(op(opname, s[j])' * T, 2 => 1; inds = s[j]'')
        val = tr(val)/normalization
      else
        val = scalar(dag(T) * noprime(op(opname, s[j]) * T)) / normalization
      end
      ex[n][entry] = (el_types[n] <: Real) ? real(val) : val
    end
  end

  if sites isa Number
    return map(arr -> arr[1], ex)
  end
  return ex
end

function expect(T::ITensor, op::AbstractString; kwargs...)
  return first(expect(T, (op,); kwargs...))
end

function expect(T::ITensor, op1::AbstractString, ops::AbstractString...; kwargs...)
  return expect(T, (op1, ops...); kwargs...)
end


@non_differentiable ITensors.name(::Any)

