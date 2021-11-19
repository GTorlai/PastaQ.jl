function partial_contraction(ψ::MPS, ϕ::MPS)
  T = ITensor(1)
  for n in 1:length(ψ)
    T = T * ψ[n] * ϕ[n]
  end
  return T
end

function inner(ϕ::ITensor, U::Vector{ITensor}, ψ::ITensor)
  Uψ = ψ
  for u in U
    s = commoninds(u, Uψ)
    s′ = s'
    Uψ = replaceinds(u * Uψ, s′ => s)
  end
  return (dag(ϕ) * Uψ)[]
end


function inner(ϕ::MPS, U::Vector{ITensor}, ψ::MPS; kwargs...)
  Uψ = runcircuit(ψ, U; kwargs...)
  return ITensors.inner(ϕ, Uψ)
end

inner(ϕ::MPS, circuit::Vector{<:Tuple}, ψ::MPS; kwargs...) = 
  inner(ϕ, buildcircuit(ψ, circuit), ψ; kwargs...)


function rrule(::typeof(inner), ϕ::MPS, U::Vector{ITensor}, ψ::MPS; cache::Bool = true, kwargs...)
  Udag = reverse([dag(swapprime(u, 0=>1)) for u in U])
  if cache
    ξ⃗ = MPS[ϕ]
    for udag in Udag
      ξ⃗  = vcat(ξ⃗, apply(udag, ξ⃗[end]; move_sites_back = true, kwargs...))
    end
    ξ⃗l = reverse(ξ⃗)
    y = inner(ξ⃗l[1], ψ)
  else
    ξl = runcircuit(ϕ, Udag; kwargs...)
    y = inner(ξl, ψ)
  end

  function inner_pullback(ȳ)
    ∇⃗ = ITensor[]
    ξr = copy(ψ)
    for (i,u) in enumerate(U)
      if cache
        ξl = ξ⃗l[i+1]
      else
        ξl = apply(u, ξl; move_sites_back = true, kwargs...)
      end
      ξl = prime(ξl, inds(u, plev = 0))
      ∇⃗ = vcat(∇⃗, partial_contraction(ξl, dag(ξr)))
      noprime!(ξl)
      ξr = apply(u, ξr; move_sites_back = true, kwargs...)
    end
    return (NoTangent(), NoTangent(), ȳ .* ∇⃗, NoTangent())
  end
  return y, inner_pullback
end


function rayleigh_quotient(O::MPO, U::Vector{ITensor}, ψ::MPS; kwargs...)
  Uψ = runcircuit(ψ, U; kwargs...)
  return real(ITensors.inner(Uψ, O, Uψ))
end

rayleigh_quotient(O::MPO, circuit::Vector{<:Tuple}, ψ::MPS; kwargs...) = 
  rayleigh_quotient(ϕ, O, buildcircuit(ψ, circuit), ψ; kwargs...)


function rrule(::typeof(rayleigh_quotient), O::MPO, U::Vector{ITensor}, ψ::MPS; kwargs...)
  ϕl = runcircuit(ψ, U; kwargs...) 
  ϕl = noprime(*(O, ϕl'; kwargs...))
  
  Udag = reverse([dag(swapprime(u, 0=>1)) for u in U])
  ξ⃗ = MPS[ϕl]
  for udag in Udag
    ξ⃗  = vcat(ξ⃗, apply(udag, ξ⃗[end]; move_sites_back = true, kwargs...))
  end
  ξ⃗l = reverse(ξ⃗)
  y = real(inner(ξ⃗l[1], ψ))

  function rayleigh_quotient_pullback(ȳ)
    ∇⃗ = ITensor[]
    ξr = copy(ψ)
    for (i,u) in enumerate(U)
      ξl = ξ⃗l[i+1]
      ξl = prime(ξl, inds(u, plev = 0))
      ∇⃗ = vcat(∇⃗, 2 * partial_contraction(ξl, dag(ξr)))
      noprime!(ξl)
      ξr = apply(u, ξr; move_sites_back = true, kwargs...)
    end
    return (NoTangent(), NoTangent(), ȳ .* ∇⃗, NoTangent())
  end
  return y, rayleigh_quotient_pullback
end


# XXX: For some reason Zygote needs these definitions?
Base.reverse(z::ZeroTangent) = z
Base.adjoint(::Tuple{Nothing}) = nothing
Base.adjoint(::Tuple{Nothing,Nothing}) = nothing
(::ProjectTo{NoTangent})(::Nothing) = nothing

@non_differentiable gate(::GateName"a", ::Tuple)
@non_differentiable ITensors.name(::Any)

#inner(ϕ::MPS, U::Vector{ITensor}, ψ::MPS, cmap::Vector; kwargs...) = 
#  inner(ϕ, U, ψ; kwargs...)
#
#function rrule(::typeof(inner_circuit), ϕ::MPS, U::Vector{ITensor}, ψ::MPS, cmap::Vector; kwargs...)
#  Udag = reverse([dag(swapprime(u, 0=>1)) for u in U])
#  ξl = runcircuit(ϕ, Udag; kwargs...) 
#  y = inner(ξl, ψ)
#  function inner_circuit_pullback(ȳ)
#    ∇⃗ = ITensor[]
#    ξr = copy(ψ)
#    gcnt = 1
#    for gloc in cmap
#      zero_tensors = [ITensors.itensor(zeros(size(U[k])),inds(U[k])) for k in gcnt:gloc-1]
#      ∇⃗ = vcat(∇⃗, zero_tensors)
#      ξl = apply(U[gcnt:gloc], ξl; move_sites_back = true, kwargs...) 
#      ξl = prime(ξl, inds(U[gloc], plev = 0))
#      if gcnt == 1
#        ξr = apply(U[gcnt:gloc-1], ξr; move_sites_back = true, kwargs...)
#      else
#        ξr = apply(U[gcnt-1:gloc-1], ξr; move_sites_back = true, kwargs...)
#      end
#      ∇⃗ = vcat(∇⃗, partial_contraction(ξl, dag(ξr)))
#      noprime!(ξl)
#      gcnt = gloc+1
#    end
#    ∇⃗ = vcat(∇⃗, U[gcnt:end])
#    return (NoTangent(), NoTangent(), ȳ .* ∇⃗, NoTangent(), NoTangent())
#  end
#  return y, inner_circuit_pullback
#end



