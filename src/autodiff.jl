#rayleigh_quotient(O::MPO, circuit::Vector{<:Tuple}, ψ::MPS; kwargs...) = 
#  rayleigh_quotient(ϕ, O, buildcircuit(ψ, circuit), ψ; kwargs...)
#
#
#function rrule(::typeof(rayleigh_quotient), O::MPO, U::Vector{ITensor}, ψ::MPS; kwargs...)
#  ϕl = runcircuit(ψ, U; kwargs...) 
#  ϕl = noprime(*(O, ϕl'; kwargs...))
#  
#  Udag = reverse([dag(swapprime(u, 0=>1)) for u in U])
#  ξ⃗ = MPS[ϕl]
#  for udag in Udag
#    ξ⃗  = vcat(ξ⃗, apply(udag, ξ⃗[end]; move_sites_back = true, kwargs...))
#  end
#  ξ⃗l = reverse(ξ⃗)
#  y = real(inner(ξ⃗l[1], ψ))
#
#  function rayleigh_quotient_pullback(ȳ)
#    ∇⃗ = ITensor[]
#    ξr = copy(ψ)
#    for (i,u) in enumerate(U)
#      ξl = ξ⃗l[i+1]
#      ξl = prime(ξl, inds(u, plev = 0))
#      ∇⃗ = vcat(∇⃗, 2 * partial_contraction(ξl, dag(ξr)))
#      noprime!(ξl)
#      ξr = apply(u, ξr; move_sites_back = true, kwargs...)
#    end
#    return (NoTangent(), NoTangent(), ȳ .* ∇⃗, NoTangent())
#  end
#  return y, rayleigh_quotient_pullback
#end
#
#function partial_contraction(ψ::MPS, ϕ::MPS)
#  T = ITensor(1)
#  for n in 1:length(ψ)
#    T = T * ψ[n] * ϕ[n]
#  end
#  return T
#end
#
#
## XXX: For some reason Zygote needs these definitions?
#Base.reverse(z::ZeroTangent) = z
#Base.adjoint(::Tuple{Nothing}) = nothing
#Base.adjoint(::Tuple{Nothing,Nothing}) = nothing
#(::ProjectTo{NoTangent})(::Nothing) = nothing
#
