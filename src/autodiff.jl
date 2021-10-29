function partial_contraction(ψ::MPS, ϕ::MPS)
  T = ITensor(1)
  for n in 1:length(ψ)
    T = T * ψ[n] * ϕ[n]
  end
  return T
end

function inner_circuit(ϕ::ITensor, U::Vector{ITensor}, ψ::ITensor)
  Uψ = ψ
  for u in U
    s = commoninds(u, Uψ)
    s′ = s'
    Uψ = replaceinds(u * Uψ, s′ => s)
  end
  return (dag(ϕ) * Uψ)[]
end

function inner_circuit(ϕ::MPS, U::Vector{ITensor}, ψ::MPS; kwargs...)
  Uψ = runcircuit(ψ, U; kwargs...)
  return inner(ϕ, Uψ)
end

function rrule(::typeof(inner_circuit), ϕ::MPS, U::Vector{ITensor}, ψ::MPS; kwargs...)
  Udag = reverse([dag(swapprime(u, 0=>1)) for u in U])
  ξl = runcircuit(ϕ, Udag; move_sites_back = true, kwargs...) 
  ξr = ψ
  y = conj(inner(ξl,ξr))
  function inner_circuit_pullback(ȳ)
    ∇ = ITensor[]
    for u in U
      x  = inds(u, plev = 0)
      ξl = apply(u, ξl; move_sites_back = true, kwargs...)
      ξl = prime(ξl, x)
      ∇  = vcat(∇, partial_contraction(dag(ξl), ξr))
      ξl = noprime(ξl)
      ξr = apply(u, ξr; move_sites_back = true, kwargs...)
    end
    return (NoTangent(), NoTangent(), ȳ .* ∇, NoTangent())
  end
  return y, inner_circuit_pullback
end


# XXX: For some reason Zygote needs these definitions?
Base.reverse(z::ZeroTangent) = z
Base.adjoint(::Tuple{Nothing}) = nothing
Base.adjoint(::Tuple{Nothing,Nothing}) = nothing
(::ProjectTo{NoTangent})(::Nothing) = nothing

# XXX Zygote: Delete once OpSum rules are better defined
Base.:+(::Base.RefValue{Any}, g::NamedTuple{(:data,), Tuple{Vector{NamedTuple{(:coef, :ops), Tuple{ComplexF64, Nothing}}}}}) = g
