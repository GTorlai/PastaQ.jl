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
  y = inner_circuit(ϕ, U, ψ) 
  function inner_circuit_pullback(ȳ)
    # build the environments
    ξr = Vector{MPS}(undef, length(U))
    ξr[1] = copy(ψ)
    for i in 1:length(U)-1
      u = U[i]
      ξ = apply(u, ξr[i]; move_sites_back = true, kwargs...)
      ξr[i+1] = ξ
    end

    ξl = Vector{MPS}(undef, length(U))
    ξl[end] = copy(ϕ)
    for i in reverse(1:length(U)-1)
      udag = dag(swapprime(U[i+1], 0=>1))
      ξl[i] = apply(udag, ξl[i+1]; move_sites_back = true, kwargs...)
    end
    
    ∇⃗ = ITensor[]
    for i in 1:length(U)
      x  = inds(U[i], plev = 0)
      ξl[i] = prime(ξl[i],x)
      ∇ = ITensor(1)
      for n in 1:length(ψ)
        # TODO: figure out the dag
        ∇ = ∇ * ξl[i][n] * dag(ξr[i][n])
      end
      ∇⃗ = vcat(∇⃗, ∇)
    end
    return (NoTangent(), NoTangent(), ȳ .* ∇⃗, NoTangent())
  end
  return y, inner_circuit_pullback
end


#function rrule(::typeof(inner_circuit), ϕ::MPS, U::Vector{ITensor}, ψ::MPS; kwargs...)
#  Udag = reverse([dag(swapprime(u, 0=>1)) for u in U])
#  ξl = runcircuit(ϕ, Udag; kwargs...) 
#  ξr = ψ
#  # TODO:double check this
#  #y = conj(inner(ξl,ξr))
#  y = inner(ξl,ξr)
#  function inner_circuit_pullback(ȳ)
#    ∇ = ITensor[]
#    for u in U
#      x  = inds(u, plev = 0)
#      ξl = apply(u, ξl; move_sites_back = true, kwargs...)
#      ξl = prime(ξl, x)
#      ∇  = vcat(∇, partial_contraction(ξl, ξr))
#      # TODO: double check this
#      #∇  = vcat(∇, partial_contraction(dag(ξl), ξr))
#      ξl = noprime(ξl)
#      ξr = apply(u, ξr; move_sites_back = true, kwargs...)
#    end
#    return (NoTangent(), NoTangent(), ȳ .* ∇, NoTangent())
#  end
#  return y, inner_circuit_pullback
#end


#function rrule(::typeof(inner_circuit), ϕ::MPS, U::Vector{ITensor}, ψ::MPS; kwargs...)
#  # |ξr⟩ = U_N ... U_1|ψ⟩
#  ξr = runcircuit(copy(ψ), U; kwargs...)
#  # |ξl⟩ = |ϕ⟩
#  ξl = copy(ϕ)
#  # output: ⟨ϕ|Uψ⟩
#  y = inner(ξl,ξr)
#  function inner_circuit_pullback(ȳ)
#    ∇⃗ = ITensor[]
#    # since we are sweeping left to right, start with the last gate
#    for u in reverse(U)
#      # get U† to undo the evolution on |ψ⟩
#      udag = dag(swapprime(u, 0=>1))
#      # |ξr⟩ → U†_j|ξr⟩ = U_j+1 ... U_1|ψ⟩
#      ξr = apply(udag, ξr; move_sites_back = true, kwargs...)
#      
#      x  = inds(udag, plev = 0)
#      ξl = prime(ξl, x)
#      
#      ∇ = ITensor(1)
#      for n in 1:length(ψ)
#        ∇ = ∇ * dag(ξl[n]) * ξr[n]
#      end
#      ∇⃗ = vcat(∇⃗, ∇)
#      ξl = noprime(ξl)
#      # propagate the final state back in time of one step
#      ξl = apply(udag, ξl; move_sites_back = true, kwargs...)
#    end
#    return (NoTangent(), NoTangent(), ȳ .* reverse(∇⃗), NoTangent())
#  end
#  return y, inner_circuit_pullback
#end

# XXX: For some reason Zygote needs these definitions?
Base.reverse(z::ZeroTangent) = z
Base.adjoint(::Tuple{Nothing}) = nothing
Base.adjoint(::Tuple{Nothing,Nothing}) = nothing
(::ProjectTo{NoTangent})(::Nothing) = nothing

# XXX Zygote: Delete once OpSum rules are better defined
Base.:+(::Base.RefValue{Any}, g::NamedTuple{(:data,), Tuple{Vector{NamedTuple{(:coef, :ops), Tuple{ComplexF64, Nothing}}}}}) = g
