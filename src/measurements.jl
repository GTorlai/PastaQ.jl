"""
    entanglemententropy(ψ::MPS; bond = nothing)

Measure the entanglement entropy of an MPS `ψ` at `bond`.
"""
function entanglemententropy(ψ0::MPS; bond::Int=length(ψ0) ÷ 2)
  # make sure the state is normalized
  ψ = normalize!(copy(ψ0))

  # number of qubits
  N = length(ψ)
  @assert (bond < N)

  # gauge the MPS
  orthogonalize!(ψ, bond)

  # get singular values
  row_inds = (bond > 1 ? (linkind(ψ, bond - 1), siteind(ψ, bond)) : siteind(ψ, bond))
  u, s, v = svd(ψ[bond], row_inds)

  # Compute Von Neumann Entropy S = -Tr(ρ log(ρ))
  S = 0.0
  for n in 1:dim(s, 1)
    λ = s[n, n]^2
    S -= λ * log(λ + 1e-20)
  end
  return S
end

function entanglemententropy(ρ0::MPO; kwargs...)
  return error("Measurement of entanglement entropy for MPOs not yet implemented")
end
function entanglemententropy(ρ0::LPDO; kwargs...)
  return error("Measurement of entanglement entropy for LPDOs not yet implemented")
end

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
      s = firstind(T, tags = "Site, n=$j", plev = 0)
      if is_operator(T)
        Top = replaceprime(T * op(ops[n], s'), 2 => 1, tags = "Site, n=$j")
        ex[n][j - offset] = real(tr(Top) / normalization)
      else
        ex[n][j - offset] = real(scalar(dag(T) * noprime(op(ops[n], s) * T))) / normalization 
      end
    end
  end

  if Nops == 1
    return Ns == 1 ? ex[1][1] : ex[1]
  else
    return Ns == 1 ? [x[1] for x in ex] : ex
  end
end

