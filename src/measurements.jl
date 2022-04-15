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
