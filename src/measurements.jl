"""
    measure(ψ::MPS, measurement::Tuple{String,Int}, s::Vector{<:Index})
    measure(ψ::MPS, measurement::Tuple{String,Int})
  
Perform a measurement of a 1-local operator on an MPS ψ. The operator
is identifyed by a String (corresponding to a `gate`) and a site.
If an additional set of indices `s` is provided, the correct site is 
extracted by comparing the MPS with the index order in `s`.
"""
function measure(ψ::MPS, measurement::Tuple{String,Int}, s::Vector{<:Index})
  site0 = measurement[2]
  site = findsite(ψ, s[site0])
  ϕ = orthogonalize!(copy(ψ), site)
  ϕs = ϕ[site]
  obs_op = gate(measurement[1], firstsiteind(ϕ, site))
  T = noprime(ϕs * obs_op)
  return real((dag(T) * ϕs)[])
end

measure(ψ::MPS, measurement::Tuple{String,Int}) = measure(ψ, measurement, siteinds(ψ))

function measure(ρ::MPO, args...)
  return error("Measurement of one-body operator on MPOs not yet implemented")
end
function measure(L::LPDO, args...)
  return error("Measurement of one-body operator on LPDOs not yet implemented")
end

"""
    measure(ψ::MPS, measurement::Tuple{String,Array{Int}}, s::Vector{<:Index})
    measure(ψ::MPS, measurement::Tuple{String,Array{Int}})
    measure(ψ::MPS, measurement::Tuple{String,AbstractRange}, s::Vector{<:Index})
    measure(ψ::MPS, measurement::Tuple{String,AbstractRange})
    measure(ψ::MPS, measurement::String, s::Vector{<:Index})
    measure(ψ::MPS, measurement::String)

Perform a measurement of a 1-local operator on an MPS ψ on a set of sites passed 
as a vector. If an additional set of indices `s` is provided, the correct site is 
extracted by comparing the MPS with the index order in `s`.
"""
function measure(ψ::MPS, measurement::Tuple{String,Array{Int}}, s::Vector{<:Index})
  result = []
  sites0 = measurement[2]
  ϕ = copy(ψ)
  for site0 in sites0
    site = findsite(ϕ, s[site0])
    orthogonalize!(ϕ, site)
    ϕs = ϕ[site]
    obs_op = gate(measurement[1], firstsiteind(ϕ, site))
    T = noprime(ϕs * obs_op)
    push!(result, real((dag(T) * ϕs)[]))
  end
  return result
end

function measure(ψ::MPS, measurement::Tuple{String,Array{Int}})
  return measure(ψ, measurement, siteinds(ψ))
end

# for a range of sites
function measure(ψ::MPS, measurement::Tuple{String,AbstractRange}, s::Vector{<:Index})
  return measure(ψ, (measurement[1], Array(measurement[2])), s)
end

function measure(ψ::MPS, measurement::Tuple{String,AbstractRange})
  return measure(ψ, (measurement[1], Array(measurement[2])), siteinds(ψ))
end

# for every sites
function measure(ψ::MPS, measurement::String, s::Vector{<:Index})
  return measure(ψ::MPS, (measurement, 1:length(ψ)), s)
end

## for every sites
function measure(ψ::MPS, measurement::String)
  return measure(ψ::MPS, (measurement, 1:length(ψ)), siteinds(ψ))
end

# at a given site
"""
    measure(ψ::MPS, measurement::Tuple{String,Int,String,Int}, s::Vector{<:Index})


Perform a measurement of a 2-body tensor-product operator on an MPS ψ. The two operators
are defined by Strings (for op name) and the sites. If an additional set of indices `s` is provided, the correct site is 
extracted by comparing the MPS with the index order in `s`.
"""
function measure(ψ::MPS, measurement::Tuple{String,Int,String,Int}, s::Vector{<:Index})
  obsA = measurement[1]
  obsB = measurement[3]
  siteA0 = measurement[2]
  siteB0 = measurement[4]
  siteA = findsite(ψ, s[siteA0])
  siteB = findsite(ψ, s[siteB0])

  if siteA > siteB
    obsA, obsB = obsB, obsA
    siteA, siteB = siteB, siteA
  end
  ϕ = orthogonalize!(copy(ψ), siteA)
  ϕdag = prime(dag(ϕ); tags="Link")

  if siteA == siteB
    C = ϕ[siteA] * gate(obsA, firstsiteind(ϕ, siteA))
    C = noprime(C; tags="Site") * gate(obsA, firstsiteind(ϕ, siteA))
    C = noprime(C; tags="Site") * noprime(ϕdag[siteA])
    return real(C[])
  end
  if siteA == 1
    C = ϕ[siteA] * gate(obsA, firstsiteind(ϕ, siteA))
    C = noprime(C; tags="Site") * ϕdag[siteA]
  else
    C =
      prime(ϕ[siteA], commonind(ϕ[siteA], ϕ[siteA - 1])) *
      gate(obsA, firstsiteind(ϕ, siteA))
    C = noprime(C; tags="Site") * ϕdag[siteA]
  end
  for j in (siteA + 1):(siteB - 1)
    C = C * ϕ[j]
    C = C * ϕdag[j]
  end
  if siteB == length(ϕ)
    C = C * ϕ[siteB] * gate(obsB, firstsiteind(ϕ, siteB))
    C = noprime(C; tags="Site") * ϕdag[siteB]
  else
    C =
      C *
      prime(ϕ[siteB], commonind(ϕ[siteB], ϕ[siteB + 1])) *
      gate(obsB, firstsiteind(ϕ, siteB))
    C = noprime(C; tags="Site") * ϕdag[siteB]
  end
  return real(C[])
end

function measure(ψ::MPS, measurement::Tuple{String,Int,String,Int})
  return measure(ψ, measurement, siteinds(ψ))
end

function measure(ψ::MPS, measurement::Tuple{String,String}, s::Vector{<:Index})
  N = length(ψ)
  C = Matrix{Float64}(undef, N, N)
  for siteA in 1:N
    for siteB in 1:N
      m = (measurement[1], siteA, measurement[2], siteB)
      result = measure(ψ, m, s)
      C[siteA, siteB] = result
    end
  end
  return C
end

# for every sites
function measure(ψ::MPS, measurement::Tuple{String,String})
  return measure(ψ::MPS, measurement, siteinds(ψ))
end

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
