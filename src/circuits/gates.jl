#
# Gate definitions.
# Gate names must not start with "basis".
#

const GateName = OpName

macro GateName_str(s)
  return GateName{Symbol(s)}
end

#
# State-like gates, used to define product input states
#

# TODO: add an arbitrary state specified by angles

state(::StateName"X+") = [
  1 / sqrt(2)
  1 / sqrt(2)
]

state(::StateName"X-") = [
  1 / sqrt(2)
  -1 / sqrt(2)
]

state(::StateName"Y+") = [
  1 / sqrt(2)
  im / sqrt(2)
]

state(::StateName"Y-") = [
  1 / sqrt(2)
  -im / sqrt(2)
]

state(::StateName"Z+") = [
  1
  0
]

state(::StateName"0") = state("Z+")

state(::StateName"Z-") = [
  0
  1
]

state(::StateName"1") = state("Z-")

state(sn::String; kwargs...) = state(StateName(sn); kwargs...)

state(sn::String, i::Index; kwargs...) = itensor(state(sn; kwargs...), i)

#
# 1-qubit gates
#

gate(::GateName"Id") = [
  1 0
  0 1
]

gate(::GateName"I") = gate("Id")

gate(::GateName"X") = [
  0 1
  1 0
]

gate(::GateName"σx") = gate("X")

gate(::GateName"σ1") = gate("X")

gate(::GateName"√NOT") = [
  (1 + im)/2 (1 - im)/2
  (1 - im)/2 (1 + im)/2
]

gate(::GateName"√X") = gate("√NOT")

gate(::GateName"Y") = [
  0 -im
  im 0
]

gate(::GateName"σy") = gate("Y")

gate(::GateName"σ2") = gate("Y")

gate(::GateName"iY") = [
  0 1
  -1 0
]

gate(::GateName"iσy") = gate("iY")

gate(::GateName"iσ2") = gate("iY")

# Rϕ with ϕ = π
gate(::GateName"Z") = [
  1 0
  0 -1
]

gate(::GateName"σz") = gate("Z")

gate(::GateName"σ3") = gate("Z")

gate(::GateName"H") = [
  1/sqrt(2) 1/sqrt(2)
  1/sqrt(2) -1/sqrt(2)
]

# Rϕ with ϕ = π/2
gate(::GateName"Phase") = [
  1 0
  0 im
]

gate(::GateName"P") = gate("Phase")

gate(::GateName"S") = gate("Phase")

# Rϕ with ϕ = π/4
gate(::GateName"π/8") = [
  1 0
  0 1 / sqrt(2)+im / sqrt(2)
]

gate(::GateName"T") = gate("π/8")

# Rotation around X-axis
gate(::GateName"Rx"; θ::Number) = [
  cos(θ / 2) -im*sin(θ / 2)
  -im*sin(θ / 2) cos(θ / 2)
]

# Rotation around Y-axis
gate(::GateName"Ry"; θ::Number) = [
  cos(θ / 2) -sin(θ / 2)
  sin(θ / 2) cos(θ / 2)
]

# Rotation around Z-axis
gate(::GateName"Rz"; ϕ::Number) = [
  1 0
  0 exp(im * ϕ)
]

# Rotation around generic axis n̂
function gate(::GateName"Rn"; θ::Real, ϕ::Real, λ::Real)
  return [
    cos(θ / 2) -exp(im * λ)*sin(θ / 2)
    exp(im * ϕ)*sin(θ / 2) exp(im * (ϕ + λ))*cos(θ / 2)
  ]
end

gate(::GateName"Rn̂"; kwargs...) = gate("Rn"; kwargs...)

#
# 2-qubit gates
#

gate(::GateName"CNOT") = [
  1 0 0 0
  0 1 0 0
  0 0 0 1
  0 0 1 0
]

gate(::GateName"CX") = gate("CNOT")

gate(::GateName"CY") = [
  1 0 0 0
  0 1 0 0
  0 0 0 -im
  0 0 im 0
]

gate(::GateName"CZ") = [
  1 0 0 0
  0 1 0 0
  0 0 1 0
  0 0 0 -1
]

# Same as CRn with (θ = 0, λ = 0)
gate(::GateName"CRz"; ϕ::Real) = [
  1 0 0 0
  0 1 0 0
  0 0 1 0
  0 0 0 exp(im * ϕ)
]

function gate(::GateName"CRn"; θ::Real, ϕ::Real, λ::Real)
  return [
    1 0 0 0
    0 1 0 0
    0 0 cos(θ / 2) -exp(im * λ)*sin(θ / 2)
    0 0 exp(im * ϕ)*sin(θ / 2) exp(im * (ϕ + λ))*cos(θ / 2)
  ]
end

gate(::GateName"SWAP") = [
  1 0 0 0
  0 0 1 0
  0 1 0 0
  0 0 0 1
]

gate(::GateName"Sw") = gate("SWAP")

gate(::GateName"Swap") = gate("SWAP")

function gate(::GateName"√SWAP")
  return [
    1 0 0 0
    0 (1 + im)/2 (1 - im)/2 0
    0 (1 - im)/2 (1 + im)/2 0
    0 0 0 1
  ]
end

gate(::GateName"iSwap") = [
  1 0 0 0
  0 0 im 0
  0 im 0 0
  0 0 0 1
];

gate(::GateName"iSw") = gate("iSwap")

# Ising (XX) coupling gate
function gate(::GateName"XX"; ϕ::Number)
  return [
    cos(ϕ) 0 0 -im*sin(ϕ)
    0 cos(ϕ) -im*sin(ϕ) 0
    0 -im*sin(ϕ) cos(ϕ) 0
    -im*sin(ϕ) 0 0 cos(ϕ)
  ]
end

# TODO: Ising (YY) coupling gate
#gate(::GateName"YY"; ϕ::Number) =
#  [...]

# TODO: Ising (ZZ) coupling gate
#gate(::GateName"ZZ"; ϕ::Number) =
#  [...]

#
# 3-qubit gates
#

function gate(::GateName"Toffoli")
  return [
    1 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0
    0 0 1 0 0 0 0 0
    0 0 0 1 0 0 0 0
    0 0 0 0 1 0 0 0
    0 0 0 0 0 1 0 0
    0 0 0 0 0 0 0 1
    0 0 0 0 0 0 1 0
  ]
end

gate(::GateName"CCNOT") = gate("Toffoli")

gate(::GateName"CCX") = gate("Toffoli")

gate(::GateName"TOFF") = gate("Toffoli")

function gate(::GateName"Fredkin")
  return [
    1 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0
    0 0 1 0 0 0 0 0
    0 0 0 1 0 0 0 0
    0 0 0 0 1 0 0 0
    0 0 0 0 0 0 1 0
    0 0 0 0 0 1 0 0
    0 0 0 0 0 0 0 1
  ]
end

gate(::GateName"CSWAP") = gate("Fredkin")

gate(::GateName"CS") = gate("Fredkin")

#
# 4-qubit gates
#

function gate(::GateName"CCCNOT")
  return [
    1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0
  ]
end

#
# Random Haard unitary:
# 
# Reference: http://math.mit.edu/~edelman/publications/random_matrix_theory.pdf
function gate(
  ::GateName"randU", N::Int=2; eltype=ComplexF64, random_matrix=randn(eltype, N, N)
)
  Q, _ = NDTensors.qr_positive(random_matrix)
  return Q
end

gate(::GateName"RandomUnitary", N::Int=2; kwargs...) = gate("randU", N; kwargs...)

#
# Noise model gate definitions
#
function gate(::GateName"pauli_error"; pX::Number=0.0, pY::Number=0.0, pZ::Number=0.0)
  kraus = zeros(Complex{Float64}, 2, 2, 4)
  kraus[:, :, 1] = √(1 - pX - pY - pZ) * gate("Id")
  kraus[:, :, 2] = √pX * gate("X")
  kraus[:, :, 3] = √pY * gate("Y")
  kraus[:, :, 4] = √pZ * gate("Z")
  return kraus
end

gate(::GateName"pauli_channel"; kwargs...) = gate("pauli_error"; kwargs...)

gate(::GateName"bit_flip"; p::Number) = gate("pauli_error"; pX=p)[:, :, 1:2]

gate(::GateName"phase_flip"; p::Number) = gate("pauli_error"; pZ=p)[:, :, [1, 4]]

gate(::GateName"bit_phase_flip"; p::Number) = gate("pauli_error"; pY=p)[:, :, [1, 3]]

gate(::GateName"phase_bit_flip"; kwargs...) = gate("bit_phase_flip"; kwargs...)

function gate(::GateName"AD"; γ::Number)
  kraus = zeros(2, 2, 2)
  kraus[:, :, 1] = [
    1 0
    0 sqrt(1 - γ)
  ]
  kraus[:, :, 2] = [
    0 sqrt(γ)
    0 0
  ]
  return kraus
end

# To accept the gate name "amplitude_damping"
gate(::GateName"amplitude_damping"; kwargs...) = gate("AD"; kwargs...)

function gate(::GateName"PD"; γ::Number)
  kraus = zeros(2, 2, 2)
  kraus[:, :, 1] = [
    1 0
    0 sqrt(1 - γ)
  ]
  kraus[:, :, 2] = [
    0 0
    0 sqrt(γ)
  ]
  return kraus
end

# To accept the gate name "phase_damping"
gate(::GateName"phase_damping"; kwargs...) = gate("PD"; kwargs...)

# To accept the gate name "dephasing"
gate(::GateName"dephasing"; kwargs...) = gate("PD"; kwargs...)

gate(::GateName"DEP"; p::Number) = gate("pauli_error"; pX=p / 3.0, pY=p / 3.0, pZ=p / 3.0)

# To accept the gate name "depolarizing"
gate(::GateName"depolarizing"; kwargs...) = gate("DEP"; kwargs...)

gate(::GateName"noiseDEP"; kwargs...) = gate("DEP"; kwargs...)

gate(::GateName"noiseAD"; kwargs...) = gate("AD"; kwargs...)

gate(::GateName"noisePD"; kwargs...) = gate("PD"; kwargs...)

#
# Qubit site type
#

#space(::SiteType"Qubit") = 2

#state(::SiteType"Qubit", ::StateName"0") = 1

#state(::SiteType"Qubit", ::StateName"1") = 2

#
# Basis definitions (eigenbases of measurement gates)
#

function phase(v::AbstractVector{ElT}) where {ElT<:Number}
  for x in v
    absx = abs(x)
    if absx > 1e-3
      return x / abs(x)
    end
  end
  return one(ElT)
end

function eigenbasis(GN::GateName; dag::Bool=false, kwargs...)
  _, U = eigen(Hermitian(gate(GN; kwargs...)))
  # Sort eigenvalues largest to smallest (defaults to smallest to largest)
  U = reverse(U; dims=2)
  # Fix the sign of the eigenvectors
  for n in 1:size(U, 2)
    v = @view U[:, n]
    p = phase(v)
    v ./= p
  end
  if dag
    return copy(U')
  end
  return U
end

#
# Get an ITensor gate from a gate definition
#

function gate(::GateName{gn}; kwargs...) where {gn}
  gn_st = String(gn)
  if startswith(gn_st, "basis")
    GN = GateName(replace(gn_st, "basis" => ""))
    return eigenbasis(GN; kwargs...)
  end
  return error(
    "A gate with the name \"$gn\" has not been implemented yet. You can define it by overloading `gate(::GateName\"$gn\") = [...]`.",
  )
end

# Maybe use Base.invokelatest since certain gate overloads
# may be made on the fly with @eval
gate(s::String; kwargs...) = gate(GateName(s); kwargs...)
gate(s::String, args...; kwargs...) = gate(GateName(s), args...; kwargs...)

# Version that accepts a dimension for the gate,
# for n-qubit gates
gate(gn::GateName, N::Int; kwargs...) = gate(gn; kwargs...)

function gate(gn::GateName, s1::Index, ss::Index...; kwargs...)
  s = tuple(s1, ss...)
  rs = reverse(s)
  g = gate(gn, dim(s); kwargs...)
  if ndims(g) == 1
    # TODO:
    #error("gate must have more than one dimension, use state(...) for state vectors.")
    return itensor(g, rs...)
  elseif ndims(g) == 2
    return itensor(g, prime.(rs)..., dag.(rs)...)
  elseif ndims(g) == 3
    kraus = Index(size(g, 3); tags="kraus")
    return itensor(g, prime.(rs)..., dag.(rs)..., kraus)
  end
  return error(
    "Gate definitions must be either Vector{T} (for a state), Matrix{T} (for a gate) or Array{T,3} (for a noise model). For gate name $gn, gate size is $(size(g)).",
  )
end

gate(gn::String, s::Index...; kwargs...) = gate(GateName(gn), s...; kwargs...)

function gate(gn::String, s::Vector{<:Index}, ns::Int...; kwargs...)
  return gate(gn, s[[ns...]]...; kwargs...)
end

#
# op overload so that ITensor functions like MPO(::AutoMPO) can use the gate
# definitions of the "Qubit" site type
#

function ITensors.op(gn::GateName, ::SiteType"Qubit", s::Index...; kwargs...)
  return gate(gn, s...; kwargs...)
end

"""
    gate(M::Union{MPS,MPO}, gatename::String, site::Int; kwargs...)

Generate a gate tensor for a single-qubit gate identified by `gatename`
acting on site `site`, with indices identical to a reference state `M`.
"""
function gate(M::Union{MPS,MPO}, gatename::String, site::Int; kwargs...)
  site_ind = (typeof(M) == MPS ? siteind(M, site) : firstind(M[site]; tags="Site", plev=0))
  return gate(gatename, site_ind; kwargs...)
end

"""
    gate(M::Union{MPS,MPO},gatename::String, site::Tuple; kwargs...)

Generate a gate tensor for a two-qubit gate identified by `gatename`
acting on sites `(site[1],site[2])`, with indices identical to a 
reference state `M` (`MPS` or `MPO`).
"""
function gate(M::Union{MPS,MPO}, gatename::String, site::Tuple; kwargs...)
  site_ind1 = (
    typeof(M) == MPS ? siteind(M, site[1]) : firstind(M[site[1]]; tags="Site", plev=0)
  )
  site_ind2 = (
    typeof(M) == MPS ? siteind(M, site[2]) : firstind(M[site[2]]; tags="Site", plev=0)
  )

  return gate(gatename, site_ind1, site_ind2; kwargs...)
end

gate(M::Union{MPS,MPO}, gatedata::Tuple) = gate(M, gatedata...)

function gate(
  M::Union{MPS,MPO}, gatename::String, sites::Union{Int,Tuple}, params::NamedTuple
)
  return gate(M, gatename, sites; params...)
end

"""
RANDOM GATE PARAMETERS
"""

randomparams(::GateName"Rx", args...; rng=Random.GLOBAL_RNG) = (θ=π * rand(rng),)
randomparams(::GateName"Ry", args...; rng=Random.GLOBAL_RNG) = (θ=π * rand(rng),)
randomparams(::GateName"Rz", args...; rng=Random.GLOBAL_RNG) = (ϕ=2 * π * rand(rng),)
randomparams(::GateName"CRz", args...; rng=Random.GLOBAL_RNG) = (ϕ=2 * π * rand(rng),)
function randomparams(::GateName"Rn", args...; rng=Random.GLOBAL_RNG)
  return (θ=π * rand(rng), ϕ=2 * π * rand(rng), λ=π * rand(rng))
end
function randomparams(::GateName"CRn", args...; rng=Random.GLOBAL_RNG)
  return (θ=π * rand(rng), ϕ=2 * π * rand(rng), λ=π * rand(rng))
end

randomparams(::GateName, args...; kwargs...) = NamedTuple()

function randomparams(
  ::GateName"RandomUnitary", N::Int=2; eltype=ComplexF64, rng=Random.GLOBAL_RNG
)
  return (random_matrix=randn(rng, eltype, N, N),)
end

randomparams(s::AbstractString; kwargs...) = randomparams(GateName(s); kwargs...)

function randomparams(s::AbstractString, args...; kwargs...)
  return randomparams(GateName(s), args...; kwargs...)
end
