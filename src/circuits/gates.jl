#
# Gate definitions.
# Gate names must not start with "basis".
#

const GateName = OpName

macro GateName_str(s)
  return GateName{Symbol(s)}
end

#
# 1-qubit gates
#

gate(::GateName"I") = gate("Id")

gate(::GateName"X") = [
  0 1
  1 0
]

gate(::GateName"σx") = gate("X")
gate(::GateName"σˣ") = gate("X")

gate(::GateName"σ1") = gate("X")
gate(::GateName"σ¹") = gate("X")

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
gate(::GateName"σʸ") = gate("Y")

gate(::GateName"σ2") = gate("Y")
gate(::GateName"σ²") = gate("Y")

gate(::GateName"iY") = [
  0 1
  -1 0
]

gate(::GateName"iσy") = gate("iY")
gate(::GateName"iσʸ") = gate("iY")

gate(::GateName"iσ2") = gate("iY")
gate(::GateName"iσ²") = gate("iY")

# Rϕ with ϕ = π
gate(::GateName"Z") = [
  1 0
  0 -1
]

gate(::GateName"σz") = gate("Z")
gate(::GateName"σᶻ") = gate("Z")

gate(::GateName"σ3") = gate("Z")
gate(::GateName"σ³") = gate("Z")

gate(::GateName"H") = [
  1/sqrt(2) 1/sqrt(2)
  1/sqrt(2) -1/sqrt(2)
]

# Rϕ with ϕ = π/2
gate(::GateName"Phase"; ϕ::Number = π/2) = [
  1 0
  0 exp(im*ϕ)
]

gate(::GateName"P"; kwargs...)     = gate("Phase"; kwargs...)
gate(::GateName"PHASE"; kwargs...) = gate("Phase"; kwargs...)

gate(::GateName"S"; kwargs...) = gate("Phase"; kwargs...)

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

gate(::GateName"RX"; kwargs...) = 
  gate("Rx"; kwargs...)

# Rotation around Y-axis
gate(::GateName"Ry"; θ::Number) = [
  cos(θ / 2) -sin(θ / 2)
  sin(θ / 2) cos(θ / 2)
]

gate(::GateName"RY"; kwargs...) = 
  gate("Ry"; kwargs...)

# Rotation around Z-axis
gate(::GateName"Rz"; ϕ::Number) = [
  exp(-im * ϕ / 2)  0
  0          exp(im * ϕ / 2)
]

gate(::GateName"RZ"; kwargs...) = 
  gate("Rz"; kwargs...)

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

gate(::GateName"CRx"; θ::Number) = [
  1 0 0 0 
  0 1 0 0 
  0 0 cos(θ / 2) -im*sin(θ / 2)
  0 0 -im*sin(θ / 2) cos(θ / 2)
]

gate(::GateName"CRX"; kwargs...) = 
  gate("CRx"; kwargs...)

gate(::GateName"CRy"; θ::Number) = [
  1 0 0 0   
  0 1 0 0 
  0 0 cos(θ / 2) -sin(θ / 2)
  0 0 sin(θ / 2) cos(θ / 2)
]
gate(::GateName"CRY"; kwargs...) = 
  gate("CRy"; kwargs...)

gate(::GateName"CRz"; ϕ::Real) = [
  1   0   0   0
  0   1   0   0
  0   0   exp(-im * ϕ / 2)    0
  0   0   0     exp(im * ϕ / 2)
]

gate(::GateName"CRZ"; kwargs...) = 
  gate("CRz"; kwargs...)

gate(::GateName"CPHASE"; ϕ::Real) = [
  1 0 0 0
  0 1 0 0
  0 0 1 0
  0 0 0 exp(im * ϕ)
]

gate(::GateName"Cphase"; kwargs...) = gate("CPHASE"; kwargs...) 
gate(::GateName"CP"; kwargs...)     = gate("CPHASE"; kwargs...)

function gate(::GateName"CRn"; θ::Real, ϕ::Real, λ::Real)
  return [
    1 0 0 0
    0 1 0 0
    0 0 cos(θ / 2) -exp(im * λ)*sin(θ / 2)
    0 0 exp(im * ϕ)*sin(θ / 2) exp(im * (ϕ + λ))*cos(θ / 2)
  ]
end

gate(::GateName"CRn̂"; kwargs...) = gate("CRn"; kwargs...)

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

gate(::GateName"√Sw") = gate("√SWAP")

gate(::GateName"√Swap") = gate("√SWAP")

gate(::GateName"iSWAP") = [
  1 0 0 0
  0 0 im 0
  0 im 0 0
  0 0 0 1
];

gate(::GateName"iSw") = gate("iSWAP")

gate(::GateName"iSwap") = gate("iSWAP")

# Ising (XX) coupling gate
function gate(::GateName"Rxx"; ϕ::Number)
  return [
    cos(ϕ) 0 0 -im*sin(ϕ)
    0 cos(ϕ) -im*sin(ϕ) 0
    0 -im*sin(ϕ) cos(ϕ) 0
    -im*sin(ϕ) 0 0 cos(ϕ)
  ]
end

gate(::GateName"RXX"; kwargs...) = 
  gate("Rxx"; kwargs...)

gate(::GateName"XX") = 
  kron(gate("X"),gate("X"))

gate(::GateName"YY") = 
  kron(gate("Y"),gate("Y"))

gate(::GateName"ZZ") = 
  kron(gate("Z"),gate("Z"))


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

gate(::GateName"CSw") = gate("Fredkin")

gate(::GateName"CSwap") = gate("Fredkin")

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
function gate(::GateName"randU", dims::Tuple = (2,);
              eltype = ComplexF64,
              random_matrix = randn(eltype, prod(dims), prod(dims)))
  Q, _ = NDTensors.qr_positive(random_matrix)
  return Q
end

gate(::GateName"RandomUnitary", dims::Tuple = (2,); kwargs...) = 
  gate("randU", dims; kwargs...)


#
# qudit gates
#

function gate(::GateName"Id", dims::Tuple = (2,))
  g = 1.0
  for k in 1:length(dims)
    g = kron(g, Matrix(I,dims[k],dims[k]))
  end
  return g
end

gate(::GateName"Φ", dims::Tuple = (2,); Φ::Number) =
  exp(im * Φ) * gate("Id", dims)


function gate(::GateName"a", dims::Tuple = (2,))
  @assert length(dims) == 1
  dim = dims[1]
  mat = zeros(dim, dim)
  for k in 1:dim-1
    mat[k,k+1] = √k
  end
  return mat
end

gate(::GateName"a†", dims::Tuple = (2,)) = 
  Array(gate("a", dims)')

gate(::GateName"n", dims::Tuple = (2,)) = 
  gate("a†", dims) * gate("a", dims)

gate(::GateName"adag", dims::Tuple) = 
  gate("a†", dims::Tuple)


gate(::GateName"a†a", dims::Tuple = (2,2)) = 
  kron(gate("a†", (dims[1],)),gate("a", (dims[2],)))

gate(::GateName"aa†", dims::Tuple = (2,2)) = 
  kron(gate("a", (dims[1],)),gate("a†", (dims[2],)))

gate(::GateName"aa", dims::Tuple = (2,2)) = 
  kron(gate("a", (dims[1],)),gate("a", (dims[2],)))

gate(::GateName"a†a†", dims::Tuple = (2,2)) = 
  kron(gate("a†", (dims[1],)),gate("a†", (dims[2],)))

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
gate(gn::GateName, dims::Tuple; kwargs...) = gate(gn; kwargs...)

"""
    combinegates(gn::GateName, s::Tuple; kwargs...)

Generate a gate (matrix) given a set of operations in the gatename string
"""
function combinegates(gn::GateName, s::Tuple; kwargs...)
  gate1 = nothing
  gate2 = nothing

  name = string(ITensors.name(gn))
  @ignore_derivatives name = filter(x -> !isspace(x), name)
  
  # first check for addition
  pluspos = findfirst("+", name)
  if !isnothing(pluspos)
    !isempty(kwargs) && error("Composition of parametric gates not allowed")
    @ignore_derivatives begin
      gate1 = name[1:prevind(name, pluspos.start)]
      gate2 = name[nextind(name, pluspos.start):end]
    end
    return combinegates(GateName(gate1), dim.(s); kwargs...) + combinegates(GateName(gate2), dim.(s); kwargs...)
  end
  # next check for multiplication
  starpos = findfirst("*", name)
  if !isnothing(starpos)
    !isempty(kwargs) && error("Composition of parametric gates not allowed")
    @ignore_derivatives begin
      gate1 = name[1:prevind(name, starpos.start)]
      gate2 = name[nextind(name, starpos.start):end]
    end
    return combinegates(GateName(gate1), dim.(s); kwargs...) * combinegates(GateName(gate2), dim.(s); kwargs...)
  end
  return gate(gn, dim.(s); kwargs...)
end

function gate(gn::GateName, s1::Index, ss::Index...; 
              dag::Bool = false,
              f = nothing,
              kwargs...)
  s = tuple(s1, ss...)
  rs = reverse(s)
  # temporary block on f. To be revised in gate system refactoring.
  !isnothing(f) && !(f isa Function) && error("gate parameter `f` not allowed")

  # generate dense gate
  g = combinegates(gn, s; kwargs...)
  
  ## apply a function if passed
  g = !isnothing(f) ? f(g) : g
  # conjugate the gate if `dag=true`
  g = dag ? Array(g') : g
  
  # generate itensor gate
  if ndims(g) == 1
    # TODO:
    #error("gate must have more than one dimension, use state(...) for state vectors.")
    return ITensors.itensor(g, rs...)
  elseif ndims(g) == 2
    return ITensors.itensor(g, prime.(rs)..., ITensors.dag.(rs)...)
  elseif ndims(g) == 3
    kraus = Index(size(g, 3); tags="kraus")
    return ITensors.itensor(g, prime.(rs)..., ITensors.dag.(rs)..., kraus)
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

ITensors.op(gn::GateName, ::SiteType"Qubit", s::Index...; kwargs...) = 
  gate(gn, s...; kwargs...)

ITensors.op(gn::GateName, ::SiteType"Qudit", s::Index...; kwargs...) = 
  gate(gn, s...; kwargs...)


gate(hilbert::Vector{<:Index}, gatedata::Tuple) = 
  gate(hilbert, gatedata...)

gate(hilbert::Vector{<:Index},
     gatename::String,
     sites::Union{Int, Tuple},
     params::NamedTuple) =
  gate(hilbert, gatename, sites; params...)

gate(hilbert::Vector{<:Index}, gatename::String, site::Int; kwargs...) = 
  gate(gatename, hilbert[site]; kwargs...) 

gate(hilbert::Vector{<:Index}, gatename::String, site::Tuple; kwargs...) = 
  gate(gatename, hilbert[collect(site)]...; kwargs...)

gate(M::Union{MPS,MPO,ITensor}, args...; kwargs...) =
  gate(originalsiteinds(M), args...; kwargs...)

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

randomparams(::GateName"RandomUnitary", N::Int = 1; eltype = ComplexF64, rng = Random.GLOBAL_RNG) = 
  (random_matrix = randn(rng, eltype, 1<<N, 1<<N),)

randomparams(s::AbstractString; kwargs...) = randomparams(GateName(s); kwargs...)

function randomparams(s::AbstractString, args...; kwargs...)
  return randomparams(GateName(s), args...; kwargs...)
end
