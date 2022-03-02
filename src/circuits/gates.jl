#
# Gate definitions.
# Gate names must not start with "basis".
#

# for backward compatibility
const GateName = OpName
macro GateName_str(s)
  return GateName{Symbol(s)}
end
const gate = op

#
# Random Haard unitary:
# 
# Reference: http://math.mit.edu/~edelman/publications/random_matrix_theory.pdf
function gate(::GateName"RandomUnitary", ::SiteType"Qubit", s::Index...;
              eltype = ComplexF64,
              random_matrix = randn(eltype, prod(dim.(s)), prod(dim.(s))))
  Q, _ = NDTensors.qr_positive(random_matrix)
  return ITensors.itensor(Q, prime.(s)..., dag.(s)...)
end


gate(::GateName"randU", t::SiteType"Qubit", s::Index...; kwargs...) = 
  gate("RandomUnitary", s...; kwargs...)

gate(::OpName"Id", ::SiteType"Qubit") = [1 0; 0 1]

#gate(::GateName"Φ", dims::Tuple = (2,); Φ::Number) =
#  exp(im * Φ) * gate("Id", dims)

function phase(v::AbstractVector{ElT}) where {ElT<:Number}
  for x in v
    absx = abs(x)
    if absx > 1e-3
      return x / abs(x)
    end
  end
  return one(ElT)
end

function eigenbasis(GN::GateName, t::SiteType"Qubit"; dag::Bool=false, kwargs...)
  _, U = eigen(Hermitian(gate(GN, t; kwargs...)))
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

# Get an ITensor gate from a gate definition
function ITensors.op(::OpName{name}, t::SiteType"Qubit"; kwargs...) where {name}
  gn_st = String(name)
  if startswith(gn_st, "basis")
    GN = GateName(replace(gn_st, "basis" => ""))
    return eigenbasis(GN, t; kwargs...)
  end
  return error(
    "A gate with the name \"$name\" has not been implemented yet. You can define it by overloading `gate(::GateName\"$name\") = [...]`.",
  )
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

randomparams(::GateName"RandomUnitary", N::Int = 1; eltype = ComplexF64, rng = Random.GLOBAL_RNG) = 
  (random_matrix = randn(rng, eltype, 1<<N, 1<<N),)

randomparams(s::AbstractString; kwargs...) = randomparams(GateName(s); kwargs...)

function randomparams(s::AbstractString, args...; kwargs...)
  return randomparams(GateName(s), args...; kwargs...)
end

