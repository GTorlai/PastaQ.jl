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

# XXX
function gate(::GateName"RandomUnitary", ::SiteType"Qubit", s::Index...;
              eltype = ComplexF64,
              random_matrix = randn(eltype, prod(dim.(s)), prod(dim.(s))))
  Q, _ = NDTensors.qr_positive(random_matrix)
  return ITensors.itensor(Q, prime.(s)..., dag.(s)...)
end


gate(::GateName"randU", t::SiteType"Qubit", s::Index...; kwargs...) = 
  gate("RandomUnitary", s...; kwargs...)

gate(::OpName"Id", ::SiteType"Qubit") = [1 0; 0 1]
#XXX


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

# Maybe use Base.invokelatest since certain gate overloads
# may be made on the fly with @eval
#gate(s::String; kwargs...) = gate(GateName(s), SiteType("Qubit"); kwargs...)
#gate(s::String, args...; kwargs...) = gate(GateName(s), args...; kwargs...)


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











# Version that accepts a dimension for the gate,
#gate(gn::GateName, dims::Tuple; kwargs...) = gate(gn; kwargs...)



#
#function gate(gn::GateName, s1::Index, ss::Index...; 
#              dag::Bool = false,
#              f = nothing,
#              kwargs...)
#  s = tuple(s1, ss...)
#  rs = reverse(s)
#  # temporary block on f. To be revised in gate system refactoring.
#  !isnothing(f) && !(f isa Function) && error("gate parameter `f` not allowed")
#
#  # generate dense gate
#  g = combinegates(gn, s; kwargs...)
#  
#  ## apply a function if passed
#  g = !isnothing(f) ? f(g) : g
#  # conjugate the gate if `dag=true`
#  g = dag ? Array(g') : g
#  
#  # generate itensor gate
#  if ndims(g) == 1
#    # TODO:
#    #error("gate must have more than one dimension, use state(...) for state vectors.")
#    return ITensors.itensor(g, rs...)
#  elseif ndims(g) == 2
#    return ITensors.itensor(g, prime.(rs)..., ITensors.dag.(rs)...)
#  elseif ndims(g) == 3
#    kraus = Index(size(g, 3); tags="kraus")
#    return ITensors.itensor(g, prime.(rs)..., ITensors.dag.(rs)..., kraus)
#  end  
#  return error(
#    "Gate definitions must be either Vector{T} (for a state), Matrix{T} (for a gate) or Array{T,3} (for a noise model). For gate name $gn, gate size is $(size(g)).",
#  )
#end
#
#
#gate(gn::String, s::Index...; kwargs...) = gate(GateName(gn), s...; kwargs...)
#
#function gate(gn::String, s::Vector{<:Index}, ns::Int...; kwargs...)
#  return gate(gn, s[[ns...]]...; kwargs...)
#end
#
##
## op overload so that ITensor functions like MPO(::AutoMPO) can use the gate
## definitions of the "Qubit" site type
##
#
#ITensors.op(gn::GateName, ::SiteType"Qubit", s::Index...; kwargs...) = 
#  gate(gn, s...; kwargs...)
#
#ITensors.op(gn::GateName, ::SiteType"Qudit", s::Index...; kwargs...) = 
#  gate(gn, s...; kwargs...)
#
#
#gate(hilbert::Vector{<:Index}, gatedata::Tuple) = 
#  gate(hilbert, gatedata...)
#
#gate(hilbert::Vector{<:Index},
#     gatename::String,
#     sites::Union{Int, Tuple},
#     params::NamedTuple) =
#  gate(hilbert, gatename, sites; params...)
#
#
#gate(hilbert::Vector{<:Index},
#     gn1::String, site1::Int, 
#     gn2::String, site2::Int,
#     params::NamedTuple; kwargs...) =
#  gate(hilbert, gn1, site1, gn2, site2; params..., kwargs...)
#
##function gate(hilbert::Vector{<:Index},
##              gn1::String, site1::Int, 
##              gn2::String, site2::Int; 
##              f = nothing,
##              kwargs...)
##
##  g1 = gate(gn1, hilbert[site1]; kwargs...)
##  g2 = gate(gn2, hilbert[site2]; kwargs...)
##  g = g1 * g2
##  g = !isnothing(f) ? f(g) : g
##  return g
##end
#
##function gate(hilbert::Vector{<:Index}, gn1::Union{String, AbstractArray{<:Number}}, site1::Int, ops_rest...; f = nothing, kwargs...)  
##  ops = (gn1, site1, ops_rest...)
##  starts = findall(x -> (x isa String) || (x isa AbstractArray{<:Number}), ops)
##  N = length(starts)
##  #vop = OpTerm(undef, N)
##  
##  g = ITensor(1)
##  for n in 1:N
##    start = starts[n]
##    stop = (n == N) ? lastindex(ops) : (starts[n + 1] - 1)
##    g = g ( gate(
##    #g = g * gate(ops[start:stop]...; kwargs...)
##    println(ops[start:stop]...)
##  #  vop[n] = SiteOp(ops[start:stop]...)
##  end
##  #return MPOTerm(c, vop)
##end
##
#
#
#gate(hilbert::Vector{<:Index}, gatename::String, site::Int; kwargs...) = 
#  gate(gatename, hilbert[site]; kwargs...) 
#
#gate(hilbert::Vector{<:Index}, gatename::String, site::Tuple; kwargs...) = 
#  gate(gatename, hilbert[collect(site)]...; kwargs...)
#
#gate(M::Union{MPS,MPO,ITensor}, args...; kwargs...) =
#  gate(originalsiteinds(M), args...; kwargs...)


