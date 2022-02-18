"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                                QUANTUM STATES                                -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

qubits(N::Int; kwargs...) = siteinds("Qubit", N; kwargs...)

qudits(N::Int; dim::Int = 3, kwargs...) = siteinds("Qudit",N; dim = dim, kwargs...)

qudits(d⃗::Vector; kwargs...) = 
  [addtags.(siteind("Qudit"; dim = d⃗[i]), "n = $i") for i in 1:length(d⃗)]
#
# State-like gates, used to define product input states
#

# TODO: add an arbitrary state specified by angles


state(sn::String) = state(StateName(sn))
state(sn::String, dim::Int) = state(StateName(sn), dim)

function state(sn::String, i::Index)
  st = hastags(i, "Qubit") ? state(sn) : state(sn, dim(i))
  return ITensors.itensor(st, i)
end

# Pauli eingenstates
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


state(::StateName"Z-") = [
  0
  1
]

state(::StateName"0") = state("Z+")
state(::StateName"1") = state("Z-")

# SIC-POVMs

state(::StateName"T1") = state("Z+")
state(::StateName"T2") = [
  1/√3
  √2/√3
]
state(::StateName"T3") = [
  1/√3
  √2/√3 * exp(im*2π/3)
]
state(::StateName"T4") = [
  1/√3
  √2/√3 * exp(im*4π/3)
]


function state(::StateName{N}, dim::Int) where {N}
  n = parse(Int, String(N))
  st = zeros(Int64,dim)
  st[n + 1] = 1
  return st
end


"""
    productstate(N::Int)
    
    productstate(sites::Vector{<:Index})


Initialize qubits to an MPS wavefunction in the 0 state (`|ψ⟩ = |0⟩ ⊗ |0⟩ ⊗ …`).
"""
# TODO: add dimension to use qudit instead of qubit
function productstate(N::Int; dim::Int = 2, sitetype::String = "Qubit")
  dim > 2 && return productstate(siteinds("Qudit", N; dim = dim))
  return productstate(siteinds(sitetype,N))
end

productstate(sites::Vector{<:Index}) = productMPS(sites, "0")

"""
    productstate(M::Union{MPS,MPO,LPDO})

Initialize qubits on the Hilbert space of a reference state,
given as `MPS`, `MPO` or `LPDO`.
"""
productstate(M::Union{MPS,MPO,LPDO}) = productstate(originalsiteinds(M))

"""
    productstate(N::Int, states::Vector{T})

    productstate(sites::Vector{<:Index}, states::Vector{T})

    productstate(M::Union{MPS,MPO,LPDO}, states::Vector{T})

Initialize the qubits to a given product state, where the state `T` can be specified either
with a Vector of states specified as Strings or bit values (0 and 1).
"""
function productstate(N::Int, states::Vector; dim::Int = 2, sitetype::String = "Qubit")
  dim > 2 && return productstate(siteinds("Qudit", N; dim = dim), states)
  return productstate(siteinds(sitetype, N), states)
end

function productstate(sites::Vector{<:Index}, states::Vector{<:Integer})
  return MPS(state.(string.(Int.(states)), sites))
end

function productstate(sites::Vector{<:Index}, states::Vector)
  return MPS(state.(states, sites))
end

function productstate(M::Union{MPS,MPO,LPDO}, states::Vector)
  return productstate(originalsiteinds(M), states)
end

function productstate(sites::Vector{<:Index}, state::Union{String,Integer})
  return productstate(sites, fill(state, length(sites)))
end

function productstate(sites::Vector{<:Index}, states::Function)
  return productstate(sites, map(states, 1:length(sites)))
end

"""
    productoperator(N::Int)

    productoperator(sites::Vector{<:Index})

Initialize an MPO that is a product of identity operators.
"""
function productoperator(N::Int; dim::Int = 2, sitetype::String = "Qubit")
  dim > 2 && return productoperator(siteinds("Qudit", N; dim = dim))
  return productoperator(siteinds(sitetype, N))
end

productoperator(M::Union{MPS,MPO,LPDO}) = productoperator(originalsiteinds(M))

productoperator(sites::Vector{<:Index}) = MPO([op("Id", s) for s in sites])

