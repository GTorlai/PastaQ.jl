"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                                QUANTUM STATES                                -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

#
# State-like gates, used to define product input states
#

# TODO: add an arbitrary state specified by angles


state(sn::String; kwargs...) = state(StateName(sn); kwargs...)

state(sn::String, i::Index; kwargs...) = itensor(state(sn; kwargs...), i)

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

state(::StateName"0") = state("Z+")

state(::StateName"Z-") = [
  0
  1
]

state(::StateName"1") = state("Z-")

# SIC-POVMs

state(::StateName"SIC1") = state("Z+")
state(::StateName"SIC2") = [
  1/√3
  √2/√3
]
state(::StateName"SIC3") = [
  1/√3
  √2/√3 * exp(im*2π/3)
]
state(::StateName"SIC4") = [
  1/√3
  √2/√3 * exp(im*4π/3)
]




"""
    productstate(N::Int)
    
    productstate(sites::Vector{<:Index})


Initialize qubits to an MPS wavefunction in the 0 state (`|ψ⟩ = |0⟩ ⊗ |0⟩ ⊗ …`).
"""
productstate(N::Int) = productstate(siteinds("Qubit", N))

productstate(sites::Vector{<:Index}) = productMPS(sites, "0")

"""
    productstate(M::Union{MPS,MPO,LPDO})

Initialize qubits on the Hilbert space of a reference state,
given as `MPS`, `MPO` or `LPDO`.
"""
productstate(M::Union{MPS,MPO,LPDO}) = productstate(hilbertspace(M))

"""
    productstate(N::Int, states::Vector{T})

    productstate(sites::Vector{<:Index}, states::Vector{T})

    productstate(M::Union{MPS,MPO,LPDO}, states::Vector{T})

Initialize the qubits to a given product state, where the state `T` can be specified either
with a Vector of states specified as Strings or bit values (0 and 1).
"""
productstate(N::Int, states::Vector) = productstate(siteinds("Qubit", N), states)

function productstate(M::Union{MPS,MPO,LPDO}, states::Vector)
  return productstate(hilbertspace(M), states)
end

productstate(sites::Vector{<:Index}, states::Vector) = MPS(state.(states, sites))

function productstate(sites::Vector{<:Index}, states::Vector{<:Integer})
  return MPS(state.(string.(Int.(states)), sites))
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
productoperator(N::Int) = productoperator(siteinds("Qubit", N))

productoperator(M::Union{MPS,MPO,LPDO}) = productoperator(hilbertspace(M))

productoperator(sites::Vector{<:Index}) = MPO([op("Id", s) for s in sites])

