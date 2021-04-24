"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                                QUANTUM STATES                                -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

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
productstate(N::Int, states::Vector) =
  productstate(siteinds("Qubit", N), states)

productstate(M::Union{MPS,MPO,LPDO}, states::Vector) =
  productstate(hilbertspace(M), states; mixed = mixed)

productstate(sites::Vector{<:Index}, states::Vector) =
  MPS(state.(states, sites))

productstate(sites::Vector{<:Index}, states::Vector{<:Integer}) =
  MPS(state.(string.(Int.(states)), sites))

productstate(sites::Vector{<:Index}, state::Union{String, Integer}) =
  productstate(sites, fill(state, length(sites)))

productstate(sites::Vector{<:Index}, states::Function) =
  productstate(sites, map(states, 1:length(sites)))

"""
    productoperator(N::Int)

    productoperator(sites::Vector{<:Index})

Initialize an MPO that is a product of identity operators.
"""
productoperator(N::Int) = productoperator(siteinds("Qubit", N))

productoperator(M::Union{MPS, MPO, LPDO}) = 
  productoperator(hilbertspace(M))

productoperator(sites::Vector{<:Index}) =
  MPO([op("Id", s) for s in sites])

