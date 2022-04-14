"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                                QUANTUM STATES                                -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

@doc raw"""
    qubits(n::Int) 

Generate a ``n``-qubit Hilbert space spanned by a basis ``\{\sigma_j\}_{j=1}^n``. 
Each local degree of freedom is represented by an `ITensors.Index` object, which 
encode the local Hilbert space dimension and a unique identifier for automated
tensor contractions.
```julia
q = qubits(3)
# 3-element Vector{ITensors.Index{Int64}}:
#  (dim=2|id=114|"Qubit,Site,n=1")
#  (dim=2|id=142|"Qubit,Site,n=2")
#  (dim=2|id=830|"Qubit,Site,n=3")
```
"""
qubits(N::Int; kwargs...) = siteinds("Qubit", N; kwargs...)

@doc raw"""
    qudits(n::Int; dim::Int = 3) 
    qudits(d⃗::Vector)

Generate a ``n``-qudit Hilbert space spanned by a basis ``\{\sigma_j\}_{j=1}^n``. 
Each local degree of freedom is represented by an `ITensors.Index` object with 
dimension ``d_j``. Accepted inputs are either the number of qudits (with the 
same local dimension ``d``), or a vector of local dimensions ``\mathrm{d}=(d_1,\dots,d_n)``.
```julia
q = qudits([3,5,3])
# 3-element Vector{ITensors.Index{Int64}}:
#  (dim=3|id=639|"Qudit,Site,n=1")
#  (dim=5|id=212|"Qudit,Site,n=2")
#  (dim=3|id=372|"Qudit,Site,n=3")
```
"""
qudits(N::Int; dim::Int = 3, kwargs...) = siteinds("Qudit",N; dim, kwargs...)

function qudits(d⃗::Vector)
  return [addtags.(siteind("Qudit"; dim = d⃗[i]), "n = $i") for i in 1:length(d⃗)]
end

@doc raw"""
    productstate(hilbert::Vector{<:Index})
    productstate(n::Int; dim = 2)

Generate an MPS wavefunction correponsponding to the product state 

``|\psi\rangle = |0\rangle_1\otimes|0\rangle_2\otimes\dots|0\rangle_n``

It accepts both a Hilbert space or the number of modes and local dimension. 
"""
function productstate(hilbert::Vector{<:Index}; eltype=nothing, device=identity)
  return device(convert_eltype_function(eltype)(MPS(hilbert, "0")))
end

function productstate(N::Int; dim::Int=2, sitetype::String="Qubit", kwargs...)
  dim > 2 && return productstate(siteinds("Qudit", N; dim = dim); kwargs...)
  return productstate(siteinds(sitetype,N); kwargs...)
end


@doc raw"""
    productstate(N::Int, states::Vector{T})
    productstate(hilbert::Vector{<:Index}, states::Vector{T})

Generate an MPS wavefunction for a given input product state `states`. 
The state `T` can be specified either with bit values ``|\psi\rangle = |1\rangle\otimes|0\rangle\otimes|1\rangle``
```julia
ψ = productstate(q, [1,0,1])
# MPS
# [1] ((dim=2|id=717|"Qubit,Site,n=1"),)
# [2] ((dim=2|id=89|"Qubit,Site,n=2"),)
# [3] ((dim=2|id=895|"Qubit,Site,n=3"),)
```
or with `String` ``|\psi\rangle = |+\rangle\otimes|0\rangle\otimes|-i\rangle``
```julia
ψ = productstate(q, ["X+","Z+","Y-"]);
```
"""
function productstate(N::Int, states::Vector; dim::Int=2, sitetype::String="Qubit", kwargs...)
  dim > 2 && return productstate(siteinds("Qudit", N; dim = dim), states; kwargs...)
  return productstate(siteinds(sitetype, N), states; kwargs...)
end



function productstate(sites::Vector{<:Index}, states::Vector{<:Integer}; eltype=nothing, device=identity)
  return device(convert_eltype_function(eltype)(MPS(state.(string.(Int.(states)), sites))))
end

function productstate(sites::Vector{<:Index}, states::Vector; eltype=nothing, device=identity)
  return device(convert_eltype_function(eltype)(MPS(state.(states, sites))))
end

function productstate(sites::Vector{<:Index}, state::Union{String,Integer}; kwargs...)
  return productstate(sites, fill(state, length(sites)); kwargs...)
end

function productstate(sites::Vector{<:Index}, states::Function; kwargs...)
  return productstate(sites, map(states, 1:length(sites)); kwargs...)
end

productstate(M::Union{MPS,MPO,LPDO}; kwargs...) = productstate(originalsiteinds(M); kwargs...)

function productstate(M::Union{MPS,MPO,LPDO}, states::Vector; kwargs...)
  return productstate(originalsiteinds(M), states; kwargs...)
end

function productoperator(N::Int; dim::Int=2, sitetype::String="Qubit", kwargs...)
  dim > 2 && return productoperator(siteinds("Qudit", N; dim); kwargs...)
  return productoperator(siteinds(sitetype, N); kwargs...)
end

productoperator(M::Union{MPS,MPO,LPDO}; kwargs...) = productoperator(originalsiteinds(M); kwargs...)

function productoperator(sites::Vector{<:Index}; eltype=nothing, device=identity)
  return device(convert_eltype_function(eltype)(MPO([op("Id", s) for s in sites])))
end
