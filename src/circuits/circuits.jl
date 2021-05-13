"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                               STANDARD CIRCUITS                              -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

"""
    qft(N::Int; inverse::Bool = false)

Generate a list of gates for the quantum fourier transform circuit on `N` sites.
"""
function qft(N::Int; inverse::Bool=false)
  circuit = []
  if inverse
    for j in N:-1:1
      for k in N:-1:(j + 1)
        angle = -π / 2^(k - j)
        push!(circuit, ("CRz", (k, j), (ϕ=angle,)))
      end
      push!(circuit, ("H", j))
    end
  else
    for j in 1:(N - 1)
      push!(circuit, ("H", j))
      for k in (j + 1):N
        angle = π / 2^(k - j)
        push!(circuit, ("CRz", (k, j), (ϕ=angle,)))
      end
    end
    push!(circuit, ("H", N))
  end
  return circuit
end

"""
    ghz(N::Int)

Generate a list of gates for the GHZ state

ψ = (|0,0,…,0⟩ + |1,1,…,1⟩)/√2
"""
function ghz(N::Int)
  #gates = Tuple[("H",1)]
  circuit = []
  push!(circuit, ("H", 1))
  #gates = [[("H",1)...],[("CX",(1,2))...]]
  for j in 1:(N - 1)
    push!(circuit, ("CX", (j, j + 1)))
  end
  return circuit
end

"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                               LAYER FUNCTIONS                                -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

"""
    gatelayer(gatename::AbstractString, N::Int; kwargs...)

Create a uniform layer containing `N` identical quantum gates, idenfitied by
`gatename`. If additional parameteres are provided, they are identically added to all gates.
"""
function gatelayer(
  gatename::AbstractString, support::Union{Vector{<:Int},AbstractRange}; kwargs...
)
  return [isempty(kwargs) ? (gatename, n) : (gatename, n, values(kwargs)) for n in support]
end

gatelayer(gatename::AbstractString, N::Int; kwargs...) = gatelayer(gatename, 1:N; kwargs...)

"""
    gatelayer(bonds::Vector{Vector{Int}}, gatename::AbstractString)

Create a uniform layer of multi-qubit gates over a set of `bonds`.
"""
function gatelayer(gatename::AbstractString, bonds::Vector{Tuple}; kwargs...)
  return [
    isempty(kwargs) ? (gatename, bonds[n]) : (gatename, bonds[n], values(kwargs)) for
    n in 1:length(bonds)
  ]
end

"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                               RANDOM CIRCUITS                                -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

"""
    randomlayer(gatename::AbstractString, support::Union{Int,Vector{<:Vector{Int}}}; rng = Random.GLOBAL_RNG) 

Generate a random layer built out of a set one or two qubit gates If `support::Int = N`, generates 
`N` single-qubit gates `gatename`. If `support::Vector=bonds`, generates a set of two-qubit
gates on the couplings contained in `support`.
"""

function randomlayer(
  gatename::AbstractString,
  support::Union{Vector{<:Int},Vector{Tuple},AbstractRange};
  rng=Random.GLOBAL_RNG,
  kwargs...,
)
  layer = []
  for n in support
    pars = randomparams(gatename, 2^length(n); rng=rng) # the 2^n is for the Haar dimension
    gatepars = (
      if isempty(pars)
        (isempty(kwargs) ? nothing : values(kwargs))
      else
        merge(pars, values(kwargs))
      end
    )
    g = (isnothing(gatepars) ? (gatename, n) : (gatename, n, gatepars))
    push!(layer, g)
  end
  return layer
end

function randomlayer(gatename::AbstractString, support::Int; kwargs...)
  return randomlayer(gatename, 1:support; kwargs...)
end

"""
    randomlayer(gatenames::Vector{<:AbstractString}, support::Union{Int,Vector{<:Vector{Int}}}; 
                rng = Random.GLOBAL_RNG, 
                weights::Union{Nothing,Vector{Float64}} = ones(length(gatenames))/length(gatenames))

Generate a random layer built out of one or two qubit gates, where `gatenames` is a set of possible
gates to choose from. By default, each single gate is sampled uniformaly over this set. If `weights`
are provided, each gate is sampled accordingly.
"""

function randomlayer(
  gatenames::Vector{<:AbstractString},
  support::Union{Vector{<:Int},AbstractRange};
  rng=Random.GLOBAL_RNG,
  weights::Union{Nothing,Vector{Float64}}=ones(length(gatenames)) / length(gatenames),
  kwargs...,
)
  gate_id = StatsBase.sample(gatenames, StatsBase.Weights(weights), length(support))
  layer = []
  for (i, n) in enumerate(support)
    pars = randomparams(gate_id[i], 2^length(n); rng=rng)
    gatepars = (
      if isempty(pars)
        (isempty(kwargs) ? nothing : values(kwargs))
      else
        merge(pars, values(kwargs))
      end
    )
    g = (isnothing(gatepars) ? (gate_id[i], n) : (gate_id[i], n, gatepars))
    push!(layer, g)
  end
  return layer
end

function randomlayer(gatenames::Vector{<:AbstractString}, support::Int; kwargs...)
  return randomlayer(gatenames, 1:support; kwargs...)
end

"""
   randomcircuit(N::Int, depth::Int, coupling_sequence::Vector{<:Vector{<:Any}};
                 twoqubitgates::Union{String,Vector{String}} = "Haar",
                 onequbitgates::Union{Nothing,String,Vector{String}} = nothing,
                 layered::Bool = true,
                 rng = Random.GLOBAL_RNG)
  

Build a random quantum circuit with `N` qubits and depth `depth`.
"""
function randomcircuit(
  N::Int,
  depth::Int,
  coupling_sequence::Vector{<:Any};
  twoqubitgates::Union{String,Vector{String}}="RandomUnitary",
  onequbitgates::Union{Nothing,String,Vector{String}}=nothing,
  layered::Bool=true,
  rng=Random.GLOBAL_RNG,
)
  circuit = Vector[]
  for d in 1:depth
    layer = []
    # two-qubit gates
    bonds = coupling_sequence[(d - 1) % length(coupling_sequence) + 1]
    append!(layer, randomlayer(twoqubitgates, bonds; rng=rng))
    # one-qubit gates
    if !isnothing(onequbitgates)
      append!(layer, randomlayer(onequbitgates, N; rng=rng))
    end
    push!(circuit, layer)
  end
  layered && return circuit
  return vcat(circuit...)
end

"""
    randomcircuit(N::Int, depth::Int; kwargs...)

Generate a 1D random quantum circuit
"""
function randomcircuit(N::Int, depth::Int; kwargs...)
  return randomcircuit(N, depth, lineararray(N); kwargs...)
end

"""
    randomcircuit(Lx::Int, Ly::Int, depth::Int; rotated::Bool = false, kwargs...)

Generate a 2D random quantum circuit
"""
function randomcircuit(Lx::Int, Ly::Int, depth::Int; rotated::Bool=false, kwargs...)
  return randomcircuit(Lx * Ly, depth, squarearray(Lx, Ly; rotated=rotated), kwargs...)
end
