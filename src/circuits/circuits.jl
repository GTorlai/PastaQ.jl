# This makes use of the `ITensors.Ops.Op` type, which
# automatically parses a gate represented as a Tuple
# into it's name, sites, and parameters.
nqubits(gate::Tuple) = maximum(Ops.sites(Op(gate)))

nqubits(gates::Vector) = maximum((nqubits(gate) for gate in gates))

nlayers(circuit::Vector) = 1
nlayers(circuit::Vector{<:Vector}) = length(circuit)

ngates(circuit::Vector) = length(circuit)
ngates(circuit::Vector{<:Vector}) = sum(length, circuit)

"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                               STANDARD CIRCUITS                              -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

@doc raw"""
    qft(n::Int; inverse::Bool = false)

Generate a list of gates for the quantum fourier transform circuit on ``n`` qubits.
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

@doc raw"""
    ghz(n::Int)

Generate a list of gates for the GHZ state.

``|\psi\rangle = (|0,0,\dots,0\rangle + |1,1,\dots,1\rangle)/\sqrt{2}``
"""
function ghz(N::Int)
  circuit = []
  push!(circuit, ("H", 1))
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

@doc raw"""
    gatelayer(gatename::AbstractString, n::Int; kwargs...)

Create a uniform layer containing `n` identical quantum gates, idenfitied by
`gatename`. If additional parameteres are provided, they are identically added to all gates.

```julia
gatelayer("H",3)
# 3-element Vector{Tuple{String, Int64}}:
#  ("H", 1)
#  ("H", 2)
#  ("H", 3)

gatelayer("X",1:2:5)
# 3-element Vector{Tuple{String, Int64}}:
#  ("X", 1)
#  ("X", 3)
#  ("X", 5)
```
"""
function gatelayer(
  gatename::AbstractString, support::Union{Vector{<:Int}, AbstractRange}; kwargs...
)
  return [isempty(kwargs) ? (gatename, n) : (gatename, n, values(kwargs)) for n in support]
end

gatelayer(gatename::AbstractString, N::Int; kwargs...) = gatelayer(gatename, 1:N; kwargs...)

@doc raw"""
    gatelayer(gatename::AbstractString, bonds::Vector{Vector{Int}}; kwargs...)

Create a uniform layer of multi-qubit gates over a set of `bonds`.

```julia
gatelayer("CX", [(j,j+1), j=1:2:5])
# 3-element Vector{Tuple{String, Tuple{Int64, Int64}}}:
#  ("CX", (1, 2))
#  ("CX", (3, 4))
#  ("CX", (5, 6))
```
"""
function gatelayer(gatename::AbstractString, bonds::Vector{<:Tuple}; kwargs...)
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


@doc raw"""
    randomlayer(
      gatename::AbstractString, 
      support::Union{Int, Vector{<:Tuple}, AbstractRange}; 
      rng = Random.GLOBAL_RNG,
      kwargs...
    ) 

Generate a random layer built out of a set one or two qubit gates If `support::Int = n`, generates 
``n`` single-qubit gates `gatename`. If `support::Vector=bonds`, generates a set of two-qubit
gates on the couplings contained in `support`.
```julia
randomlayer("Ry", 3)
# 3-element Vector{Any}:
#  ("Ry", 1, (θ = 0.5029516521736083,))
#  ("Ry", 2, (θ = 2.5324545964433693,))
#  ("Ry", 3, (θ = 2.0510824561219523,))
```
"""
function randomlayer(
  gatename::AbstractString,
  support::Union{Vector{<:Int},Vector{<:Tuple},AbstractRange};
  rng=Random.GLOBAL_RNG,
  kwargs...,
)
  layer = []
  for n in support
    pars = randomparams(gatename, length(n); rng = rng) # the 2^n is for the Haar dimension
    gatepars = (isempty(pars) ? (isempty(kwargs) ? nothing : values(kwargs)) : merge(pars,values(kwargs)))
    g = (isnothing(gatepars) ? (gatename, n) : (gatename, n, gatepars))
    layer = vcat(layer, [g])
  end
  return layer
end

function randomlayer(gatename::AbstractString, support::Int; kwargs...)
  return randomlayer(gatename, 1:support; kwargs...)
end

@doc raw"""
    randomlayer(
      gatenames::Vector{<:AbstractString},
      support::Union{Vector{<:Int}, AbstractRange, Vector{<:Tuple}};
      rng=Random.GLOBAL_RNG,
      weights::Union{Nothing,Vector{Float64}} = ones(length(gatenames)) / length(gatenames),
      kwargs...,
    ) 

Generate a random layer built out of one or two qubit gates, where `gatenames` is a set of possible
gates to choose from. By default, each single gate is sampled uniformaly over this set. If `weights`
are provided, each gate is sampled accordingly.
```julia
randomlayer(["X","Y","Z"], 3)
# 3-element Vector{Any}:
#  ("Y", 1)
#  ("Y", 2)
#  ("X", 3)
```
"""
function randomlayer(
  gatenames::Vector{<:AbstractString},
  support::Union{Vector{<:Int}, AbstractRange, Vector{<:Tuple}};
  rng=Random.GLOBAL_RNG,
  weights::Union{Nothing,Vector{Float64}}=ones(length(gatenames)) / length(gatenames),
  kwargs...,
)
  gate_id = StatsBase.sample(gatenames, StatsBase.Weights(weights), length(support))
  layer = []
  for (i,n) in enumerate(support)
    pars = randomparams(gate_id[i], length(n); rng = rng)
    gatepars = (isempty(pars) ? (isempty(kwargs) ? nothing : values(kwargs)) : merge(pars,values(kwargs)))
    g = (isnothing(gatepars) ? (gate_id[i], n) : (gate_id[i], n, gatepars))
    layer = vcat(layer, [g])
  end
  return layer
end

function randomlayer(gatenames::Vector{<:AbstractString}, support::Int; kwargs...)
  return randomlayer(gatenames, 1:support; kwargs...)
end



@doc raw"""
    randomcircuit(
      coupling_sequence::Vector;
      depth::Int = 1,
      twoqubitgates::Union{String,Vector{String}}="RandomUnitary",
      onequbitgates::Union{Nothing,String,Vector{String}}=nothing,
      layered::Bool=true,
      rng=Random.GLOBAL_RNG)

Build a circuit with given `depth`, where each layer consists of a set of 
two-qubit gates applied on pairs of qubits in according to a set of `coupling_sequences`. Each layer also contains ``n`` single-qubit gates. IN both cases, the chosen gates are passed as keyword arguments `onequbitgates` and `twoqubitgates`. 

The default configurations consists of two-qubit random Haar unitaries, and no single-qubit gates. 

If `layered = true`, the object returned in a `Vector` of circuit layers, rather than the full collection  of quantum gates.
"""
function randomcircuit(
  coupling_sequence::Vector;
  depth::Int = 1,
  twoqubitgates::Union{String,Vector{String}}="RandomUnitary",
  onequbitgates::Union{Nothing,String,Vector{String}}=nothing,
  layered::Bool=true,
  rng=Random.GLOBAL_RNG,
)
  #N = (coupling_sequence isa Vector{AbstractVector} ? maximum(vcat([maximum.(c) for c in coupling_sequence]...)) : 
  #                                            maximum([maximum(c) for c in coupling_sequence]))
  
  N = 0
  coupling_sequence = coupling_sequence isa Vector{<:Tuple} ? [coupling_sequence] : coupling_sequence
  for seq in coupling_sequence
    for b in seq
      N = max(N,b[1],b[2])
    end
  end
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
    circuit = vcat(circuit, [layer])
    #push!(circuit, layer)
  end
  layered && return circuit
  return vcat(circuit...)
end

@doc raw"""
    randomcircuit(n::Int; kwargs...)

One-dimensional random quantum circuit:
```julia
randomcircuit(4; depth = 2, twoqubitgates = "CX", onequbitgates = "Ry")
# [("CX", (1, 2)), 
#  ("CX", (3, 4)), 
#  ("Ry", 1, (θ = 0.52446,)), 
#  ("Ry", 2, (θ = 3.01059,)), 
#  ("Ry", 3, (θ = 0.25144,)), 
#  ("Ry", 4, (θ = 1.93356,))]
# [("CX", (2, 3)), 
#  ("Ry", 1, (θ = 2.15460,)), 
#  ("Ry", 2, (θ = 2.52480,)), 
#  ("Ry", 3, (θ = 1.85756,)), 
#  ("Ry", 4, (θ = 0.02405,))]
```
"""
function randomcircuit(N::Int; kwargs...)
  return randomcircuit(lineararray(N); kwargs...)
end

@doc raw"""
    randomcircuit(size::Tuple; rotated::Bool = false, kwargs...)

Two-dimensional random quantum circuit on a square lattice. If
`rotated = true`, use rotated lattice of 45 degrees.
"""
function randomcircuit(size::Tuple; rotated::Bool=false, kwargs...)
  Lx, Ly = size
  return randomcircuit(squarearray(Lx, Ly; rotated=rotated); kwargs...)
end

function randomcircuit(L::Int, depth::Int; rotated::Bool=false, kwargs...)
  error("randomcircuit(N::Int, depth::Int; kwargs...) is depracated\n
         - for a 1d random circuit: randomcircuit(N::Int; depth = depth, kwargs...)\n
         - for a 2d random circuit: randomcircuit((Lx, Ly); depth = depth, kwargs...)")
end

ITensors.dag(single_gate::Tuple{String,Union{Int,Tuple}}) = 
  (single_gate[1], single_gate[2], (adjoint = true,))

function ITensors.dag(single_gate::Tuple{String,Union{Int,Tuple},NamedTuple})
  prev_dag = get(single_gate[3], :adjoint, false)
  nt = Base.setindex(single_gate[3], !prev_dag, :adjoint)
  return (single_gate[1], single_gate[2], nt)
end

ITensors.dag(layer::Vector{<:Any}) = 
  [ITensors.dag(g) for g in reverse(layer)]

ITensors.dag(circuit::Vector{<:Vector{<:Any}}) = 
  [dag(layer) for layer in reverse(circuit)]

