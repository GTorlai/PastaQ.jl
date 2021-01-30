"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                              STANDARD CIRCUITS                               -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""


"""
    qft(N::Int; inverse::Bool = false)

Generate a list of gates for the quantum fourier transform circuit on `N` sites.
"""
function qft(N::Int; inverse::Bool = false)
  gates = Tuple[]
  if inverse
    for j in N:-1:1
      for k in N:-1:j+1
        angle = -π / 2^(k-j)
        push!(gates, ("CRz", (k, j), (ϕ=angle,)))
      end
      push!(gates, ("H", j))
    end
  else
    for j in 1:N-1
      push!(gates, ("H", j))
      for k in j+1:N
        angle = π / 2^(k-j)
        push!(gates, ("CRz", (k,j), (ϕ=angle,)))
      end
    end
    push!(gates,("H",N))
  end
  return gates
end

"""
    ghz(N::Int)

Generate a list of gates for the GHZ state

ψ = (|0,0,…,0⟩ + |1,1,…,1⟩)/√2
"""
function ghz(N::Int)
  gates = Tuple[("H",1)]
  for j in 1:N-1
    push!(gates,("CX",(j,j+1)))
  end
  return gates
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
gatelayer(gatename::AbstractString, N::Int; kwargs...) =
  Tuple[isempty(kwargs) ? (gatename, n) : (gatename, n, values(kwargs)) for n in 1:N]

"""
    gatelayer(bonds::Vector{Vector{Int}}, gatename::AbstractString)

Create a uniform layer of two-qubit gates over a set of `bonds`.
"""
gatelayer(gatename::AbstractString,bonds::Vector{Vector{Int}}) = 
  Tuple[(gatename, Tuple(bonds[n])) for n in 1:length(bonds)]  




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
function randomlayer(gatename::AbstractString, support::Union{Int,Vector{<:Vector{Int}}}; rng = Random.GLOBAL_RNG) 
  layer = Tuple[]
  support = (support isa Int ? (1:support|>collect) : Tuple.(support))
  for n in support
    pars = randomparams(gatename, 2*length(n); rng = rng) # the 2*n is for the Haar dimension
    g = (isnothing(pars) ? (gatename, n) : (gatename, n, pars))
    push!(layer,g)
  end
  return layer
end

"""
    randomlayer(gatenames::Vector{<:AbstractString}, support::Union{Int,Vector{<:Vector{Int}}}; 
                rng = Random.GLOBAL_RNG, 
                weights::Union{Nothing,Vector{Float64}} = ones(length(gatenames))/length(gatenames))

Generate a random layer built out of one or two qubit gates, where `gatenames` is a set of possible
gates to choose from. By default, each single gate is sampled uniformaly over this set. If `weights`
are provided, each gate is sampled accordingly.
"""
function randomlayer(gatenames::Vector{<:AbstractString}, support::Union{Int,Vector{<:Vector{Int}}}; 
    rng = Random.GLOBAL_RNG, weights::Union{Nothing,Vector{Float64}} = ones(length(gatenames))/length(gatenames))
  
  support = (support isa Int ? (1:support|>collect) : Tuple.(support))
  # sample each gate
  gate_id = StatsBase.sample(gatenames, StatsBase.Weights(weights),length(support))
  layer = Tuple[]
  for (i,n) in enumerate(support)
    pars = randomparams(gate_id[i], 2*length(n); rng = rng)
    g = (isnothing(pars) ? (gate_id[i], n) : (gate_id[i], n, pars))
    push!(layer,g)
  end
  return layer
end



"""
   randomcircuit(N::Int, depth::Int, coupling_sequence::Vector{<:Vector{<:Any}};
                 twoqubitgates::Union{String,Vector{String}} = "Haar",
                 onequbitgates::Union{Nothing,String,Vector{String}} = nothing,
                 layered::Bool = true,
                 rng = Random.GLOBAL_RNG)
  

Build a random quantum circuit with `N` qubits and depth `depth`.
"""
function randomcircuit(N::Int, depth::Int, coupling_sequence::Vector{<:Vector{<:Any}};
                       twoqubitgates::Union{String,Vector{String}} = "Haar",
                       onequbitgates::Union{Nothing,String,Vector{String}} = nothing,
                       layered::Bool = true,
                       rng = Random.GLOBAL_RNG)
  
  circuit = Vector{Vector{<:Tuple}}(undef, depth)
  for d in 1:depth
    layer = Tuple[]
    # two-qubit gates
    bonds = coupling_sequence[(d-1)%length(coupling_sequence)+1]
    append!(layer, randomlayer(twoqubitgates, bonds; rng = rng))
    # one-qubit gates
    if !isnothing(onequbitgates)
      append!(layer, randomlayer(onequbitgates, N; rng = rng))
    end
    circuit[d] = layer
  end
  layered && return circuit
  return vcat(circuit...)
end


"""
    randomcircuit(N::Int, depth::Int; kwargs...)

Generate a 1D random quantum circuit
"""
randomcircuit(N::Int, depth::Int; kwargs...) = 
  randomcircuit(N, depth, lineararray(N); kwargs...)

"""
    randomcircuit(Lx::Int, Ly::Int, depth::Int; rotated::Bool = false, kwargs...)

Generate a 2D random quantum circuit
"""
randomcircuit(Lx::Int, Ly::Int, depth::Int; rotated::Bool = false, kwargs...) = 
  randomcircuit(Lx*Ly, depth, squarearray(Lx,Ly; rotated = rotated), kwargs...) 

