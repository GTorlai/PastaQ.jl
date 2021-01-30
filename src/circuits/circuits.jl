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
    for j in 1:N
      push!(gates, ("H", j))
      for k in j+1:N
        angle = π / 2^(k-j)
        push!(gates, ("CRz", (k,j), (ϕ=angle,)))
      end
    end
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

Create a uniform layer of single-qubit gates. If additional parameteres are 
provided, they are identically added to all gates.
"""
gatelayer(gatename::AbstractString, N::Int; kwargs...) =
  Tuple[isempty(kwargs) ? (gatename, n) : (gatename, n, values(kwargs)) for n in 1:N]

"""
    gatelayer(bonds::Vector{Vector{Int}}, gatename::AbstractString)

Create a layer of two-qubit gates for a set of bonds
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

randomparams(::GateName"Rx"; rng = Random.GLOBAL_RNG) = (θ = π*rand(rng),)
randomparams(::GateName"Ry"; rng = Random.GLOBAL_RNG) = (θ = π*rand(rng),) 
randomparams(::GateName"Rz"; rng = Random.GLOBAL_RNG) = (ϕ = 2*π*rand(rng),)

randomparams(::GateName"Rn"; rng = Random.GLOBAL_RNG) = 
  ( θ = π * rand(rng), ϕ = 2 * π * rand(rng), λ = π * rand(rng))

randomparams(::GateName"Haar", N::Int = 2; rng = Random.GLOBAL_RNG) = 
  (random_matrix = randn(rng,ComplexF64, N, N),)


randomparams(s::AbstractString; kwargs...) = 
  randomparams(GateName(s); kwargs...)

randomparams(s::AbstractString, args...; kwargs...) = 
  randomparams(GateName(s), args...; kwargs...)


randomlayer(gatename::AbstractString, N::Int; rng = Random.GLOBAL_RNG) = 
  Tuple[(gatename, n, randomparams(gatename; rng = rng)) for n in 1:N]

function randomlayer(gatenames::Vector{<:AbstractString}, N::Int; rng = Random.GLOBAL_RNG) 
  gate_id = rand(rng,gatenames,N)
  return Tuple[(gate_id[n], n) for n in 1:N]
  #return Tuple[(gate_id[n], n, randomparams(gate_id[n]; rng = rng)) for n in 1:N]
end

randomlayer(gatename::AbstractString, bonds::Vector{<:Vector{Int}}; rng = Random.GLOBAL_RNG) = 
  Tuple[(gatename, Tuple(bonds[n]), randomparams(gatename, 4; rng = rng)) for n in 1:length(bonds)]

#randomlayer(gatename::AbstractString, bonds::Matrix{Int}; rng = Random.GLOBAL_RNG) = 
#  Tuple[(gatename, Tuple(bonds[n,:]), randomparams(gatename, 4; rng = rng)) for n in 1:size(bonds,1)]

randomcircuit(N::Int, depth::Int; kwargs...) = 
  randomcircuit(N, depth, lineararray(N); kwargs...)

randomcircuit(Lx::Int, Ly::Int, depth::Int; rotated::Bool = false, kwargs...) = 
  randomcircuit(Lx*Ly, depth, squarearray(Lx,Ly; rotated = rotated), kwargs...) 


"""
    function randomcircuit(N::Int, depth::Int, coupling_sequence::Vector{<:Vector{<:Any}};
                           twoqubitgate::String   = "Haar",
                           onequbitgates = nothing,
                           layered::Bool = true,
                           rng = Random.GLOBAL_RNG)

Build a random quantum circuit with `N` qubits and depth `depth`.
"""
function randomcircuit(N::Int, depth::Int, coupling_sequence::Vector{<:Vector{<:Any}};
                       twoqubitgate::String   = "Haar",
                       onequbitgates = nothing,
                       layered::Bool = true,
                       rng = Random.GLOBAL_RNG)
  
  circuit = Vector{Vector{<:Tuple}}(undef, depth)
  
  for d in 1:depth
    layer = Tuple[]
    bonds = coupling_sequence[(d-1)%length(coupling_sequence)+1]
    if twoqubitgate == "Haar"
      append!(layer, randomlayer("Haar", bonds; rng = rng))
    else
      append!(layer, gatelayer(twoqubitgate, bonds))
    end 
    if onequbitgates == ["Rn"] ||  onequbitgates == ["Haar"]
      append!(layer, randomlayer(onequbitgates[], N; rng = rng))
    elseif !isnothing(onequbitgates)
      append!(layer, randomlayer(onequbitgates, N; rng = rng))
    end
    circuit[d] = layer
  end
  layered && return circuit
  return vcat(circuit...)
end

