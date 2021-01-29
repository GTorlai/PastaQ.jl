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
  gates = [("H",1)]
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
gatelayer(N::Int,gatename::AbstractString; kwargs...) =
  Tuple[isempty(kwargs) ? (gatename, n) : (gatename, n, values(kwargs)) for n in 1:N]

"""
    gatelayer(bonds::Vector{Vector{Int}}, gatename::AbstractString)

Create a layer of two-qubit gates for a set of bonds
"""
gatelayer(bonds::Vector{Vector{Int}}, gatename::AbstractString) = 
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


randomlayer(N::Int, gatename::AbstractString; rng = Random.GLOBAL_RNG) = 
  Tuple[(gatename, n, randomparams(gatename; rng = rng)) for n in 1:N]

function randomlayer(N::Int, gatenames::Vector{<:AbstractString}; rng = Random.GLOBAL_RNG) 
  gate_id = rand(gatenames,N)
  return Tuple[(gate_id[n], n, randomparams(gate_id[n]; rng = rng)) for n in 1:N]
end

randomlayer(bonds::Vector{Vector{Int}}, gatename::AbstractString; rng = Random.GLOBAL_RNG) = 
  Tuple[(gatename, Tuple(bonds[n]), randomparams(gatename; rng = rng)) for n in 1:length(bonds)]



function randomcircuit_haar(depth::Int, couplings_set::Vector; rng = Random.GLOBAL_RNG)
  circuit = Vector{Vector{<:Tuple}}()
  
  for d in 1:depth
    bonds = couplings_set[(d-1)%length(couplings_set)+1]
    push!(circuit, randomlayer(bonds, "Haar"; rng = rng))
  end
  return circuit
end

function randomcircuit_Rn(N::Int, depth::Int, couplings_set::Vector; twoqubitgate = "CX", rng = Random.GLOBAL_RNG)
  circuit = Vector{Vector{<:Tuple}}()
  
  for d in 1:depth
    bonds = couplings_set[(d-1)%length(couplings_set)+1]
    push!(circuit, gatelayer(bonds, "CX"))
    push!(circuit, randomlayer(N, "Rn"; rng = rng))
  end
  return circuit
end

function randomcircuit(N::Int, depth::Int; randomgate::String = "Rn", rng = Random.GLOBAL_RNG)
  geometry = lineararray(N)
  randomgate == "Rn" && return randomcircuit_Rn(N,depth, geometry)
  return randomcircuit_haar(depth, geometry; rng = rng)
end


function randomcircuit(Lx::Int, Ly::Int, depth::Int; randomgate::String = "Rn", rng = Random.GLOBAL_RNG)
  geometry = squarearray(Lx,Ly) 
  randomgate == "Rn" && return randomcircuit_Rn(Lx*Ly,depth, geometry)
  return randomcircuit_haar(depth, geometry; rng = rng)
end



"""
    randomcircuit(N::Int,depth::Int,twoqubit_bonds::Array;
                  twoqubitgate   = "CX",
                  onequbitgates  = ["Rn"])

Build a random quantum circuit with `N` qubits and depth `depth`.
Each layer in the circuit is built with a layer of two-qubit gates
constructed according to a list of bonds contained in `twoqubit_bonds`,
followed by a layer of single qubit gates. By default, the two-qubit gate
is controlled-NOT, and the single-qubit gate is a rotation around a random
axis. 
"""
#function randomcircuit(N::Int, depth::Int, twoqubit_bonds::Array;
#                       twoqubitgate::String   = "randU",
#                       onequbitgates::Array  = [],
#                       layered::Bool = true,
#                       rng = Random.GLOBAL_RNG)
#  
#  gates = Vector{Vector{<:Tuple}}(undef, depth)
#  numgates_1q = length(onequbitgates)
#  
#  for d in 1:depth
#    layer = Tuple[]
#    bonds = twoqubit_bonds[(d-1)%length(twoqubit_bonds)+1]
#    
#    #twoqubitlayer!(layer, twoqubitgate, bonds; rng = rng)
#
#    if !isempty(onequbitgates) 
#      for j in 1:N
#        onequbitgatename = onequbitgates[rand(1:numgates_1q)]
#        if onequbitgatename == "Rn"
#          g = ("Rn", j, (θ = π*rand(rng), ϕ = 2*π*rand(rng), λ = 2*π*rand(rng)))
#        elseif onequbitgatename == "randU"
#          g = ("randU", j, (random_matrix = randn(rng,ComplexF64, 2, 2),))
#        else
#          g = (onequbitgatename, j)
#        end
#        if layered
#          push!(layer,g)
#        else
#          push!(gates,g)
#        end
#      end
#    end
#    gates[d] = layer
#  end
#  layered && return gates
#  return vcat(gates...)
#end






#"""
#Generate a random quantum circuits with long-range gates at maximum range R.
#Each layer (up to the maximum `depth`) is built with:
#- a layer of two-qubit gates according to a random pairing with max range R
#  The specific gate to be used is input as a kwarg.
#- a layer of single-qubit random rotations.
#"""
#function randomcircuit(N::Int, depth::Int, R::Int;
#                       twoqubitgate::String   = "randU",
#                       onequbitgates::Array  = [],
#                       layered::Bool = true,
#                       seed = nothing)
#  rng = (isnothing(seed) ? Random.GLOBAL_RNG : MersenneTwister(seed)) 
#  
#  gates = Vector{Vector{<:Tuple}}(undef, depth)
#  numgates_1q = length(onequbitgates)
#  
#  for d in 1:depth
#    layer = Tuple[]
#    bonds = randompairing(N,R)
#    twoqubitlayer!(layer, twoqubitgate, bonds; rng = rng)
#
#    for j in 1:N
#      g = ("Rn", j, (θ = π*rand(), ϕ = 2*π*rand(), λ = 2*π*rand()))
#      push!(gates,g)
#    end
#  end
#  return gates
#end;

