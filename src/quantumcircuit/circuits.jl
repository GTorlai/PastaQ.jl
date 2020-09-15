"""
Circuit geometries
"""
# Returns two-qubit bonds for a linear array
function lineararray(N::Int)
  twoqubit_bonds = []
  # Cycle 1
  cycle = []
  for j in 1:2:N-1
    push!(cycle,(j,j+1))
  end
  push!(twoqubit_bonds,cycle)
  # Cycle 2
  cycle = []
  for j in 2:2:N-1
    push!(cycle,(j,j+1))
  end
  push!(twoqubit_bonds,cycle)
  return twoqubit_bonds
end

# Get site number from coordinate
coord_to_site(Lx::Int,Ly::Int,x::Int,y::Int) = Lx*(y-1) + x

# Returns two-qubit bonds fro a square array
function squarearray(Lx::Int,Ly::Int)
  N = Lx * Ly
  twoqubit_bonds = []
  # Cycle 1
  cycle = []
  for y in 1:2:Ly-1
    for x in 1:Lx
      push!(cycle,(coord_to_site(Lx,Ly,x,y),coord_to_site(Lx,Ly,x,y+1)))
    end
  end
  push!(twoqubit_bonds,cycle)
  # Cycle 2
  cycle = []
  for y in 1:2:Ly-1
    for x in 2:Lx
      push!(cycle,(coord_to_site(Lx,Ly,x,y),coord_to_site(Lx,Ly,x-1,y+1)))
    end
  end
  push!(twoqubit_bonds,cycle)
  # Cycle 3
  cycle = []
  for y in 2:2:Ly-1
    for x in 1:Lx-1
      push!(cycle,(coord_to_site(Lx,Ly,x,y),coord_to_site(Lx,Ly,x+1,y+1)))
    end
  end
  push!(twoqubit_bonds,cycle)
  # Cycle 4
  cycle = []
  for y in 2:2:Ly-1
    for x in 1:Lx
      push!(cycle,(coord_to_site(Lx,Ly,x,y),coord_to_site(Lx,Ly,x,y+1)))
    end
  end
  push!(twoqubit_bonds,cycle)
  return twoqubit_bonds
end

"""
Create a layer of gates.
"""
gatelayer(gatename::AbstractString, N::Int; kwargs...) =
  Tuple[isempty(kwargs) ? (gatename, n) : (gatename, n, values(kwargs)) for n in 1:N]

"""
Append a layer of gates to a gate list.
"""
appendlayer!(gates::AbstractVector{ <: Tuple},
             gatename::AbstractString, N::Int) =
  append!(gates, gatelayer(gatename, N))

"""
Random rotation
"""
randomrotation(site::Int) =
  ("Rn", site, (θ = π*rand(), ϕ = 2*π*rand(), λ = 2*π*rand()))

# TODO: replace with gatelayer(gatename, bonds; nqubit = 2)
# bonds could be:
# Union{Int, AbstractRange, Vector{Int}}
# to specify the starting location.
"""
Layer of two-qubit gates 
"""
function twoqubitlayer(gatename::String,bonds::Array)
  gates = Tuple[]
  for bond in bonds
    push!(gates,(gatename, bond))
  end
  return gates
end

function twoqubitlayer!(gates::Array,gatename::String,bonds::Array)
  newgates = twoqubitlayer(gatename,bonds)
  append!(gates, newgates)
  return gates
end

"""
Random quantum circuit.
"""
function randomcircuit(N::Int,depth::Int,twoqubit_bonds::Array;
                       twoqubitgate   = "CX",
                       onequbitgates  = ["Rn"])
  gates = Tuple[]
  numgates_1q = length(onequbitgates)
  
  for d in 1:depth
    cycle = twoqubit_bonds[(d-1)%2+1]
    twoqubitlayer!(gates,twoqubitgate,cycle) 
    for j in 1:N
      onequbitgatename = onequbitgates[rand(1:numgates_1q)]
      if onequbitgatename == "Rn"
        g = randomrotation(j)
      else
        g = (onequbitgatename, j)
      end
      push!(gates,g) 
    end
  end
  return gates
end

function randomcircuit(N::Int,depth::Int;
                       twoqubitgate   = "CX",
                       onequbitgates  = ["Rn"])
  twoqubit_bonds = lineararray(N)
  return randomcircuit(N,depth,twoqubit_bonds;
                       twoqubitgate=twoqubitgate,
                       onequbitgates=onequbitgates)
end

function randomcircuit(Lx::Int,Ly::Int,depth::Int;
                       twoqubitgate   = "CX",
                       onequbitgates  = ["Rn"])
  twoqubit_bonds = squarearray(Lx,Ly)
  N = Lx * Ly
  return randomcircuit(N,depth,twoqubit_bonds;
                       twoqubitgate=twoqubitgate,
                       onequbitgates=onequbitgates)
end

