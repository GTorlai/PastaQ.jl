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
Add a list of gates to gates (data structure) 
"""
function appendgates!(gates::Vector{<:Tuple},newgates::Vector{<:Tuple})
  for newgate in newgates
    push!(gates,newgate)
  end
end

"""
Layer of Hadamard gates
"""
function hadamardlayer(N::Int)
  gates = Tuple[]
  for j in 1:N
    push!(gates,("H", j))
  end
  return gates
end

function hadamardlayer!(gates::Array,N::Int)
  newgates = hadamardlayer(N)
  appendgates!(gates,newgates)
  return gates
end

"""
Random rotation
"""
function randomrotation(site::Int)
  θ,ϕ,λ = rand!(zeros(3))
  return ("Rn", site, (θ = π*θ, ϕ = 2*π*ϕ, λ = 2*π*λ))
end

"""
Layer of random rotations
"""
function randomrotationlayer(N::Int)
  gates = Tuple[]
  for j in 1:N
    g = randomrotation(j)
    push!(gates,g)
  end
  return gates
end

function randomrotationlayer!(gates::Array,N::Int)
  newgates = randomrotationlayer(N)
  appendgates!(gates,newgates)
  return gates
end

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
  appendgates!(gates,newgates)
  return gates
end

"""
Random quantum circuit
"""
function randomquantumcircuit(N::Int,depth::Int,twoqubit_bonds::Array;
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
        g = (onequbitgatename,j)
      end
      push!(gates,g) 
    end
  end
  return gates
end

function randomquantumcircuit(N::Int,depth::Int;
                              twoqubitgate   = "CX",
                              onequbitgates  = ["Rn"])
  twoqubit_bonds = lineararray(N)
  return randomquantumcircuit(N,depth,twoqubit_bonds;
                              twoqubitgate=twoqubitgate,
                              onequbitgates=onequbitgates)
end

function randomquantumcircuit(Lx::Int,Ly::Int,depth::Int;
                              twoqubitgate   = "CX",
                              onequbitgates  = ["Rn"])
  twoqubit_bonds = squarearray(Lx,Ly)
  N = Lx * Ly
  return randomquantumcircuit(N,depth,twoqubit_bonds;
                              twoqubitgate=twoqubitgate,
                              onequbitgates=onequbitgates)
end

