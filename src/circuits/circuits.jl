"""
Circuit geometries
"""

# Get site number from coordinate
coord_to_site(Lx::Int,Ly::Int,x::Int,y::Int) = Lx*(y-1) + x

"""
    lineararray(N::Int)

Return a vector of bonds for a open 1d lattice with 
`N` sites.
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
  if N>2
    # Cycle 2
    cycle = []
    for j in 2:2:N-1
      push!(cycle,(j,j+1))
    end
    push!(twoqubit_bonds,cycle)
  end
  return twoqubit_bonds
end


# Returns two-qubit bonds fro a square array
"""
    squarearray(Lx::Int,Ly::Int)

Return a vector containing 4 different "cycles" of bonds,
corresponding to the different tiling of a square lattice
with dimensions `Lx` and `Ly`.
"""
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
    gatelayer(gatename::AbstractString, N::Int; kwargs...)

Create a layer of gates.
"""
gatelayer(gatename::AbstractString, N::Int; kwargs...) =
  Tuple[isempty(kwargs) ? (gatename, n) : (gatename, n, values(kwargs)) for n in 1:N]

"""
    appendlayer!(gates::AbstractVector{ <: Tuple},
                 gatename::AbstractString, N::Int)

Append a layer of gates to a gate list.
"""
appendlayer!(gates::AbstractVector{ <: Tuple},
             gatename::AbstractString, N::Int) =
  append!(gates, gatelayer(gatename, N))

# TODO: replace with gatelayer(gatename, bonds; nqubit = 2)
# bonds could be:
# Union{Int, AbstractRange, Vector{Int}}
# to specify the starting location.
"""
    twoqubitlayer(gatename::String,bonds::Array)

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
function randomcircuit(N::Int,depth::Int,twoqubit_bonds::Array;
                       twoqubitgate   = "CX",
                       onequbitgates  = ["Rn"])
  gates = Tuple[]
  numgates_1q = length(onequbitgates)
  
  for d in 1:depth
    cycle = twoqubit_bonds[(d-1)%length(twoqubit_bonds)+1]
    twoqubitlayer!(gates,twoqubitgate,cycle) 
    for j in 1:N
      onequbitgatename = onequbitgates[rand(1:numgates_1q)]
      if onequbitgatename == "Rn"
        g = ("Rn", j, (θ = π*rand(), ϕ = 2*π*rand(), λ = 2*π*rand()))
      elseif onequbitgatename == "randU"
        g = ("randU", j, (random_matrix = randn(ComplexF64, 2, 2),))
      else
        g = (onequbitgatename, j)
      end
      push!(gates,g) 
    end
  end
  return gates
end

"""
    randomcircuit(N::Int,depth::Int;
                  twoqubitgate   = "CX",
                  onequbitgates  = ["Rn"])

Build a 1-D random quantum circuit with `N` qubits and depth `depth`.

# Circuit:

O   O   O   O   O   O  …

# Gates:
O ▭ O   O ▭ O   O ▭ O  … (Cycle 1)
O   O ▭ O   O ▭ O   O  … (Cycle 2)

"""
function randomcircuit(N::Int,depth::Int;
                       twoqubitgate   = "CX",
                       onequbitgates  = ["Rn"])
  twoqubit_bonds = lineararray(N)
  return randomcircuit(N,depth,twoqubit_bonds;
                       twoqubitgate=twoqubitgate,
                       onequbitgates=onequbitgates)
end

"""
    randomcircuit(Lx::Int,Ly::Int,depth::Int;
                  twoqubitgate   = "CX",
                  onequbitgates  = ["Rn"])

Build a 2-D random quantum circuit with `N` qubits and depth `depth`.

# 4x4 Circuit:
O   O   O   O 
  O   O   O   O
O   O   O   O
  O   O   O   O

# Gates:
    Cycle 1               Cycle 2
O   O   O   O         O   O   O   O         
 ╲   ╲   ╲   ╲           ╱   ╱   ╱           
  O   O   O   O         O   O   O   O       
                                            
O   O   O   O         O   O   O   O         
 ╲   ╲   ╲   ╲           ╱   ╱   ╱          
  O   O   O   O         O   O   O   O       

    Cycle 3               Cycle 4
O   O   O   O         O   O   O   O          
                      
  O   O   O   O         O   O   O   O
   ╲   ╲   ╲               ╱   ╱   ╱
O   O   O   O         O   O   O   O   
                      
  O   O   O   O         O   O   O   O 


"""
function randomcircuit(Lx::Int,Ly::Int,depth::Int;
                       twoqubitgate   = "CX",
                       onequbitgates  = ["Rn"])
  twoqubit_bonds = squarearray(Lx,Ly)
  N = Lx * Ly
  return randomcircuit(N,depth,twoqubit_bonds;
                       twoqubitgate=twoqubitgate,
                       onequbitgates=onequbitgates)
end

"""
    qft(N::Int)

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
