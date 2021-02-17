abstract type QuantumCode end

"""
    SurfaceCode(d::Int64)

Generate data structure for distance-`d` surface code
"""
struct SurfaceCode <: QuantumCode
  d::Int64
  Qcoord::Vector
  Xcoord::Vector
  Zcoord::Vector
  QonS::NamedTuple
  SonQ::Vector
end

function SurfaceCode(d::Int64)
  #@assert isodd(d)
  Qcoord = []
  Xcoord = []
  Zcoord = []

  # build coordinates of qubits
  for y in 1:2*d-1
    if isodd(y)
      for x in 1:2:2*d 
        push!(Qcoord,(x,y))
      end
    else
      for x in 2:2:2*d-1
        push!(Qcoord,(x,y))
      end
    end
  end

  # build coordinates of stabilizers
  Zcoord = vec(Iterators.product(1:2:2*d-1, 2:2:2*d-1)|>collect) 
  Xcoord = vec(Iterators.product(2:2:2*d-1, 1:2:2*d-1)|>collect)


  # build list of qubits connected to each stabilizer
  stabX = []
  stabZ = []
  # X stabilizers
  for y in 1:2:2*d-1
    for x in 2:2:2*(d-1)
      if y == 1
        # lower smooth boundary
        push!(stabX,[qubit_at(x-1,y,d), qubit_at(x,y+1,d), qubit_at(x+1,y,d)]) 
      elseif y == 2*d-1
        # upper smooth boundary
        push!(stabX,[qubit_at(x-1,y,d), qubit_at(x,y-1,d), qubit_at(x+1,y,d)])
      else
        # bulk stabilizers
        push!(stabX,[qubit_at(x-1,y,d), qubit_at(x,y+1,d), qubit_at(x+1,y,d),qubit_at(x,y-1,d)])
      end
    end
  end

  # Z stabilizers
  for y in 2:2:2*d-1
    for x in 1:2:2*d-1
      if x == 1
        # left rough boundary
        push!(stabZ,[qubit_at(x,y-1,d), qubit_at(x+1,y,d), qubit_at(x,y+1,d)])
      elseif x == 2*d-1
        # right rough boundary
        push!(stabZ,[qubit_at(x,y-1,d), qubit_at(x-1,y,d), qubit_at(x,y+1,d)])
      else
        # bulk stabilizers
        push!(stabZ,[qubit_at(x,y-1,d), qubit_at(x-1,y,d), qubit_at(x,y+1,d), qubit_at(x+1,y,d)])
      end
    end
  end

  QonS = (X = stabX, Z = stabZ)
  SonQ = []
  
  # build the stabilizers around each qubit
  # left-bottom corner
  push!(SonQ, (X = [1], Z = [1]))
  # bottom row
  for x in 3:2:2*(d-1)-1
    push!(SonQ, (X = [stabX_at(x-1,1,d),stabX_at(x+1,1,d)], Z = [stabZ_at(x,2,d)]))
  end
  # right-bottom corner
  push!(SonQ, (X = [stabX_at(2*d-2,1,d)], Z = [stabZ_at(2*d-1,2,d)]))
  
  # loop over
  for i in 1:d-2
    y = 2*i
    for x in 2:2:2*d-1
      push!(SonQ, (X = [stabX_at(x,y-1,d), stabX_at(x,y+1,d)] , Z = [stabZ_at(x-1,y,d),stabZ_at(x+1,y,d)]))
    end
    y = 2*i +1
    push!(SonQ, (X = [stabX_at(2,y,d)], Z = [stabZ_at(1,y+1,d),stabZ_at(1,y-1,d)]))
    for x in 3:2:2*(d-1)-1
      push!(SonQ, (X = [stabX_at(x-1,y,d),stabX_at(x+1,y,d)], Z = [stabZ_at(x,y+1,d),stabZ_at(x,y-1,d)]))
    end
    push!(SonQ, (X = [stabX_at(2*d-2,y,d)], Z = [stabZ_at(2*d-1,y+1,d),stabZ_at(2*d-1,y-1,d)]))
  end
  y = 2*(d-1)
  for x in 2:2:2*d-1
    push!(SonQ, (X = [stabX_at(x,y-1,d), stabX_at(x,y+1,d)] , Z = [stabZ_at(x-1,y,d),stabZ_at(x+1,y,d)]))
  end
  
  y = 2*d-1
  push!(SonQ, (X = [stabX_at(2,y,d)], Z = [stabZ_at(1,y-1,d)]))
  for x in 3:2:2*(d-1)-1
    push!(SonQ, (X = [stabX_at(x-1,y,d),stabX_at(x+1,y,d)], Z = [stabZ_at(x,y-1,d)]))
  end
  push!(SonQ, (X = [stabX_at(2*d-2,y,d)], Z = [stabZ_at(2*d-1,y-1,d)]))
  
  return SurfaceCode(d,Qcoord,Xcoord,Zcoord,QonS,SonQ)
end

"""
Return code distance
"""
distance(code::SurfaceCode) = code.d

"""
Return number of qubits
"""
nqubits(code::SurfaceCode) = distance(code)^2+(distance(code)-1)^2

"""
Return coordinates of qubits
"""
coordinates_ofQubits(code::SurfaceCode) = code.Qcoord 
coordinates_ofQubits(code::SurfaceCode, qubit::Int) = code.Qcoord[qubit] 

"""
Return coordinates of stabilizers
"""
coordinates_ofStabX(code::SurfaceCode) = code.Xcoord 
coordinates_ofStabZ(code::SurfaceCode) = code.Zcoord

coordinates_ofStabX(code::SurfaceCode,stab::Int) = code.Xcoord[stab] 
coordinates_ofStabZ(code::SurfaceCode,stab::Int) = code.Zcoord[stab]

"""
Return qubits appearing in the stabilizer
"""
qubits_atStab(code::SurfaceCode) = code.QonS
qubits_atStabX(code::SurfaceCode) = code.QonS[:X]
qubits_atStabZ(code::SurfaceCode) = code.QonS[:Z]

qubits_atStab(code::SurfaceCode, stab::Int)  = code.QonS[stab]
qubits_atStabX(code::SurfaceCode, stab::Int) = code.QonS[:X][stab] 
qubits_atStabZ(code::SurfaceCode, stab::Int) = code.QonS[:Z][stab]

"""
Return the stabilizers connected to the qubits
"""
stab_atQubit(code::SurfaceCode)  = code.SonQ
stabX_atQubit(code::SurfaceCode) = first.(code.SonQ)
stabZ_atQubit(code::SurfaceCode) = last.(code.SonQ)

stab_atQubit(code::SurfaceCode, qubit::Int)  = code.SonQ[qubit]
stabX_atQubit(code::SurfaceCode, qubit::Int) = code.SonQ[qubit][:X]
stabZ_atQubit(code::SurfaceCode, qubit::Int) = code.SonQ[qubit][:Z]

"""
    qubit_at(code::SurfaceCode, x::Int64, y::Int64)

Return the index of qubit with coordinates `(x,y)`.

Example:

     d = 3 Surface Code
 y    
 ↑      ............ 
 |
 |   6̂       7̂       8̂
 |  1,3     1,5     1,7  
 |       4̂       5̂    
 |      2,2     2,4
 |   1̂       2̂       3̂     
 |  1,1     3,1     5,1  
  - - - - - - - - - - - - - → x

"""
function qubit_at(x::Int64, y::Int64, d::Int64)
  (x < 1 || x > 2*d-1) && error("x out of bounds")
  (y < 1 || y > 2*(d)-1) && error("y out of bounds")
  (isodd(x) && isodd(y)) && return (2*d-1)*(y-1)÷2 +(x+1)÷2
  (iseven(x) && iseven(y)) && return (2*d-1)*(y-2)÷2 + x÷2 + d 
  error("No data qubit at localtion (",x,",",y,")")
end

qubit_at(code::SurfaceCode, x::Int64, y::Int64) = 
  qubit_at(x, y, distance(code))

qubit_at(code::SurfaceCode, coordinates_list::Vector{Tuple{Int64, Int64}}) = 
  [qubit_at(code, coordinates...) for coordinates in coordinates_list]

"""
    stabX_at(code::SurfaceCode,x::Int64, y::Int64)

Returns the index of a X stabilizer located at the vertex
with coordinates `(x,y)`.

Example:

     d = 3 Surface Code
 y    
 ↑      ............ 
 |
 |   o   3   o   4    o
 |     (2,3)   (4,3) 
 |       o       o    
 |   
 |   o   1   o   2   o    
 |     (2,1)   (4,1)
  - - - - - - - - - - - - - → x
"""
function stabX_at(x::Int64, y::Int64, d::Int64)
  (x < 2 || x > 2*(d-1)) && error("x out of bounds")
  (y < 1 || y > 2*d-1) && error("y out of bounds")
  (iseven(x) && isodd(y)) && return (d-1)*(y-1)÷2+x÷2 
  error("No X stabilizer at  localtion (",x,",",y,")")
end

stabX_at(code::SurfaceCode, x::Int64, y::Int64) =
  stabX_at(x, y, distance(code))
"""
    stabZ_at(code::SurfaceCode,x::Int64, y::Int64)

Returns the index of a Z stabilizer located at the plaquette
with coordinates `(x,y)`.

Example:

     d = 3 Surface Code
 y    
 ↑      ............ 
 |
 |   1    o  2    o   3   
 | (1,2)   (1,3)    (1,5) 
 |   o       o       o    
  - - - - - - - - - - - - - → x
"""
function stabZ_at(x::Int64, y::Int64, d::Int64)
  (x < 1 || x > 2*d-1) && error("x out of bounds")
  (y < 1 || y > 2*(d-1)) && error("y out of bounds")
  (isodd(x) && iseven(y)) && return d*(y-2)÷2+x÷2+1 
  error("No X stabilizer at  localtion (",x,",",y,")")
end

stabZ_at(code::SurfaceCode, x::Int64, y::Int64) =
  stabZ_at(x, y, distance(code))


"""
    logicaloperator(code::SurfaceCode, coset::String)
    logicaloperator(code::SurfaceCode, coset::Tuple{Int64,Int64})
    logicaloperator(code::SurfaceCode, coset::Vector{Int64})

Return a logical operator correpsonding to a given `coset` of the
surface code. The output consists of a pauli vector (built out of X
and Z bits on each qubits). Input can be either the string  or the binary
encoding of the logical operator. For the surface code:
- `I` ≡ (0,0)
- `Z` ≡ (0,1)
- `X` ≡ (1,0)
- `Y` ≡ (1,1)
"""
function logicaloperator(coset::String, code::SurfaceCode)
  n = nqubits(code)
  ## trivial
  coset == "I" && return [(0,0) for _ in 1:n] 
  
  # logical X
  Lx = support_to_pauli(nqubits(code), [y*(2*code.d-1)+1 for y in 0:code.d-1])
  coset == "X" && return [(Lx[j],0) for j in 1:n]
  
  # logical Z
  Lz = support_to_pauli(nqubits(code), 1:code.d|>collect)
  coset == "Z" && return [(0,Lz[j]) for j in 1:n] 
  
  # logical Y
  coset == "Y" && return [(Lx[j],Lz[j]) for j in 1:n] 
 
  error("Coset not recognized. Surface code cosets are: \n`I` , [0,0]\n`Z` , [0,1]\n`X` , [1,0]\n`Y` , [1,1]\n")
end

function logicaloperator(coset::Tuple{Int64,Int64}, code::SurfaceCode)
  coset == (0,0) && return logicaloperator("I", code) 
  coset == (1,0) && return logicaloperator("X", code)
  coset == (0,1) && return logicaloperator("Z", code)
  coset == (1,1) && return logicaloperator("Y", code)
  error("logical operator not recognized")
end

logicaloperator(coset::Vector{Int64}, code::SurfaceCode) = 
  logicaloperator(Tuple(coset), code)


"""
    Wilsonloops(pauli::Vector{<:Array}, code::SurfaceCode)

Return the measurement of the Wilson loops
"""
function Wilsonloops(pauli::Vector{Tuple{Int64,Int64}}, code::SurfaceCode)
  wX = sum(first.(pauli)[[2*code.d-1+x for x in 1:code.d]]) % 2
  wZ = sum(last.(pauli)[[y*(2*code.d-1)+2 for y in 0:code.d-1]]) % 2
  return (wX,wZ)
end


"""
Return the Pauli operator that moves a charge at a given location to the 
closest smooth boundary
"""
function movecharge(vertex::Int, code::SurfaceCode)
  d = distance(code)
  x0,y0 = code.Xcoord[vertex]
  
  chargepath = (x0 > distance(code)-1 ? 
                       qubit_at(code,[(x+1,y0) for x in x0:2:2*(distance(code)-1)]) :
                       qubit_at(code,[(x-1,y0) for x in x0:-2:2]))
  
  return support_to_pauli(nqubits(code), chargepath)
end

"""
Return the Pauli operator that moves a flux at a given location to the 
closest rough boundary
"""
function moveflux(plaquette::Int, code::SurfaceCode)
  d = distance(code)
  x0,y0 = code.Zcoord[plaquette]
  fluxpath = (y0 > code.d-1 ?  qubit_at(code,[(x0, y+1) for y in y0:2:2*(d-1)]) : 
                               qubit_at(code,[(x0, y-1) for y in y0:-2:2]))
  return support_to_pauli(nqubits(code), fluxpath)
end

"""
Generate a Pauli operator consistent with a given syndrome
"""
function purepaulierror(s::NamedTuple, code::SurfaceCode)
  N = nqubits(code)
  pauliX = zeros(Int64,N) 
  pauliZ = zeros(Int64,N) 
  
  charges = findall(x -> x == 1, s[:X])
  fluxes  = findall(x -> x == 1, s[:Z])
  for charge in charges
    pauliZ = pauliZ ⊙ movecharge(charge, code)
  end
  for flux in fluxes
    pauliX = pauliX ⊙ moveflux(flux, code)
  end
  return [(pauliX[j], pauliZ[j]) for j in 1:N] 
end



