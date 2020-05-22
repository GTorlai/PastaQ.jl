mutable struct QuantumCircuit
  N::Int
  seed::Int
  rng::MersenneTwister
  gates::Array
  tensors::Array
end

function QuantumCircuit(;N::Int,
                         seed::Int=1234)

  rng = MersenneTwister(seed)
  gates        = []
  tensors      = []
  
  return QuantumCircuit(N,seed,rng,
                        gates,tensors)
end

"""----------------------------------------------
                  INITIALIZATIONS 
------------------------------------------------- """

"""
initialize the wavefunction
Create an MPS for N sites on the ``|000\\dots0\\rangle`` state.
"""
function qubits(N::Int)
  sites = [Index(2; tags="Site, n=$s") for s in 1:N]
  psi = productMPS(sites, [1 for i in 1:length(sites)])
  return psi
end

"""
Reset the gates and tensor in QuantumCircuit
"""
function reset!(qc::QuantumCircuit)
  qc.gates = []
  qc.tensors = []
end

"""----------------------------------------------
                  CIRCUIT FUNCTIONS 
------------------------------------------------- """

#"""
#Add a single gate to quantumcircuit.gates
#"""
#function addgate!(gates::Array,newgate::NamedTuple)
#  push!(gates,newgate)
#end

"""
Add a list of gates to quantumcircuit.gates
"""
function addgates!(gates::Array,newgates::Array)
  for newgate in newgates
    push!(gates,newgate)
  end
end

"""
Compile the gates into tensors and return
"""
function compilecircuit(gates::Array,mps::MPS)
  tensors = []
  for gate in gates
    push!(tensors,makegate(mps,gate))
  end
  return tensors
end

"""
Compile news gates into existing tensors list 
"""
function compilecircuit!(tensors::Array,gates::Array,mps::MPS)
  for gate in gates
    push!(tensors,makegate(mps,gate))
  end
  return tensors
end

""" Run the quantumcircuit.tensors without modifying input state
"""
function runcircuit(mps::MPS,tensors::Array;cutoff=1e-10)
  return runcircuit!(copy(mps),tensors;cutoff=cutoff)
end

""" Run quantumcircuit.tensors on the input state"""
function runcircuit!(mps::MPS,tensors::Array;cutoff=1e-10)
  for gate in tensors
    applygate!(mps,gate,cutoff=cutoff)
  end
  return mps
end

"""----------------------------------------------
               MEASUREMENT FUNCTIONS 
------------------------------------------------- """

"""
Given as input a measurement basis, returns the corresponding
gate data structure.
Example:
basis = ["X","Z","Z","Y"]
-> gate_list = [(gate = "mX", site = 1),
                (gate = "mY", site = 4)]
"""
function makemeasurementgates(basis::Array)
  gate_list = []
  for j in 1:length(basis)
    if (basis[j]!= "Z")
      push!(gate_list,(gate = "m$(basis[j])", site = j))
    end
  end
  return gate_list
end


## Set up custom state preparation
#function setup_statepreparation(qc::QuantumCircuit;
#                                kwargs...)
##TODO
#
#end
#
#"""

                                
#
## Changed it so it makes namedtupled
#function statepreparationcircuit(mps::MPS,prep::Array)
#  circuit = []
#  for j in 1:N
#    if prep[j] == "Xp"
#      push!(circuit,makegate(mps,"H",j))
#    elseif prep[j] == "Xm"
#      push!(circuit,makegate(mps,"X",j))
#      push!(circuit,makegate(mps,"H",j))
#    elseif prep[j] == "Yp"
#      push!(circuit,makegate(mps,"Kp",j))
#    elseif prep[j] == "Ym"
#      push!(circuit,makegate(mps,"X",j))
#      push!(circuit,makegate(mps,"Kp",j))
#    elseif prep[j] == "Zp"
#      nothing
#    elseif prep[j] == "Zm"
#      push!(circuit,makegate(mps,"X",j))
#    end
#  end
#  return circuit
#end
#function measure(mps::MPS,nshots::Int)
#  measurements = Matrix{Int64}(undef, nshots, length(mps))
#  orthogonalize!(mps,1)
#  for n in 1:nshots
#    measurement = sample(mps)
#    measurement .-= 1
#    measurements[n,:] = measurement
#  end
#  return measurements
#end
#
#function measure(mps::MPS,nshots::Int,bases::Array)
#  measurements = Matrix{Int64}(undef, nshots, length(mps))
#  measurement_bases = Matrix{String}(undef,nshots,length(mps)) 
#  for n in 1:nshots
#    psi = copy(mps)
#    basis,gate_list = generatemeasurementcircuit(length(mps),bases)
#    circuit = makecircuit(psi,gate_list)
#    runcircuit!(psi,circuit)
#    measurement = sample!(psi)
#    measurement .-= 1
#    measurements[n,:] = measurement
#    measurement_bases[n,:] = basis
#  end
#  return measurements,measurement_bases
#end
#
#
#" INNER CIRCUITS "
#
#function hadamardlayer!(gates::Array,N::Int)
#  for j in 1:N
#    push!(gates,(gate = "H",site = j))
#  end
#end
#
#function rand1Qrotationlayer!(gates::Array,N::Int;
#                              rng=nothing)
#  for j in 1:N
#    if isnothing(rng)
#      θ,ϕ,λ = rand!(zeros(3))
#    else
#      θ,ϕ,λ = rand!(rng,zeros(3))
#    end
#    push!(gates,(gate = "Rn",site = j, params = (θ = θ, ϕ = ϕ, λ = λ)))
#  end
#end
#
#function Cxlayer!(gates::Array,N::Int,sequence::String)
#  if (N ≤ 2)
#    throw(ArgumentError("Cxlayer is defined for N ≥ 3"))
#  end
#  
#  if sequence == "odd"
#    for j in 1:2:(N-N%2)
#      push!(gates,(gate = "Cx", site = [j,j+1]))
#    end
#  elseif sequence == "even"
#    for j in 2:2:(N+N%2-1)
#      push!(gates,(gate = "Cx", site = [j,j+1]))
#    end
#  else
#    throw(ArgumentError("Sequence not recognized"))
#  end
#end
#
#"""
#    initializecircuit(N::Int)
#"""
#function initializecircuit(N::Int)
#  sites = [Index(2; tags="Site, n=$s") for s in 1:N]
#  U = MPO(sites, "Id")
#  return U
#end
#

#
#  circuit = []
#  basis = []
#  randomsamples = rand(1:length(bases),N)
#  for j in 1:N
#    localbasis = bases[randomsamples[j]]
#    push!(basis,localbasis)
#    if localbasis == "X"
#      push!(circuit,(gate = "H", site = j))
#    elseif localbasis == "Y"
#      push!(circuit,(gate = "Km", site = j))
#    elseif localbasis =="Z"
#      nothing
#    else
#      throw(argumenterror("Basis not recognized"))
#    end
#  end
#  return basis,circuit
#end
#
