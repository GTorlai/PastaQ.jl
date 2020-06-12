"""
initialize the wavefunction
Create an MPS for N sites on the ``|000\\dots0\\rangle`` state.
"""
function qubits(N::Int)
  sites = [Index(2; tags="Site, n=$s") for s in 1:N]
  psi = productMPS(sites, [1 for i in 1:length(sites)])
  return psi
end

function resetqubits!(psi::MPS)
  sites = siteinds(psi)
  psi = productMPS(sites, [1 for i in 1:length(sites)])
  return psi
end
"""----------------------------------------------
                  CIRCUIT FUNCTIONS 
------------------------------------------------- """

"""
Add a list of gates to gates (data structure) 
"""
function addgates!(gates::Array,newgates::Array)
  for newgate in newgates
    push!(gates,newgate)
  end
end

"""
Compile the gates into tensors and return
"""
function compilecircuit(mps::MPS,gates::Array)
  tensors = []
  for gate in gates
    push!(tensors,makegate(mps,gate))
  end
  return tensors
end

"""
Compile news gates into existing tensors list 
"""
function compilecircuit!(tensors::Array,mps::MPS,gates::Array)
  for gate in gates
    push!(tensors,makegate(mps,gate))
  end
  return tensors
end

""" Run the a quantum circuit without modifying input state
"""
function runcircuit(mps::MPS,tensors::Array;cutoff=1e-10)
  return runcircuit!(copy(mps),tensors;cutoff=cutoff)
end

""" Run a quantum circuit on the input state"""
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

"""
Given as input a preparation state, returns the corresponding
gate data structure.
Example:
prep = ["X+","Z+","Z+","Y+"]
-> gate_list = [(gate = "pX+", site = 1),
                (gate = "pY+", site = 4)]
"""
function makepreparationgates(prep::Array)
  gate_list = []
  for j in 1:length(prep)
    if (prep[j]!= "Zp")
      push!(gate_list,(gate = "p$(prep[j])", site = j))
    end
  end
  return gate_list
end

"""
Generate a set of measurement bases:
- nshots = total number of bases
if numbases=nothing: nshots different bases
"""
function generatemeasurementsettings(N::Int,numshots::Int;numbases=nothing,bases_id=nothing)
  if isnothing(bases_id)
    bases_id = ["X","Y","Z"]
  end
  # One shot per basis
  if isnothing(numbases)
    measurementbases = rand(bases_id,numshots,N)
  else
    @assert(numshots%numbases ==0)
    shotsperbasis = numshots÷numbases
    measurementbases = repeat(rand(bases_id,1,N),shotsperbasis)
    for n in 1:numbases-1
      newbases = repeat(rand(bases_id,1,N),shotsperbasis)
      measurementbases = vcat(measurementbases,newbases)
    end
  end
  return measurementbases
end

"""
Generate a set of preparation states:
- nshots = total number of states
if numprep=nothing: nshots different states
"""
function generatepreparationsettings(N::Int,numshots::Int;numprep=nothing,prep_id=nothing)
  if isnothing(prep_id)
    prep_id = ["X+","X-","Y+","Y-","Z+","Z-"]
  end
  # One shot per basis
  if isnothing(numprep)
    preparationstates = rand(prep_id,numshots,N)
  else
    @assert(numshots%numprep ==0)
    shotsperstate = numshots÷numprep
    preparationstates = repeat(rand(prep_id,1,N),shotsperstate)
    for n in 1:numprep-1
      newstates = repeat(rand(prep_id,1,N),shotsperstate)
      preparationstates = vcat(preparationstates,newstates)
    end
  end
  return preparationstates
end

"""
Perform a projective measurements on a wavefunction
"""
function measure(mps::MPS,nshots::Int)
  orthogonalize!(mps,1)
  if (nshots==1)
    measurements = sample(mps)
    measurements .-= 1
  else
    measurements = Matrix{Int64}(undef, nshots, length(mps))
    for n in 1:nshots
      measurement = sample(mps)
      measurement .-= 1
      measurements[n,:] = measurement
    end
  end
  return measurements
end

"""
Generate a dataset of measurements in different bases
"""
function generatedata(psi::MPS,nshots::Int,bases::Array)
  data = Matrix{String}(undef, nshots,length(psi))
  for n in 1:nshots
    meas_gates = makemeasurementgates(bases[n,:])
    meas_tensors = compilecircuit(psi,meas_gates)
    psi_out = runcircuit(psi,meas_tensors)
    measurement = measure(psi_out,1)
    data[n,:] = convertdata(measurement,bases[n,:])
  end
  return data 
end

"""
Convert a data point from (sample,basis) -> data
Ex: (0,1,0,0) (X,Z,Y,X) -> (X+,Z-,Y+,X+)
"""
function convertdata(datapoint::Array,basis::Array)
  newdata = []
  for j in 1:length(datapoint)
    if basis[j] == "X"
      if datapoint[j] == 0
        push!(newdata,"X+")
      else
        push!(newdata,"X-")
      end
    elseif basis[j] == "Y"
      if datapoint[j] == 0
        push!(newdata,"Y+")
      else
        push!(newdata,"Y-")
      end
    elseif basis[j] == "Z"
      if datapoint[j] == 0
        push!(newdata,"Z+")
      else
        push!(newdata,"Z-")
      end
    end
  end
  return newdata
end 

"""
Append a layer of Hadamard gates
"""
function hadamardlayer!(gates::Array,N::Int)
  for j in 1:N
    push!(gates,(gate = "H",site = j))
  end
end

"""
Append a layer of random single-qubit rotations
"""
function rand1Qrotationlayer!(gates::Array,N::Int;
                              rng=nothing)
  for j in 1:N
    if isnothing(rng)
      θ,ϕ,λ = rand!(zeros(3))
    else
      θ,ϕ,λ = rand!(rng,zeros(3))
    end
    push!(gates,(gate = "Rn",site = j, 
                 params = (θ = π*θ, ϕ = 2*π*ϕ, λ = 2*π*λ)))
  end
end

"""
Append a layer of CX gates
"""
function Cxlayer!(gates::Array,N::Int;sequence::String)
  if (N ≤ 2)
    throw(ArgumentError("Cxlayer is defined for N ≥ 3"))
  end
  
  if sequence == "odd"
    for j in 1:2:(N-N%2)
      push!(gates,(gate = "Cx", site = [j,j+1]))
    end
  elseif sequence == "even"
    for j in 2:2:(N+N%2-1)
      push!(gates,(gate = "Cx", site = [j,j+1]))
    end
  else
    throw(ArgumentError("Sequence not recognized"))
  end
end

"""
Generate a random quantum circuit
"""
function randomquantumcircuit(N::Int,depth::Int;rng=nothing)
  gates = []
  for d in 1:depth
    rand1Qrotationlayer!(gates,N,rng=rng)
    if d%2==1
      Cxlayer!(gates,N,sequence="odd")
    else
      Cxlayer!(gates,N,sequence="even")
    end
  end
  return gates
end

