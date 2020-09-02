"""----------------------------------------------
                  INITIALIZATION 
------------------------------------------------- """

"""
Initialize MPS wavefunction |ψ⟩
"""
wavefunction(sites::Vector{<:Index}) = productMPS(sites, "0")

wavefunction(N::Int) = wavefunction(siteinds("qubit", N))

""" 
Initialize MPO density matrix ρ
"""
densitymatrix(sites::Vector{<:Index}) = 
  densitymatrix(productMPS(sites, "0"))

densitymatrix(N::Int) = 
  densitymatrix(siteinds("qubit",N))

""" 
Initialize qubits
"""
qubits(sites::Vector{<:Index}; mixed::Bool=false) = 
  mixed ? densitymatrix(sites) : wavefunction(sites) 

qubits(N::Int; mixed::Bool=false) = qubits(siteinds("qubit", N); mixed=mixed)

""" 
Reset qubits to the initial state
"""
function resetqubits!(M::Union{MPS,MPO})
  indices = [firstind(M[j],tags="Site",plev=0) for j in 1:length(M)]
  M_new = (typeof(M) == MPS ? wavefunction(indices) : densitymatrix(indices))
  M[:] = M_new
  return M
end

"""
Build a density matrix ρ = |ψ⟩⟨ψ|
"""
function densitymatrix(ψ::MPS)
  ρ = MPO([ψ[n]' * dag(ψ[n]) for n in 1:length(ψ)])
  for n in 1:length(ρ)-1 
    C = combiner(commoninds(ρ[n], ρ[n+1]))
    ρ[n] *= C
    ρ[n+1] *= dag(C)
  end
  return ρ
end

"""
Initialize a circuit MPO
"""
circuit(sites::Vector{<:Index}) = MPO(sites, "Id")

circuit(N::Int) = circuit(siteinds("qubit", N))

"""----------------------------------------------
                  CIRCUIT FUNCTIONS 
------------------------------------------------- """

"""
Generates a vector of ITensors from a tuple of gates
"""
function compilecircuit(M::Union{MPS,MPO},gates::Vector{<:Tuple}; 
                        noise=nothing, kwargs...)
  gate_tensors = ITensor[]
  for g in gates
    push!(gate_tensors, gate(M, g))
    ns = g[2]
    if !isnothing(noise)
      if ns isa Int
        push!(gate_tensors, gate(M, noise, g[2]; kwargs...))
      else
        for n in ns
          push!(gate_tensors, gate(M, noise, n; kwargs...))
        end
      end
    end
  end
  return gate_tensors
end

"""
Apply the circuit to a state M using a set of gate tensors
"""
function runcircuit(M::Union{MPS,MPO},gate_tensors::Vector{<:ITensor}; kwargs...) 
  # Check if gate_tensors contains Kraus operators
  inds_sizes = [length(inds(g)) for g in gate_tensors]
  noiseflag = any(x -> x%2==1 , inds_sizes)
  
  # Run a noisy circuit, generating an output density operator (MPO)
  if noiseflag
    ρ = (typeof(M) == MPS ? densitymatrix(M) : M)
    Mc = apply(reverse(gate_tensors),ρ; apply_dag=true, kwargs...)
  # Run a noiseless circuit, genereating either a wavefunction (MPS) of density operator (MPO)
  else
    Mc = (typeof(M) == MPS ? apply(reverse(gate_tensors),M; kwargs...) :
                             apply(reverse(gate_tensors), M; apply_dag=true, kwargs...))
  end
  return Mc
end

"""
Apply the circuit on state M using a set of gates (Tuple)
"""
function runcircuit(M::Union{MPS,MPO},gates::Vector{<:Tuple}; noise=nothing, 
                    cutoff=1e-15,maxdim=10000,kwargs...)
  gate_tensors = compilecircuit(M,gates; noise=noise, kwargs...) 
  runcircuit(M,gate_tensors; cutoff=cutoff,maxdim=maxdim,kwargs...)
end

function splitchoi(Λ::MPO;noise=nothing,cutoff=1e-15,maxdim=1000)
  T = ITensor[]
  U,S,V = (isnothing(noise) ? svd(Λ[1],firstind(Λ[1],tags="Input"), cutoff=cutoff, maxdim=maxdim) :
                              svd(Λ[1],    inds(Λ[1],tags="Input"), cutoff=cutoff, maxdim=maxdim)  )

  push!(T,U*S)
  push!(T,V)
  for j in 2:length(Λ)
    U,S,V = (isnothing(noise) ? svd(Λ[j],firstind(Λ[j],tags="Input"),commonind(Λ[j-1],Λ[j]),cutoff=cutoff,maxdim=maxdim) :
                                svd(Λ[j],    inds(Λ[j],tags="Input")[1],inds(Λ[j],tags="Input")[2],
                                    commonind(Λ[j-1],Λ[j]),cutoff=cutoff,maxdim=maxdim)  )
    push!(T,U*S)
    push!(T,V)
  end
  Λ_split = (isnothing(noise) ? MPS(T) : MPO(T))
  return Λ_split
end


function choimatrix(N::Int,gates::Vector{<:Tuple};noise=nothing,
                    cutoff=1e-15,maxdim=10000,kwargs...)
  if isnothing(noise)
    # Initialize circuit MPO 
    U0 = circuit(N)
    # Compile gate tensors
    gate_tensors = compilecircuit(U0, gates; noise=nothing, kwargs...)
    # Build MPO for unitary circuit
    U = apply(reverse(gate_tensors), U0; cutoff=cutoff, maxdim=maxdim)
    # Introduce new Choi index notation
    addtags!(U,"Input", plev=1,tags="qubit")
    addtags!(U,"Output",plev=0,tags="qubit")
    noprime!(U)
    # SVD to bring into 2N-sites MPS
    Λ = splitchoi(U,noise=nothing,cutoff=cutoff,maxdim=maxdim)
  else
    # Initialize circuit MPO
    U = circuit(N)
    # Initialize input state
    ρ = qubits(firstsiteinds(U,plev=0),mixed=true)
    
    addtags!(U,"Input",plev=1,tags="qubit")
    addtags!(U,"Output",plev=0,tags="qubit")
    noprime!(U,tags="Input")
    
    compiler = circuit(firstsiteinds(U,tags="Output"))
    gate_tensors = compilecircuit(compiler, gates; noise=nothing, kwargs...)


    ρ[1] = U[1] * prime(U[1])
    Cdn = combiner(inds(ρ[1],tags="Link")[1],inds(ρ[1],tags="Link")[2],
                  tags="Link,n=1")
    ρ[1] = ρ[1] * Cdn
    for j in 2:N-1
      ρ[j] = U[j] * prime(U[j])
      Cup = Cdn
      Cdn = combiner(inds(ρ[j],tags="Link,n=$j")[1],inds(ρ[j],tags="Link,n=$j")[2],tags="Link,n=$j")
      ρ[j] = ρ[j] * Cup * Cdn
    end
    ρ[N] = U[N] * prime(U[N])
    ρ[N] = ρ[N] * Cdn
    
    Λ0 = apply(reverse(gate_tensors), ρ; apply_dag=true,cutoff=cutoff, maxdim=maxdim)
    Λ = splitchoi(Λ0,noise=noise,cutoff=cutoff,maxdim=maxdim)
  end
  return Λ
end

function runcircuit(N::Int,gates::Vector{<:Tuple}; noise=nothing,
                    unitary=false,process=false,
                    cutoff=1e-15,maxdim=10000,kwargs...)
  # Compute the unitary matrix of the circuit
  if unitary
    U0 = circuit(N)
    gate_tensors = compilecircuit(U0,gates; noise=nothing, kwargs...)
    return apply(reverse(gate_tensors),U0; kwargs...)
  # Compute the Choi maitrx
  elseif process
    return choimatrix(N,gates;noise=noise,cutoff=1e-15,maxdim=10000,kwargs...)
  # Compute the output quantum state from |000>
  else
    ψ0 = qubits(N)
    return runcircuit(ψ0,gates;noise=noise,cutoff=cutoff,maxdim=maxdim,kwargs...)
  end
end


"""
Apply the circuit to a ITensor 
"""
function runcircuit(M::ITensor,gates::Vector{<:Tuple}; cutoff=1e-15,maxdim=10000,kwargs...)
  gate_tensors = compilecircuit(M,gates)
  return runcircuit(M,gate_tensors;cutoff=1e-15,maxdim=10000,kwargs...)
end

runcircuit(M::ITensor,
           gate_tensors::Vector{ <: ITensor};
           kwargs...) =
  apply(reverse(gate_tensors), M; kwargs...)





"""----------------------------------------------
               MEASUREMENT FUNCTIONS 
------------------------------------------------- """

"""
Given as input a measurement basis, returns the corresponding
gate data structure.
Example:
basis = ["X","Z","Z","Y"]
-> gate_list = [("measX", 1),
                ("measY", 4)]
"""
function makemeasurementgates(basis::Array)
  gate_list = Tuple[]
  for j in 1:length(basis)
    if (basis[j]!= "Z")
      push!(gate_list,("meas$(basis[j])", j))
    end
  end
  return gate_list
end

"""
Given as input a preparation state, returns the corresponding
gate data structure.
Example:
prep = ["X+","Z+","Z+","Y+"]
-> gate_list = [("prepX+", 1),
                ("prepY+", 4)]
"""
function makepreparationgates(prep::Array)
  gate_list = Tuple[]
  for j in 1:length(prep)
    if (prep[j]!= "Z+")
      gatename = "prep$(prep[j])"
      push!(gate_list, (gatename, j))
    end
  end
  return gate_list
end

"""
Generate a set of measurement bases:
- nshots = total number of bases
if numbases=nothing: nshots different bases
"""
function generatemeasurementsettings(N::Int,numshots::Int;
                                     numbases=nothing,bases_id=nothing)
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
        push!(newdata,"stateX+")
      else
        push!(newdata,"stateX-")
      end
    elseif basis[j] == "Y"
      if datapoint[j] == 0
        push!(newdata,"stateY+")
      else
        push!(newdata,"stateY-")
      end
    elseif basis[j] == "Z"
      if datapoint[j] == 0
        push!(newdata,"stateZ+")
      else
        push!(newdata,"stateZ-")
      end
    end
  end
  return newdata
end

