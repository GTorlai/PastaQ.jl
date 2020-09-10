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
  MPO(productMPS(sites, "0"))

densitymatrix(N::Int) = 
  MPO(siteinds("qubit",N))

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
Apply the circuit to a ITensor from a list of tensors 
"""
runcircuit(M::ITensor,
           gate_tensors::Vector{ <: ITensor};
           kwargs...) =
  apply(gate_tensors, M; kwargs...)

"""
Apply the circuit to a ITensor from a list of gates 
"""
function runcircuit(M::ITensor,gates::Vector{<:Tuple}; cutoff=1e-15,maxdim=10000,kwargs...)
  gate_tensors = compilecircuit(M,gates)
  return runcircuit(M,gate_tensors;cutoff=1e-15,maxdim=10000,kwargs...)
end

"""
Apply the circuit to a state (wavefunction/densitymatrix) from a list of tensors
"""
function runcircuit(M::Union{MPS,MPO},gate_tensors::Vector{<:ITensor}; kwargs...) 
  # Check if gate_tensors contains Kraus operators
  inds_sizes = [length(inds(g)) for g in gate_tensors]
  noiseflag = any(x -> x%2==1 , inds_sizes)
  
  apply_dag = get(kwargs,:apply_dag,nothing)
  cutoff    = get(kwargs,:cutoff,1E-15)
  maxdim    = get(kwargs,:maxdim,10000)

  if apply_dag==false & noiseflag==true
    error("noise simulation requires apply_dag=true")
  end
 
  # Default mode (apply_dag = nothing)
  if isnothing(apply_dag)
    # Noisy evolution: MPS/MPO -> MPO
    if noiseflag
      # If M is an MPS, |ψ⟩ -> ρ = |ψ⟩⟨ψ| (MPS -> MPO)
      ρ = (typeof(M) == MPS ? MPO(M) : M)
      # ρ -> ε(ρ) (MPO -> MPO, conjugate evolution)
      Mc = apply(gate_tensors,ρ; apply_dag=true, cutoff=cutoff,maxdim=maxdim)
    # Pure state evolution
    else
      # |ψ⟩ -> U |ψ⟩ (MPS -> MPS)
      #  ρ  -> U ρ U† (MPO -> MPO, conjugate evolution)
      Mc = (typeof(M) == MPS ? apply(gate_tensors, M; apply_dag=false,cutoff=cutoff,maxdim=maxdim) :
                               apply(gate_tensors, M; apply_dag=true,cutoff=cutoff,maxdim=maxdim))
    end
  # Custom mode (apply_dag = true / false)
  else
    if typeof(M) == MPO
      # apply_dag = true:  ρ -> U ρ U† (MPO -> MPO, conjugate evolution)
      # apply_dag = false: ρ -> U ρ (MPO -> MPO)
      Mc = apply(gate_tensors, M; apply_dag=apply_dag,kwargs...)
    elseif typeof(M) == MPS
      # apply_dag = true:  ψ -> U ψ -> ρ = (U ψ) (ψ† U†) (MPS -> MPO, conjugate)
      # apply_dag = false: ψ -> U ψ (MPS -> MPS)
      Mc = (apply_dag ? MPO(apply(gate_tensors, M; apply_dag=false,cutoff=cutoff,maxdim=maxdim)) :
                        apply(gate_tensors, M; apply_dag=apply_dag=false,cutoff=cutoff,maxdim=maxdim))
    else
      error("Input state must be an MPS or an MPO")
    end
  end
  return Mc
end

"""
Apply the circuit to a state (wavefunction/densitymatrix) from a list of gates
"""
function runcircuit(M::Union{MPS,MPO},gates::Vector{<:Tuple}; noise=nothing,apply_dag=nothing, 
                    cutoff=1e-15,maxdim=10000,kwargs...)
  gate_tensors = compilecircuit(M,gates; noise=noise, kwargs...) 
  runcircuit(M,gate_tensors; cutoff=cutoff,maxdim=maxdim,apply_dag=apply_dag, kwargs...)
end

"""
Empty run of the quantum circuit
"""
function runcircuit(N::Int,gates::Vector{<:Tuple}; process=false,noise=nothing,
                    cutoff=1e-15,maxdim=10000,kwargs...)
  if process==false
    ψ = qubits(N) # = |0,0,0,…,0⟩
    # noiseless: ψ -> U ψ
    # noisy:     ψ -> ρ = ε(|ψ⟩⟨ψ|)
    return runcircuit(ψ,gates;noise=noise,cutoff=cutoff,maxdim=maxdim,kwargs...)
  
  elseif process==true & isnothing(noise)==true
    if isnothing(noise)
      U = circuit(N) # = 1⊗1⊗1⊗…⊗1
      return runcircuit(U,gates;noise=nothing,apply_dag=false,cutoff=cutoff,maxdim=maxdim,kwargs...) 
    else
      error("Cannot build the circuit MPO if noise!=nothing")
    end
  end
    
end

"""
Compute the Choi matrix
"""
function choimatrix(N::Int,gates::Vector{<:Tuple};noise=nothing,apply_dag=false,
                    cutoff=1e-15,maxdim=10000,kwargs...)
  if isnothing(noise)
    # Get circuit MPO
    U = runcircuit(N,gates;process=true,cutoff=1e-15,maxdim=10000,kwargs...)
    
    # Choi indices 
    addtags!(U,"Input", plev=0,tags="qubit")
    addtags!(U,"Output",plev=1,tags="qubit")
    noprime!(U)
    # SVD to bring into 2N-sites MPS
    Λ0 = splitchoi(U,noise=nothing,cutoff=cutoff,maxdim=maxdim)
    # if apply_dag = true:  Λ = |U⟩⟩ ⟨⟨U†|
    # if apply_dag = false: Λ = |U⟩⟩
    Λ = (apply_dag ? MPO(Λ0) : Λ0)
  else
    # Initialize circuit MPO
    U = circuit(N)
    addtags!(U,"Input",plev=0,tags="qubit")
    addtags!(U,"Output",plev=1,tags="qubit")
    prime!(U,tags="Input")
    prime!(U,tags="Link")
    
    s = [siteinds(U,tags="Output")[j][1] for j in 1:length(U)]
    compiler = circuit(s)
    prime!(compiler,-1,tags="qubit") 
    gate_tensors = compilecircuit(compiler, gates; noise=noise, kwargs...)

    M = ITensor[]
    push!(M,U[1] * noprime(U[1]))
    Cdn = combiner(inds(M[1],tags="Link")[1],inds(M[1],tags="Link")[2],
                  tags="Link,n=1")
    M[1] = M[1] * Cdn
    for j in 2:N-1
      push!(M,U[j] * noprime(U[j]))
      Cup = Cdn
      Cdn = combiner(inds(M[j],tags="Link,n=$j")[1],inds(M[j],tags="Link,n=$j")[2],tags="Link,n=$j")
      M[j] = M[j] * Cup * Cdn
    end
    push!(M, U[N] * noprime(U[N]))
    M[N] = M[N] * Cdn
    ρ = MPO(M)
    Λ0 = runcircuit(ρ,gate_tensors;apply_dag=true,cutoff=cutoff, maxdim=maxdim)
    Λ = splitchoi(Λ0,noise=noise,cutoff=cutoff,maxdim=maxdim)
  end
  return Λ
end

function splitchoi(Λ::MPO;noise=nothing,cutoff=1e-15,maxdim=1000)
  T = ITensor[]
  if isnothing(noise)
    u,S,v = svd(Λ[1],firstind(Λ[1],tags="Input"), 
                cutoff=cutoff, maxdim=maxdim)
  else
    u,S,v = svd(Λ[1],inds(Λ[1],tags="Input"), 
                cutoff=cutoff, maxdim=maxdim)
  end
  push!(T,u*S)
  push!(T,v)
  
  for j in 2:length(Λ)
    if isnothing(noise)
      u,S,v = svd(Λ[j],firstind(Λ[j],tags="Input"),commonind(Λ[j-1],Λ[j]),
                  cutoff=cutoff,maxdim=maxdim)
    else
      u,S,v = svd(Λ[j],inds(Λ[j],tags="Input")[1],inds(Λ[j],tags="Input")[2],
                  commonind(Λ[j-1],Λ[j]),cutoff=cutoff,maxdim=maxdim) 
    end
    push!(T,u*S)
    push!(T,v)
  end
  Λ_split = (isnothing(noise) ? MPS(T) : MPO(T))
  return Λ_split
end

#"""----------------------------------------------
#               MEASUREMENT FUNCTIONS 
#------------------------------------------------- """
#
#"""
#Given as input a measurement basis, returns the corresponding
#gate data structure.
#Example:
#basis = ["X","Z","Z","Y"]
#-> gate_list = [("measX", 1),
#                ("measY", 4)]
#"""
#function makemeasurementgates(basis::Array)
#  gate_list = Tuple[]
#  for j in 1:length(basis)
#    if (basis[j]!= "Z")
#      push!(gate_list,("meas$(basis[j])", j))
#    end
#  end
#  return gate_list
#end
#
#"""
#Given as input a preparation state, returns the corresponding
#gate data structure.
#Example:
#prep = ["X+","Z+","Z+","Y+"]
#-> gate_list = [("prepX+", 1),
#                ("prepY+", 4)]
#"""
#function makepreparationgates(prep::Array)
#  gate_list = Tuple[]
#  for j in 1:length(prep)
#    if (prep[j]!= "Z+")
#      gatename = "prep$(prep[j])"
#      push!(gate_list, (gatename, j))
#    end
#  end
#  return gate_list
#end
#
#"""
#Generate a set of measurement bases:
#- nshots = total number of bases
#if numbases=nothing: nshots different bases
#"""
#function generatemeasurementsettings(N::Int,numshots::Int;
#                                     numbases=nothing,bases_id=nothing)
#  if isnothing(bases_id)
#    bases_id = ["X","Y","Z"]
#  end
#  # One shot per basis
#  if isnothing(numbases)
#    measurementbases = rand(bases_id,numshots,N)
#  else
#    @assert(numshots%numbases ==0)
#    shotsperbasis = numshots÷numbases
#    measurementbases = repeat(rand(bases_id,1,N),shotsperbasis)
#    for n in 1:numbases-1
#      newbases = repeat(rand(bases_id,1,N),shotsperbasis)
#      measurementbases = vcat(measurementbases,newbases)
#    end
#  end
#  return measurementbases
#end
#
#"""
#Generate a set of preparation states:
#- nshots = total number of states
#if numprep=nothing: nshots different states
#"""
#function generatepreparationsettings(N::Int,numshots::Int;
#                                     numprep=nothing,prep_id=nothing)
#  if isnothing(prep_id)
#    prep_id = ["X+","X-","Y+","Y-","Z+","Z-"]
#  end
#  # One shot per basis
#  if isnothing(numprep)
#    preparationstates = rand(prep_id,numshots,N)
#  else
#    @assert(numshots%numprep ==0)
#    shotsperstate = numshots÷numprep
#    preparationstates = repeat(rand(prep_id,1,N),shotsperstate)
#    for n in 1:numprep-1
#      newstates = repeat(rand(prep_id,1,N),shotsperstate)
#      preparationstates = vcat(preparationstates,newstates)
#    end
#  end
#  return preparationstates
#end
#
#"""
#Perform a projective measurements on a wavefunction
#"""
#function measure(M::Union{MPS,MPO},nshots::Int)
#  orthogonalize!(M,1)
#  if (nshots==1)
#    measurements = sample(M)
#    measurements .-= 1
#  else
#    measurements = Matrix{Int64}(undef, nshots, length(M))
#    for n in 1:nshots
#      measurement = sample(M)
#      measurement .-= 1
#      measurements[n,:] = measurement
#    end
#  end
#  return measurements
#end
#
#"""
#Generate a dataset of measurements in different bases
#"""
#function generatedata(M0::Union{MPS,MPO},nshots::Int,bases::Array)
#  data = Matrix{String}(undef, nshots,length(M0))
#  for n in 1:nshots
#    meas_gates = makemeasurementgates(bases[n,:])
#    meas_tensors = compilecircuit(M0,meas_gates)
#    M = runcircuit(M0,meas_tensors)
#    measurement = measure(M,1)
#    data[n,:] = convertdata(measurement,bases[n,:])
#  end
#  return data 
#end
#
#function generatedata(M::Union{MPS,MPO},nshots::Int64)
#  bases = generatemeasurementsettings(length(M),nshots,bases_id=["X","Y","Z"])
#  generatedata(M,nshots,bases)
#end
#
##function generatedata(N::Int,gates::Array,nshots::Int,prep::Array,bases::Array)
##
##  data_in  = Matrix{String}(undef, nshots,length(M0))
##  data_out = Matrix{String}(undef, nshots,length(M0))
##  
##  for n in 1:nshots
##    prep_gates = makepreparationgates(prep[n,:])
##    ψ_in = runcircuit(N,prep_gates)
##    
##    M = runcircuit(ψ_in,gates
##    #meas_gates = makemeasurementgates(bases[n,:])
##  end
##end
#
