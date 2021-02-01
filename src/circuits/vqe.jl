struct VariationalQuantumEigensolver
  Hamiltonian::MPO
  circuit::Vector{Vector{<:Tuple}}
  parameters::Array
end

VQE(H::MPO) = VariationalQuantumEigensolver(H,[],[])

function VQE(H::MPO, circuit::Vector{<:Vector{<:Tuple}}) 
  N = length(H)
  @assert N ≤ numberofqubits(circuit)
  depth = length(circuit)
  parameters = [] 
  
  # loop over the layers
  for d in 1:depth
    layer = circuit[d] 
    # find where there are parametrized gates
    parametrizedgate_location = findall(x -> x==1, length.(layer) .== 3)    
    parametrizedgates = layer[parametrizedgate_location]
    # remove gates with the arg `nograd=true` 
    mask = .!BitArray(haskey.(last.(parametrizedgates),:nograd))
    trainablegates = parametrizedgates[findall(x->x==1, mask .== 1)]
    trainablegates_location = parametrizedgate_location[findall(x->x==1, mask .== 1)]
    X = (isempty(trainablegates) ? [] : trainablegates_location .=> keys.(last.(trainablegates))) 
    push!(parameters, X)
  end
  VariationalQuantumEigensolver(H,circuit,parameters)
end

Base.copy(vqe::VariationalQuantumEigensolver) = VQE(copy(vqe.Hamiltonian), copy.(vqe.circuit))


function loss(vqe::VariationalQuantumEigensolver)
  ψθ = runcircuit(qubits(vqe.Hamiltonian), vqe.circuit)
  loss = inner(ψθ, vqe.Hamiltonian, ψθ)
  @assert imag(loss) < 1e-7
  return real(loss)
end

"""
Compute the left and right environments with respect to the bra
"""

function environments(vqe::VariationalQuantumEigensolver)
  # number of qubits
  N = length(vqe.Hamiltonian)
  # depth of the vqe circuit
  depth = length(vqe.circuit)

  # left environment
  ΨL = MPS[] 
  ψ = qubits(hilbertspace(vqe.Hamiltonian))
  push!(ΨL, ψ)
  for d in 1:depth-1
    layer = vqe.circuit[d]
    ψ = runcircuit(ψ, layer; move_sites_back = true)  
    push!(ΨL, ψ)
  end
  ψ = runcircuit(ψ, vqe.circuit[end]; move_sites_back = true)
  
  # right environment
  ΨR = MPS[]
  ψ = noprime(vqe.Hamiltonian * ψ)
  push!(ΨR,ψ) 
  for d in reverse(2:depth)
    layer = vqe.circuit[d]
    ψ = runcircuit(ψ,dag(layer); move_sites_back = true)
    push!(ΨR,ψ)
  end
  ΨR = reverse(ΨR)
  return ΨL,ΨR 
end

function gradients(vqe::VariationalQuantumEigensolver)
  # number of qubits
  N = length(vqe.Hamiltonian)
  # depth of the vqe circuit
  depth = length(vqe.circuit)
  # environments
  ΨL, ΨR = PastaQ.environments(vqe)
  
  # gradients
  gradients = []
  
  # loop over the layers
  for d in 1:depth
    #gradlayer = []
    
    for g in vqe.parameters[d]
      gateposition = first(g)
      gatename, support, trainable_params = vqe.circuit[d][gateposition]
      # remove specific gate from the layer
      partiallayer = deleteat!(copy(vqe.circuit[d]), gateposition)
      # apply the partial layer to the left environment
      ξL = runcircuit(ΨL[d], partiallayer; move_sites_back = true)
      grad = array(inner(ξL, ΨR[d], support))
      
      for par in keys(trainable_params)
        gradgate = gradient(gatename;trainable_params...)[par]
        ∇ = 2*real(tr(grad * gradgate'))
        push!(gradients, ∇)
        #push!(gradlayer,∇)
      end
    end
    #push!(gradients, gradlayer)
  end
  return gradients
end


function getparameters(vqe::VariationalQuantumEigensolver)
  params = []
  for d in 1:length(vqe.circuit)
    parlayer = []
    for g in vqe.parameters[d]
      gateposition = first(g)
      gatename, support, trainable_params = vqe.circuit[d][gateposition]
      push!(params,values(trainable_params)...)
    end
  end
  return params
end



"""
    updateangle!(gate::Tuple,eps::Float64)
Update single-qubit rotation gate angle.
"""
function updateangles!(vqe::VariationalQuantumEigensolver, newparameters::Vector)
  cnt = 1 
  for d in 1:length(vqe.circuit)
    for g in vqe.parameters[d]
      gateposition = first(g)
      gatename, support, trainable_params = vqe.circuit[d][gateposition]
      for par in keys(trainable_params)
        vqe.circuit[d][gateposition] = Base.setindex(vqe.circuit[d][gateposition],
                                                     Base.setindex(vqe.circuit[d][gateposition][3], 
                                                                   newparameters[cnt],par),3) 
        cnt += 1
      end
    end
  end
end

function ITensors.inner(ψL::MPS, ψR::MPS, site::Int)
  N = length(ψL)
  ψL = dag(ψL)
  @assert length(ψR) == N
  if site == 1
    overlap = ψR[N] * prime(ψL[N],"Link")
    for j in reverse(2:N-1)
      overlap = overlap * ψR[j]
      overlap = overlap * prime(ψL[j],"Link")
    end
    overlap = overlap * ψR[1]
    overlap = overlap * prime(ψL[1])
  elseif site == N
    overlap = ψR[1] * prime(ψL[1],"Link")
    for j in 2:N-1
      overlap = overlap * ψR[j]
      overlap = overlap * prime(ψL[j],"Link")
    end
    overlap = overlap * ψR[N]
    overlap = overlap * prime(ψL[N])
  else
    overlap = ψR[1] * prime(ψL[1],"Link")
    for j in 2:site-1
      overlap = overlap * ψR[j]
      overlap = overlap * prime(ψL[j],"Link")
    end
    overlap = overlap * ψR[site]
    overlap = overlap * prime(ψL[site])
    for j in site+1:N
      overlap = overlap * ψR[j]
      overlap = overlap * prime(ψL[j],"Link")
    end
  end
  return overlap
end

function ITensors.inner(ψL::MPS, ψR::MPS, sites::Tuple)
  error("VQE for 1qubit variational gates not yet implemented")
end



#
#"""
#    updateangle!(gate::Tuple,eps::Float64)
#Update single-qubit rotation gate angle.
#"""
#function updateangles!(vqe::VariationalQuantumEigensolver, gradients::Array; η::Float64 = 0.01)
#  
#  
#  #depth = length(vqe.circuit)
#  #for d in 1:depth
#  #  counter = 1
#  #  for i in 1:length(vqe.circuit[d])
#  #    if PastaQ.istrainable(vqe.circuit[d][i])
#  #      par_ids =  keys(vqe.circuit[d][i][3])
#  #      for par_id in par_ids
#  #        old_angle = vqe.circuit[d][i][3][par_id]
#  #        new_angle = old_angle - η * gradients[d][counter]
#  #        vqe.circuit[d][i] = Base.setindex(vqe.circuit[d][i],Base.setindex(vqe.circuit[d][i][3], new_angle,par_id),3) 
#  #        counter += 1
#  #      end
#  #    end
#  #  end
#  #end
#end
#
#
