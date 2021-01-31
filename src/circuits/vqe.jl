struct VariationalQuantumEigensolver
  Hamiltonian::MPO
  circuit::Vector{Vector{<:Tuple}}
  #parameters::Vector{Vector{Pair{Union{Int,Tuple},Union{Symbol,Tuple}}}}
end
VQE(H::MPO, circuit::Vector{<:Vector{<:Tuple}}) = VariationalQuantumEigensolver(H,circuit)
#VQE(H::MPO, circuit::Vector{<:Tuple}, args...) = VQE(H,[circuit], args...)

Base.copy(vqe::VariationalQuantumEigensolver) = VQE(copy(vqe.Hamiltonian), copy.(vqe.circuit))

function configure(vqe::VariationalQuantumEigensolver)
  

end

function energy(vqe::VariationalQuantumEigensolver)
  ψθ = runcircuit(qubits(vqe.Hamiltonian), vqe.circuit)
  E = inner(ψθ, vqe.Hamiltonian, ψθ)
  @assert(imag(E)<1e-7)
  return real(E)
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
    ψ = runcircuit(ψ, layer)  
    push!(ΨL, ψ)
  end
  ψ = runcircuit(ψ, vqe.circuit[end])
  
  # right environment
  ΨR = MPS[]
  ψ = noprime(vqe.Hamiltonian * ψ)
  push!(ΨR,ψ) 
  for d in reverse(2:depth)
    layer = vqe.circuit[d]
    ψ = runcircuit(ψ,dag(layer))
    push!(ΨR,ψ)
  end
  ΨR = reverse(ΨR)
  return ΨL,ΨR 
end


function gradients_tensors(vqe::VariationalQuantumEigensolver)
  # number of qubits
  N = length(vqe.Hamiltonian)
  # depth of the vqe circuit
  depth = length(vqe.circuit)
  # environments
  ΨL, ΨR = PastaQ.environments(vqe)
  
  # gradients
  gradients = []
  trainable_gates = []
  
  # loop over the layers
  for d in 1:depth
    gradlayer = ITensor[]
    
    layer = vqe.circuit[d]
    ngates = length(layer)
    
    trainable_index = findall(x -> x==1, PastaQ.istrainable.(layer))
    push!(trainable_gates, layer[trainable_index]) 
    if !isempty(trainable_index)
      for i in 1:length(trainable_index)
        # remove specific gate from the layer
        partiallayer = deleteat!(copy(layer),trainable_index[i])
        
        # apply the partial layer to the left environment
        ξL = runcircuit(ΨL[d], partiallayer)
        grad = inner(ξL, ΨR[d], layer[trainable_index[i]][2])
        push!(gradlayer,grad)
      end
    end
    push!(gradients, gradlayer)
  end
  return 2 * gradients
end


function gradients_parameters(vqe)
  # number of qubits
  N = length(vqe.Hamiltonian)
  # depth of the vqe circuit
  depth = length(vqe.circuit)
  grad_tensors = gradients_tensors(vqe)
  
  gradients = []
  trainable_gates = []
  for d in 1:depth
    gradlayer = [] 
    tensor_counter = 1
    layer = vqe.circuit[d]
    trainable_index = findall(x -> x==1, PastaQ.istrainable.(layer))
    push!(trainable_gates, layer[trainable_index]) 
    if !isempty(trainable_index)
      for i in 1:length(trainable_index)
        #@show layer[trainable_index[i]]
        gateid = layer[trainable_index[i]][1]
        par_id = keys(layer[trainable_index[i]][3])
        for par in par_id
          gradgate = gradient(gateid; layer[trainable_index[i]][3]...)[par]
          X = gradgate'
          ∇ = tr(array(grad_tensors[d][tensor_counter]) * X)
          push!(gradlayer,real(∇))
        end
        tensor_counter += 1
      end
    end
    push!(gradients,gradlayer)
  end
  return gradients
end
"""
    updateangle!(gate::Tuple,eps::Float64)
Update single-qubit rotation gate angle.
"""
function updateangles!(vqe::VariationalQuantumEigensolver, gradients::Array; η::Float64 = 0.01)
  depth = length(vqe.circuit)
  for d in 1:depth
    counter = 1
    for i in 1:length(vqe.circuit[d])
      if PastaQ.istrainable(vqe.circuit[d][i])
        par_ids =  keys(vqe.circuit[d][i][3])
        for par_id in par_ids
          old_angle = vqe.circuit[d][i][3][par_id]
          new_angle = old_angle - η * gradients[d][counter]
          vqe.circuit[d][i] = Base.setindex(vqe.circuit[d][i],Base.setindex(vqe.circuit[d][i][3], new_angle,par_id),3) 
          counter += 1
        end
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

istrainable(g::Tuple) = length(g) == 3 

#VQE(H::MPO) = VariationalQuantumEigensolver(H,Vector{Vector{<:Tuple}}(undef, 0),Vector{Vector{Pair{Union{Int,Tuple},Union{Symbol,Tuple}}}}(undef,0))
#
#function VQE(H::MPO, circuit::Vector{<:Vector{<:Tuple}}) 
#  N = length(H)
#  depth = length(circuit)
#  @assert numberofqubits(circuit) == N 
#  parameters = Vector{Vector{Pair{<:Union{Int64,Tuple},<:Union{Symbol,Tuple}}}}(undef,0)
#  for d in 1:depth
#    layer = circuit[d]
#    layerpars = Vector{Pair{<:Union{Int64,Tuple},<:Union{Symbol,Tuple}}}(undef,0)
#    for (i,g) in enumerate(layer)
#      if length(g) == 3
#        pars = keys(g[3])
#        tmp = g[2] => pars
#        push!(layerpars, g[2] => pars)
#      end
#    end
#    push!(parameters,layerpars)
#  end
#  return VariationalQuantumEigensolver(H, circuit, parameters)
#end
#
#VQE(H::MPO, circuit::Vector{<:Tuple}, args...) = VQE(H,[circuit], args...)

