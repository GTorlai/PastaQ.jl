"""
Evaluate a MPO cost function on a variational circuit
"""
function _loss(circuit::Vector{<:Vector{<:Any}}, costfunction::MPO)
  ψθ = runcircuit(qubits(costfunction), circuit)
  L = inner(ψθ, costfunction, ψθ)
  @assert imag(L) < 1e-7
  return real(L)
end

"""
Minimize a MPO cost function
"""
function minimize!(costfunction::MPO, 
                   circuit::Vector{<:Vector{<:Any}};
                   optimizer = Flux.Optimise.Descent(0.1),
                   epochs::Int = 1000,
                   maxdim::Int64 = 10_000,
                   cutoff::Float64 = 1e-12)
  
  # identity trainable parameters
  circuitmap = circuitmap(circuit)
  for ep in 1:epochs
    θ = _getparameters(circuit)
    loss, ∇ = gradients(circuit, costfunction, circuitmap; maxdim = maxdim, cutoff = cutoff)
    Flux.Optimise.update!(optimizer, θ, ∇)
    _setparameters!(circuit, θ)
    @printf("iter = %d  loss = %.5f",ep,real(loss))
    println()
  end
end

maximize!(costfunction::MPO, args...; kwargs...)  = 
  minimize!(-costfunction,args...; kwargs...)


function circuitmap(circuit::Vector{<:Vector{<:Any}}) 
  circuitmap = []
  # loop over the layers
  for layer in circuit# in 1:nlayers(circuit)
    #layer = circuit[d] 
    # find where there are parametrized gates
    parametrizedgate_location = findall(x -> x == 1, length.(layer) .== 3)    
    parametrizedgates = layer[parametrizedgate_location]
    # remove gates with the arg `nograd=true` 
    mask = .!BitArray(haskey.(last.(parametrizedgates),:nograd))
    trainablegates = parametrizedgates[findall(x->x==1, mask .== 1)]
    trainablegates_location = parametrizedgate_location[findall(x->x==1, mask .== 1)]
    layermap = (isempty(trainablegates) ? [] : trainablegates_location .=> keys.(last.(trainablegates))) 
    push!(circuitmap, layermap)
  end
  return circuitmap
end


"""
Compute the left and right environments with respect to the bra
"""
function environments(circuit::Vector{<:Vector{<:Any}}, costfunction::MPO; kwargs...)
  # number of qubits
  N = length(costfunction)
  # depth of the vqe circuit
  depth = length(circuit)

  # left environment
  ΨL = MPS[] 
  ψ = qubits(costfunction)
  push!(ΨL, ψ)
  for d in 1:depth-1
    layer = circuit[d]
    ψ = runcircuit(ψ, layer; move_sites_back = true, kwargs...)  
    push!(ΨL, ψ)
  end
  ψ = runcircuit(ψ, circuit[end]; move_sites_back = true, kwargs...)
  
  # right environment
  ΨR = MPS[]
  ψ = noprime(costfunction * ψ)
  push!(ΨR,ψ) 
  for d in reverse(2:depth)
    layer = circuit[d]
    ψ = runcircuit(ψ,dag(layer); move_sites_back = true, kwargs...)
    push!(ΨR,ψ)
  end
  ΨR = reverse(ΨR)
  return ΨL,ΨR 
end

gradients(circuit::Vector{<:Vector{<:Any}}, costfunction::MPO; kwargs...) = 
  gradients(circuit, costfunction, circuitmap(circuit); kwargs...) 

function gradients(circuit::Vector{<:Vector{<:Any}}, costfunction::MPO, circuitmap::Vector{<:Any}; kwargs...)
  # environments
  ΨL, ΨR = PastaQ.environments(circuit, costfunction; kwargs...)
  
  # gradients
  gradients = []
  
  # loop over the layers
  for d in 1:length(circuit)
    for g in circuitmap[d]
      gateposition = first(g)
      gatename, support, trainable_params = circuit[d][gateposition]
      # remove specific gate from the layer
      partiallayer = deleteat!(copy(circuit[d]), gateposition)
      #insert!(partiallayer,gateposition,
      #           (gatename,support,(trainable_params...,grad=true)))
      # apply the partial layer to the left environment
      ξL = runcircuit(ΨL[d], partiallayer; move_sites_back = true, kwargs...)
       grad = array(_circuitgradient(ξL, ΨR[d], support))
      
      for par in keys(trainable_params)
        gradgate = gradient(gatename;trainable_params...)[par]
        ∇ = 2*real(tr(grad * gradgate'))
        push!(gradients, ∇)
      end
    end
  end
  return inner(ΨL[2],ΨR[1]), gradients
end

_getparameters(circuit::Vector{<:Vector{<:Any}}) = 
  _getparameters(circuit,circuitmap(circuit)) 

function _getparameters(circuit::Vector{<:Vector{<:Any}}, circuitmap::Vector{<:Any})
  params = []
  for d in 1:length(circuit)
    parlayer = []
    for g in circuitmap[d]
      gateposition = first(g)
      gatename, support, trainable_params = circuit[d][gateposition]
      push!(params,values(trainable_params)...)
    end
  end
  return params
end


"""
    updateangle!(gate::Tuple,eps::Float64)
Update single-qubit rotation gate angle.
"""

_setparameters!(circuit::Vector{<:Vector{<:Any}}, newparameters::Array) = 
  _setparameters!(circuit,newparameters,circuitmap(circuit)) 

function _setparameters!(circuit::Vector{<:Vector{<:Any}}, newparameters::Array, circuitmap::Vector{<:Any})
  cnt = 1 
  for d in 1:length(circuit)
    for g in circuitmap[d]
      gateposition = first(g)
      gatename, support, trainable_params = circuit[d][gateposition]
      for par in keys(trainable_params)
        circuit[d][gateposition] = Base.setindex(circuit[d][gateposition],
                                                 Base.setindex(circuit[d][gateposition][3], 
                                                               newparameters[cnt],par),3) 
        cnt += 1
      end
    end
  end
end

function _circuitgradient(ψL::MPS, ψR::MPS, site::Int)
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

