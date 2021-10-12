""" 
SIMPLE GRADIENTS
to be susbstitute by Flux/ITensors AD
"""

# Rotation around X-axis
gradient(::GateName"Rx"; θ::Number, kwargs...) =
  (θ = 0.5*[-sin(θ/2) -im*cos(θ/2); -im*cos(θ/2) -sin(θ/2)],)

# Rotation around Y-axis
gradient(::GateName"Ry"; θ::Number, kwargs...) =
  (θ = 0.5*[-sin(θ/2) -cos(θ/2); cos(θ/2) -sin(θ/2)],)

# Rotation around Z-axis
gradient(::GateName"Rz"; ϕ::Number, kwargs...) =
  (ϕ = [0 0; 0 im*exp(im*ϕ)],)

# Rotation around generic axis n̂
gradient(::GateName"Rn"; θ::Real, ϕ::Real, λ::Real, kwargs...) =
  ( θ = 0.5*[-sin(θ/2)  -exp(im*λ)*cos(θ/2); exp(im*ϕ)*cos(θ/2) -exp(im*(ϕ+λ))*sin(θ/2)],
    ϕ = [0 0; im*exp(im*ϕ)*sin(θ/2)  im*exp(im*(ϕ+λ))*cos(θ/2)],
    λ = [0 -im*exp(im*λ)*sin(θ/2); 0 im*exp(im*(ϕ+λ))*cos(θ/2)]
  )
gradient(s::String; kwargs...) = gradient(GateName(s); kwargs...)


"""
    variationalgate(g::Tuple)
Given an input gate, this functions makes the gate variational,
by appending a `∇=true` in its parameters.
"""
function variationalgate(g::Tuple)
  length(g) == 2 && error("Cannot train a non-parametric gate")
  gatename, support, params = g
  newparams = merge(params, (∇ = true,))
  return (gatename, support, newparams)
end


#"""
#    variationalcircuit(circuit::Vector{<:Vector{<:Any}}; variationalgates = nothing)
#    variationalcircuit(circuit::Vector{<:Any}; kwargs...)
#Given an input parametric circuit, promote a set of gates (`variationalgates`) to 
#trainable gates, by appending the `∇=true` parameter
#"""
#function variationalcircuit(circuit::Vector{<:Vector{<:Any}}; variationalgates = nothing)
#  varcircuit = Vector{Tuple}[]
#  if !isnothing(variationalgates) && !(variationalgates isa Vector)
#    variationalgates = [variationalgates]
#  end
#  novariationalgatesfound = true
#  # loop over circuit layers
#  for layer in circuit
#    varlayer = Tuple[]
#    # loop over gates
#    for g in layer
#      # if g is non-parametric, simply add that to the layer
#      if length(g) != 3
#        push!(varlayer, g)
#      # 
#      # if g is parametric
#      else
#        # all parametric gates are variational
#        if isnothing(variationalgates)
#          novariationalgatesfound = false
#          push!(varlayer, variationalgate(g))
#        else
#          gatename, support, _ = g
#          # loop over specific gate to train
#          foundit_flag = false
#          for vargate in variationalgates
#            # if gate is applied to any site
#            if (vargate isa String) && (vargate == gatename) 
#              foundit_flag = true
#              push!(varlayer, variationalgate(g))
#              novariationalgatesfound = false
#            elseif (vargate isa Tuple) && (vargate == (gatename,support)) 
#              novariationalgatesfound = false
#              foundit_flag = true
#              push!(varlayer, variationalgate(g))
#            end
#          end
#          if !foundit_flag
#            push!(varlayer, g)
#          end
#        end
#      end
#    end
#    push!(varcircuit, varlayer)
#  end
#  if novariationalgatesfound
#    @warn "No variational gate in the set $variationalgates was found in the circuit!"
#  end
#  return varcircuit
#end
#
#
#variationalcircuit(circuit::Vector{<:Any}; kwargs...) = 
#  vcat(variationalcircuit([circuit]; kwargs...)...)
#
#variationalcircuit(N::Int, args...; 
#                   variationalgates = nothing,
#                   twoqubitgates = "CX",
#                   onequbitgates = "Rn",
#                   kwargs...) = 
#  variationalcircuit(randomcircuit(N, args...; 
#                                   twoqubitgates = twoqubitgates,
#                                   onequbitgates = onequbitgates,
#                                   kwargs...); variationalgates = variationalgates)

"""
    circuitmap(circuit::Vector{<:Vector{<:Any}}) 
Generates an encoding of the locations of trainable gates within a 
variational quantum circuit
"""
function circuitmap(circuit::Vector{<:Vector{<:Any}}) 
  cmap = []
  # loop over the layers
  for layer in circuit
    # find where there are parametrized gates
    lmap = []
    for (i,g) in enumerate(layer)
      if (length(g) == 3) && (haskey(g[3],:∇)) && g[3][:∇]
        push!(lmap,i)
      end
    end
    push!(cmap, lmap)
  end
  return cmap
end

circuitmap(circuit::Vector{<:Any}) = 
  vcat(circuitmap([circuit])...)


#"""
#    trainableparameters(circuit::Vector{<:Any})
#Return a list of trainable parameters (e.g. θ, ϕ etc) for each
#gate in a trainable position. This faciliates parameters handling
#during the optimization.
#"""
#function trainableparameters(circuit::Vector{<:Any})
#  trainpars = []
#  cmap = circuitmap(circuit)
#  for gloc in cmap
#    gatename, support, params = circuit[gloc]
#    push!(trainpars, keys(gradient(gatename; params...)))
#  end
#  return trainpars
#end
#
#
#
#"""
#    loss(circuit::Vector{<:Vector{<:Any}}, costfunction::MPO)
#Estimate the MPO objective function on the parametric wavefunction
#obtained from a variational circuit.
#"""
#function loss(circuit::AbstractVector, objectivefunction::MPO; kwargs...)
#  ψθ = runcircuit(productstate(objectivefunction), circuit; kwargs...)
#  L = inner(ψθ, objectivefunction, ψθ)
#  @assert imag(L) < 1e-7
#  return real(L)
#end
#
#
#"""
#    gradients(circuit::Vector{<:Any},  ...)
#Compute the gradients with respect to each trainable parameters of the
#expectation value of the objective function MPO on the wavefunction 
#generate by the parametric circuit:
#C = ⟨ψ(θ)|H|ψ(θ)⟩ = ⟨0|U† H U|0⟩
#"""
#function gradients(circuit::Vector{<:Any}, 
#                   objectivefunction::MPO,
#                   cmap::Vector;
#                   kwargs...)
#
#  qubits = firstsiteinds(objectivefunction)
#  dagcircuit = dag(circuit)
#  ∇ = []  
#  ψθ = runcircuit(qubits, circuit; kwargs...)
#  ψL = copy(ψθ)
#  ψR = noprime(objectivefunction * ψθ)
#  loss = inner(ψL, ψR)
#  
#  gcnt = 1
#  for gloc in cmap
#    ψL = runcircuit(ψL, dagcircuit[gcnt:gloc])
#    if gloc != 1
#      ψR = runcircuit(ψR, dagcircuit[gcnt:gloc-1])
#    end
#    gcnt = gloc +1
#    
#    gatename, support, params = dagcircuit[gloc]
#    
#    grad = gradient(gatename; params...)
#    pars = keys(grad)
#    
#    s = qubits[[support...]]
#    ∇gate = []
#    for par in pars
#      ∇tensor = itensor(Array(grad[par]'), prime.(s...), ITensors.dag(s)...)
#      ϕ = runcircuit(ψR, [∇tensor]; kwargs...)
#      push!(∇gate, par => 2*real(inner(ψL, ϕ)))
#    end
#    push!(∇, ∇gate)
#    ψR = runcircuit(ψR, dagcircuit[gloc]; kwargs...)
#  end
#  return real(loss), ψθ, reverse(∇)
#end
#
#gradients(circuit::AbstractVector, objectivefunction::MPO; kwargs...) = 
#  gradients(circuit, objectivefunction, circuitmap(dag(circuit)))
#
#function _getparameters(circuit::Vector{<:Any}, cmap::Vector, trainpars::Vector)
#  params = Float64[]
#  for (k,gloc) in enumerate(cmap)
#    gatename, support, trainable_params = circuit[gloc]
#    for p in trainpars[k]
#      push!(params, trainable_params[p])
#    end
#  end
#  return params
#end
#
#function _setparameters!(circuit::Vector{<:Any}, newparameters::Vector, cmap::Vector, trainpars::Vector)
#  cnt = 1 
#  for (k,gloc) in enumerate(cmap)
#    gatename, support, params = circuit[gloc]
#    for p in trainpars[k]
#      circuit[gloc] = Base.setindex(circuit[gloc], 
#                                    Base.setindex(circuit[gloc][3],
#                                                  newparameters[cnt],p), 3)
#      cnt += 1
#    end
#  end
#end
#
#function _parsegradient(∇::Vector, circuit::Vector{<:Any}, cmap::Vector, trainpars::Vector)
#  ∇dense = Float64[]
#  for (k,gloc) in enumerate(cmap)
#    gatename, support, trainable_params = circuit[gloc]
#    for (i,p) in enumerate(trainpars[k])
#      g = ∇[k]
#      gradkey = first(g[i])
#      push!(∇dense, last(∇[k][i])) 
#    end
#  end
#  return ∇dense
#end
#"""
#    minimize!(circuit0::Vector{<:Vector{<:Any}}, args...; kwargs...)
#Minimize a cost function using parametric circuit (e.g. VQE).
#"""
#function minimize!(circuit::Vector{<:Any},
#                   costfunction::MPO;
#                   optimizer = Optimisers.Descent(0.01),
#                   (observer!) = nothing, 
#                   epochs::Int = 1000,
#                   maxdim::Int64 = 10_000,
#                   cutoff::Float64 = 1e-12,
#                   earlystop::Bool = false,
#                   print_metrics = [], 
#                   outputlevel::Int = 1,
#                   observe_step::Int = 1)
#  
#  if !isnothing(observer!)
#    observer!["loss"] = nothing
#  end
#
#  # identify the location of trainable gates
#  cmap  = circuitmap(circuit)
#  cdagmap = circuitmap(dag(circuit))
#  # identify the variational parameters of each trainable gates
#  trainpars = trainableparameters(circuit)
#  
#  θ  = _getparameters(circuit, cmap, trainpars)
#  st = Optimisers.state(optimizer, θ)
#  
#  observe_time = 0.0
#  # training iterations
#  for ep in 1:epochs
#    # parse the trainable parameters
#    θ = _getparameters(circuit, cmap, trainpars)
#    # evaluate gradients
#    ep_time = @elapsed begin 
#      loss, ψθ, ∇ = gradients(circuit, costfunction, cdagmap; maxdim = maxdim, cutoff = cutoff)
#      # perform gradient update
#      ∇ = _parsegradient(∇, circuit, cmap, trainpars)
#      st, θ′ = Optimisers.update(optimizer, st, θ, ∇)
#      # parse the parameters back into the circuit data structure
#      _setparameters!(circuit, θ′, cmap, trainpars)
#    end
#    observe_time += ep_time
#    if ep % observe_step == 0
#      if !isnothing(observer!)
#        update!(observer!, ψθ; loss = loss)
#        push!(last(observer!["loss"]), loss)
#      end
#      if outputlevel > 0
#        @printf("%-4d  ", ep)
#        @printf("loss = %-4.4f  ", loss)
#        # TODO: add the trace preserving cost function here for QPT
#        !isnothing(observer!) && printobserver(observer!, print_metrics)
#        @printf("elapsed = %-4.3fs", observe_time)
#        println()
#        observe_time = 0.0
#      end
#    end
#  end
#  return circuit
#end
#
#function minimize!(circuit0::Vector{<:Vector{<:Any}}, costfunction::MPO; kwargs...)
#  circuit = copy(circuit0)
#  depth = length(circuit)
#  layersize = [length(layer) for layer in circuit]
#   
#  minimize!(vcat(circuit...), costfunction; kwargs...)
#
#  layeredcircuit = Vector{Tuple}[]
#  cnt = 1
#  for d in 1:depth
#    layer = Tuple[]
#    for j in 1:layersize[d]
#      push!(layer, circuit[cnt])
#      cnt += 1
#    end
#    push!(layeredcircuit, layer)
#  end
#  circuit0[:] = layeredcircuit
#  return circuit0 
#end
#
#maximize!(circuit::AbstractVector, costfunction::MPO, args...; kwargs...)  = 
#  minimize!(circuit, -costfunction, args...; kwargs...)
