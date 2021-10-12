function _drivinghamiltonian(H₀::OpSum, drives::Vector{<:Pair}, t::Float64; kwargs...)
  H = copy(H₀)
  for drive in drives
    drive_function = first(drive)
    if drive_function isa Function
      ft = drive_function(t)
    else
      @assert length(drive_function) > 1
      f, θ... = drive_function
      ft = f(t,θ...)
    end
    driveop, support = last(drive)
    H += ft, driveop, support
  end
  return ITensors.sortmergeterms!(H)
end



function _coherentcontrolcircuit(H₀::OpSum, drives::Vector{<:Pair}, ts::Vector; kwargs...)
  
#  # first we back out the undriven hamiltonian H0, as well as any constant offset in the 
#  # driven terms
#  H₀ = OpSum()
#  offset = zeros(Float64, length(drives))
#  
#  # loop over Hamiltonian terms
#  for k in 1:length(H)
#    hk = H[k]
#    coupling = ITensors.coef(hk)
#    O = ITensors.ops(hk)
#    length(O) > 1 && error("only a single operator allowed per term")
#    localop = ITensors.name(O[1])
#    support = ITensors.sites(O[1])
#    params = ITensors.params(O[1])
#    support = length(support) == 1 ? support[1] : support
#    
#    # check if this term is being driven
#    drive_index = findfirst(x -> x == (localop,support), last.(drives))
#    if !isnothing(drive_index)
#      # save a constant offset (if any)
#      offset[drive_index] = coupling 
#    else
#      push!(H₀, hk) 
#    end
#  end
#
#  Hsequence = OpSum[]
#  # time sequence
#  for t in 0.0:δt:T
#    # time-independent term
#    Ht = copy(H₀)
#    # driving operators
#    for drive_index in 1:length(drives)
#      drive_function = first(drives[drive_index])
#      if drive_function isa Function
#        ft = drive_function(t)
#      else
#        @assert length(drive_function) == 2
#        f, θ = drive_function
#        ft = f(t,θ)
#      end
#      driveop, support = last(drives[drive_index])
#      Ht += offset[drive_index] + ft, driveop, support
#    end
#    push!(Hsequence, Ht)
#  end
#  return trottercircuit(Hsequence; δt = δt, kwargs...)
end
#
#trottercircuit(H::OpSum, drive::Pair, args...; kwargs...) =
#  trottercircuit(H, [drive], args...; kwargs...)

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
