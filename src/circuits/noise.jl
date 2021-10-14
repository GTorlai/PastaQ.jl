
function gate(::GateName"pauli_channel", dims::Tuple = (2,); 
              pauli_ops = ["Id","X","Y","Z"],
              error_probabilities = prepend!(zeros(length(pauli_ops)^length(dims)-1),1))
  @assert sum(error_probabilities) ≈ 1
  @assert all(dims .== 2)
  N = length(dims)
  length(error_probabilities) > (1 << 10) && error("Hilbert space too large")
  error_probabilities ./= sum(error_probabilities)
  kraus = zeros(Complex{Float64},1<<N,1<<N,length(pauli_ops)^N)
  basis = vec(reverse.(Iterators.product(fill(pauli_ops,N)...)|>collect))
  for (k,ops) in enumerate(basis)
    kraus[:,:,k] = √error_probabilities[k] * reduce(kron,gate.(ops)) 
  end
  return kraus
end

gate(::GateName"bit_flip", dims::Tuple = (2,); p::Number) = 
  gate("pauli_channel", dims; error_probabilities = prepend!(p/(2^length(dims)-1) * ones(2^length(dims)-1), 1-p), pauli_ops = ["Id","X"])

gate(::GateName"phase_flip", dims::Tuple = (2,); p::Number) = 
  gate("pauli_channel", dims; error_probabilities = prepend!(p/(2^length(dims)-1) * ones(2^length(dims)-1), 1-p), pauli_ops = ["Id","Z"])

function gate(::GateName"AD", dims::Tuple = (2,); γ::Real)
  N = length(dims)
  @assert all(dims .== 2)
  kraus = zeros(2,2,2)
  kraus[:,:,1] = [1 0
                  0 sqrt(1-γ)]
  kraus[:,:,2] = [0 sqrt(γ)
                  0 0]
  N == 1 && return kraus 
  k1 = kraus[:,:,1]
  k2 = kraus[:,:,2]
  T = vec(Iterators.product(fill([k1,k2],N)...)|>collect)
  K = zeros(Float64,1<<N,1<<N,1<<N)
  for x in 1:1<<N
    K[:,:,x] = reduce(kron,T[x]) 
  end
  return K
end
#
# To accept the gate name "amplitude_damping"
gate(::GateName"amplitude_damping", dims::Tuple = (2,); kwargs...) = gate("AD", dims; kwargs...)

function gate(::GateName"PD", dims::Tuple = (2,); γ::Real)
  N = length(dims)
  @assert all(dims .== 2)
  kraus = zeros(2,2,2)
  kraus[:,:,1] = [1 0
                  0 sqrt(1-γ)]
  kraus[:,:,2] = [0 0
                  0 sqrt(γ)]

  N == 1 && return kraus 
  k1 = kraus[:,:,1]
  k2 = kraus[:,:,2]
  T = vec(Iterators.product(fill([k1,k2],N)...)|>collect)
  K = zeros(Float64,1<<N,1<<N,1<<N)
  for x in 1:1<<N
    K[:,:,x] = reduce(kron,T[x]) 
  end
  return K
end

# To accept the gate name "phase_damping"
gate(::GateName"phase_damping", dims::Tuple = (2,); kwargs...) = gate("PD", dims; kwargs...)
#
# To accept the gate name "dephasing"
gate(::GateName"dephasing", dims::Tuple = (2,); kwargs...) = gate("PD", dims; kwargs...)

# make general n-qubit
gate(::GateName"DEP", dims::Tuple = (2,); p::Number) =
  gate("pauli_channel", dims; error_probabilities = prepend!(p/(4^length(dims)-1) * ones(4^length(dims)-1), 1-p), pauli_ops = ["Id","X","Y","Z"])

# To accept the gate name "depolarizing"
gate(::GateName"depolarizing", dims::Tuple = (2,); kwargs...) = 
  gate("DEP", dims; kwargs...)

function insertnoise(circuit::Vector{<:Vector{<:Any}}, noisemodel::Tuple; gate = nothing)#idlenoise::Bool = false) 
  max_g_size = maxgatesize(circuit) 

  # single noise model for all
  if noisemodel[1] isa String
    tmp = []
    for k in 1:max_g_size
      push!(tmp, k => noisemodel)
    end
    #if idlenoise
    #  push!(tmp, "idle" => noisemodel)
    #end
    noisemodel = Tuple(tmp)
  end

  noisycircuit = []
  for layer in circuit
    noisylayer = []
    for g in layer
      push!(noisylayer, g)
      applynoise = (isnothing(gate) ? true : 
                    gate isa String ? g[1] == gate : g[1] in gate)
      #@applytogate = gate isa String ? g[1] == gate : g[1] in gate
      #if isnothing(gate) || applytogate
      if applynoise
        nq = g[2]
        # n-qubit gate
        if nq isa Tuple
          gatenoiseindex = findfirst(x -> x == length(nq), first.(noisemodel))
          isnothing(gatenoiseindex) && error("Noise model not defined for $(length(nq))-qubit gates!")
          gatenoise = last(noisemodel[gatenoiseindex])
          # Check whether the n-qubit Kraus channel has been defined
          # n -qubit Kraus operator
          noisecheck = PastaQ.gate(gatenoise[1], Tuple(repeat([2], length(nq))); gatenoise[2]...)
          # if the single-qubit copy is return, use productnoise, and throw a warning
          if size(noisecheck,1) < 1<<length(nq)
            @warn "$(length(nq))-qubit Kraus operators for the $(gatenoise[1]) noise not defined. Applying tensor-product noise instead.\n"
            # tensor-product noise
            for q in nq
              push!(noisylayer,(gatenoise[1], q, gatenoise[2]))
            end
          else
            # correlated n-qubit noise
            push!(noisylayer,(gatenoise[1], nq, gatenoise[2]))
          end
        # 1-qubit gate
        else
          gatenoiseindex = findfirst(x -> x == 1, first.(noisemodel))
          isnothing(gatenoiseindex) && error("Noise model not defined for 1-qubit gates!")
          gatenoise = last(noisemodel[gatenoiseindex])
          push!(noisylayer, (gatenoise[1], nq, gatenoise[2]))  
        end
      end
    end
    #if idlenoise
    #  gatenoiseindex = findfirst(x -> x == "idle", first.(noisemodel))
    #  isnothing(gatenoiseindex) && error("Noise model not defined for idling qubits!")
    #  gatenoise = last(noisemodel[gatenoiseindex])
    #  busy_qubits = vcat([collect(g[2]) for g in layer]...)
    #  idle_qubits = filter(y -> y ∉ busy_qubits, 1:nqubits(circuit))
    #  for n in idle_qubits
    #    push!(noisylayer, (gatenoise[1], n, gatenoise[2]))
    #  end
    #end
    push!(noisycircuit, noisylayer)
  end
  return noisycircuit
end

insertnoise(circuit::Vector{<:Any}, noisemodel::Tuple; kwargs...) = 
  insertnoise([circuit], noisemodel; kwargs...)[1]

function maxgatesize(circuit::Vector{<:Vector{<:Any}})
  maxsize = 0
  for layer in circuit
    for g in layer
      maxsize = length(g[2]) > maxsize ? length(g[2]) : maxsize
    end
  end
  return maxsize
end

maxgatesize(circuit::Vector{<:Any}) = 
  maxgatesize([circuit])
 
