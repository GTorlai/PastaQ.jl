function gate(::GateName"pauli_channel", ::SiteType"Qubit", s::Index...; 
              pauli_ops = ["Id","X","Y","Z"],
              error_probabilities = prepend!(zeros(length(pauli_ops)^length(dim.(s))-1),1))
  dims = dim.(s)
  @assert sum(error_probabilities) ≈ 1
  @assert all(dims .== 2)
  N = length(dims)
  length(error_probabilities) > (1 << 10) && error("Hilbert space too large")
  error_probabilities ./= sum(error_probabilities)
  kraus = zeros(Complex{Float64},1<<N,1<<N,length(pauli_ops)^N)
  krausind = Index(size(kraus, 3); tags="kraus")
  
  basis = vec(reverse.(Iterators.product(fill(pauli_ops,N)...)|>collect))
  for (k,ops) in enumerate(basis)
    kraus[:,:,k] = √error_probabilities[k] * reduce(kron, [gate(op, SiteType("Qubit")) for op in ops]) 
  end
  return ITensors.itensor(kraus, prime.(s)..., ITensors.dag.(s)..., krausind)
end

# XXX
# why not passing the t here again
gate(::GateName"bit_flip", t::SiteType"Qubit", s::Index...; p::Number = 0.0) = 
  gate("pauli_channel", s...; error_probabilities = prepend!(p/(2^length(dim.(s))-1) * ones(2^length(dim.(s))-1), 1-p), pauli_ops = ["Id","X"])

gate(::GateName"phase_flip", t::SiteType"Qubit", s::Index...; p::Number = 0.0) = 
  gate("pauli_channel", s...; error_probabilities = prepend!(p/(2^length(dim.(s))-1) * ones(2^length(dim.(s))-1), 1-p), pauli_ops = ["Id","Z"])
#XXX


function gate(::GateName"AD", ::SiteType"Qubit", s::Index...; γ::Real = 0.0)
  dims = dim.(s)
  N = length(dims)
  @assert all(dims .== 2)
  
  kraus = zeros(2,2,2)
  kraus[:,:,1] = [1 0
                  0 sqrt(1-γ)]
  kraus[:,:,2] = [0 sqrt(γ)
                  0 0]
  krausind = Index(size(kraus, 3); tags="kraus")
  N == 1 && return ITensors.itensor(kraus, prime.(s)..., ITensors.dag.(s)..., krausind)
  
  k1 = kraus[:,:,1]
  k2 = kraus[:,:,2]
  T = vec(Iterators.product(fill([k1,k2],N)...)|>collect)
  K = zeros(Float64,1<<N,1<<N,1<<N)
  for x in 1:1<<N
    K[:,:,x] = reduce(kron,T[x]) 
  end
  krausind = Index(size(K, 3); tags="kraus")
  return ITensors.itensor(K, prime.(s)..., ITensors.dag.(s)..., krausind)
end

# To accept the gate name "amplitude_damping"
gate(::GateName"amplitude_damping", t::SiteType"Qubit", s::Index...; kwargs...) = gate("AD", s...; kwargs...)


function gate(::GateName"PD", ::SiteType"Qubit", s::Index...; γ::Real)
  dims = dim.(s)
  N = length(dims)
  @assert all(dims .== 2)
  kraus = zeros(2,2,2)
  kraus[:,:,1] = [1 0
                  0 sqrt(1-γ)]
  kraus[:,:,2] = [0 0
                  0 sqrt(γ)]

  krausind = Index(size(kraus, 3); tags="kraus")
  N == 1 && return ITensors.itensor(kraus, prime.(s)..., ITensors.dag.(s)..., krausind)
  
  k1 = kraus[:,:,1]
  k2 = kraus[:,:,2]
  T = vec(Iterators.product(fill([k1,k2],N)...)|>collect)
  K = zeros(Float64,1<<N,1<<N,1<<N)
  for x in 1:1<<N
    K[:,:,x] = reduce(kron,T[x]) 
  end
  krausind = Index(size(K, 3); tags="kraus")
  return ITensors.itensor(K, prime.(s)..., ITensors.dag.(s)..., krausind)
end

# To accept the gate name "phase_damping"
gate(::GateName"phase_damping", ::SiteType"Qubit", s::Index...; kwargs...) = gate("PD", s...; kwargs...)
# To accept the gate name "dephasing"
gate(::GateName"dephasing", ::SiteType"Qubit", s::Index...; kwargs...) = gate("PD", s...; kwargs...)

# make general n-qubit
gate(::GateName"DEP", ::SiteType"Qubit", s::Index...; p::Number) =
  gate("pauli_channel", s...; error_probabilities = prepend!(p/(4^length(dim.(s))-1) * ones(4^length(dim.(s))-1), 1-p), pauli_ops = ["Id","X","Y","Z"])

# To accept the gate name "depolarizing"
gate(::GateName"depolarizing", ::SiteType"Qubit", s::Index...; kwargs...) = 
  gate("DEP", s...; kwargs...)



function insertnoise(circuit::Vector{<:Vector{<:Any}}, noisemodel::Tuple; gate = nothing)
  max_g_size = maxgatesize(circuit) 
  numqubits = nqubits(circuit)
  
  check_inds = siteinds("Qubit", numqubits)
  # single noise model for all
  if noisemodel[1] isa String
    tmp = []
    for k in 1:max_g_size
      push!(tmp, k => noisemodel)
    end
    noisemodel = Tuple(tmp)
  end

  noisycircuit = []
  for layer in circuit
    noisylayer = []
    for g in layer
      push!(noisylayer, g)
      applynoise = (isnothing(gate) ? true : 
                    gate isa String ? g[1] == gate : g[1] in gate)
      if applynoise
        nq = g[2]
        # n-qubit gate
        if nq isa Tuple
          gatenoiseindex = findfirst(x -> x == length(nq), first.(noisemodel))
          isnothing(gatenoiseindex) && error("Noise model not defined for $(length(nq))-qubit gates!")
          gatenoise = last(noisemodel[gatenoiseindex])
          push!(noisylayer,(gatenoise[1], nq, gatenoise[2]))
        # 1-qubit gate
        else
          gatenoiseindex = findfirst(x -> x == 1, first.(noisemodel))
          isnothing(gatenoiseindex) && error("Noise model not defined for 1-qubit gates!")
          gatenoise = last(noisemodel[gatenoiseindex])
          push!(noisylayer, (gatenoise[1], nq, gatenoise[2]))  
        end
      end
    end
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
 
