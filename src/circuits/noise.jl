
function gate(::GateName"pauli_channel", N::Int = 1; 
              pauli_ops = ["Id","X","Y","Z"],
              error_probabilities = prepend!(zeros(length(pauli_ops)^N-1),1))
  length(error_probabilities) > (1 << 10) && error("Hilbert space too large")
  error_probabilities ./= sum(error_probabilities)
  kraus = zeros(Complex{Float64},1<<N,1<<N,length(pauli_ops)^N)
  basis = vec(reverse.(Iterators.product(fill(pauli_ops,N)...)|>collect))
  for (k,ops) in enumerate(basis)
    kraus[:,:,k] = √error_probabilities[k] * reduce(kron,gate.(ops)) 
  end
  return kraus
end

gate(::GateName"bit_flip", N::Int = 1; p::Number) = 
  gate("pauli_channel", N; error_probabilities = prepend!(p/(2^N-1) * ones(2^N-1), 1-p), pauli_ops = ["Id","X"])

gate(::GateName"phase_flip", N::Int = 1; p::Number) = 
  gate("pauli_channel", N; error_probabilities = prepend!(p/(2^N-1) * ones(2^N-1), 1-p), pauli_ops = ["Id","Z"])

function gate(::GateName"AD", N::Int = 1; γ::Real)
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
gate(::GateName"amplitude_damping", N::Int=1; kwargs...) = gate("AD", N; kwargs...)

function gate(::GateName"PD", N::Int = 1; γ::Real)
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
gate(::GateName"phase_damping",N::Int = 1; kwargs...) = gate("PD", N; kwargs...)
#
# To accept the gate name "dephasing"
gate(::GateName"dephasing", N::Int = 1; kwargs...) = gate("PD", N; kwargs...)

# make general n-qubit
gate(::GateName"DEP", N::Int = 1; p::Number) =
  gate("pauli_channel", N; error_probabilities = prepend!(p/(4^N-1) * ones(4^N-1), 1-p), pauli_ops = ["Id","X","Y","Z"])

# To accept the gate name "depolarizing"
gate(::GateName"depolarizing", N::Int = 1; kwargs...) = 
  gate("DEP", N; kwargs...)

insertnoise!(circuit::Vector, noise::Tuple; kwargs...) =  
  insertnoise!(circuit, (noise1Q = noise, noise2Q = noise); kwargs...)

function insertnoise!(circuit::Vector, noise::NamedTuple; idle_noise::Bool = false) 
  noise1Q = noise[:noise1Q]
  noise2Q = noise[:noise2Q]

  noisycircuit = []
  
  if circuit[1] isa Tuple
    circuit = [circuit]
  end
  for layer in circuit
    for g in layer
      push!(noisycircuit, g)
      nq = g[2]
      # n-qubit gate
      if nq isa Tuple
        # n -qubit Kraus operator
        noisegate = gate(noise2Q[1], length(nq); noise2Q[2]...)
        # if the single-qubit copy is return, use productnoise, and throw a warning
        size(noisegate,1) < 1<<length(nq) && error("$(length(nq))-qubit Kraus operators for the $(noise2Q[1]) noise not defined.\n")
        # correlated n-qubit noise
        push!(noisycircuit,(noise2Q[1], nq, noise2Q[2])) 
      # 1-qubit gate
      else
        push!(noisycircuit,(noise1Q[1], nq, noise1Q[2]))  
      end
    end
    if idle_noise
      busy_qubits = vcat([collect(g[2]) for g in layer]...)
      idle_qubits = filter(y -> y ∉ busy_qubits, 1:nqubits(circuit))
      for n in idle_qubits
        push!(noisycircuit, (noise1Q[1], n, noise1Q[2]))
      end
    end
  end
  return noisycircuit
end

