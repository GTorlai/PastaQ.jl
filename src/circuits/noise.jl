
function gate(::GateName"pauli_channel", N::Int = 1; 
              error_probabilities = prepend!(zeros(4^N-1),1),
              pauli_ops = ["Id","X","Y","Z"]) 
  length(error_probabilities) > (1 << 10) && error("Hilbert space too large")
  error_probabilities ./= sum(error_probabilities)
  kraus = zeros(Complex{Float64},1<<N,1<<N,4^N)
  basis = vec(reverse.(Iterators.product(fill(pauli_ops,N)...)|>collect))
  for (k,ops) in enumerate(basis)
    kraus[:,:,k] = √error_probabilities[k] * reduce(kron,gate.(ops)) 
  end
  return kraus
end

gate(::GateName"bit_flip", N::Int = 1; p::Number) = 
  gate("pauli_channel", N; error_probabilities = prepend!(p/(4^N-1) * ones(4^N-1), 1-p), pauli_ops = ["Id","X"])

gate(::GateName"phase_flip", N::Int = 1; p::Number) = 
  gate("pauli_channel", N; error_probabilities = prepend!(p/(4^N-1) * ones(4^N-1), 1-p), pauli_ops = ["Id","Z"])

function gate(::GateName"AD", N::Int = 1; γ::Number)
  K = zeros(Complex{Float64},1<<N,1<<N,2^N)
  kraus = zeros(2,2,2)
  kraus[:,:,1] = [1 0
                  0 sqrt(1-γ)]
  kraus[:,:,2] = [0 sqrt(γ)
                  0 0]
  N == 1 && return kraus 
  k1 = kraus[:,:,1]
  k2 = kraus[:,:,2]
  T = vec(Iterators.product(fill([k1,k2],N)...)|>collect)
  X = zeros(Complex{Float64},1<<N,1<<N,1<<N)
  for x in 1:1<<N
    X[:,:,x] = reduce(kron,T[x]) 
  end
  return X
end
#
# To accept the gate name "amplitude_damping"
gate(::GateName"amplitude_damping", N::Int=1; kwargs...) = gate("AD", N; kwargs...)

#function gate(::GateName"PD"; γ::Number)
#  kraus = zeros(2,2,2)
#  kraus[:,:,1] = [1 0
#                  0 sqrt(1-γ)]
#  kraus[:,:,2] = [0 0
#                  0 sqrt(γ)]
#  return kraus 
#end

## To accept the gate name "phase_damping"
#gate(::GateName"phase_damping"; kwargs...) = gate("PD"; kwargs...)
#
## To accept the gate name "dephasing"
#gate(::GateName"dephasing"; kwargs...) = gate("PD"; kwargs...)
#

# make general n-qubit
gate(::GateName"DEP", N::Int = 1; p::Number) =
  gate("pauli_channel", N; error_probabilities = prepend!(p/(4^N-1) * ones(4^N-1), 1-p), pauli_ops = ["Id","X","Y","Z"])

# To accept the gate name "depolarizing"
gate(::GateName"depolarizing", N::Int = 1; kwargs...) = 
  gate("DEP", N; kwargs...)

function applynoise(circuit::Vector, noise::Tuple; kwargs...) 
  applynoise(circuit, (noise1Q = noise, noise2Q = noise); kwargs...)
end 

# TODO: add insert!
function applynoise(circuit::Vector, noise::NamedTuple; idle = false, productnoise = false) 
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
        # apply noise channel as product of local channels
        if productnoise
          noisegate = gate(noise2Q[1],1; noise2Q[2]...)
          for n in nq
            push!(noisycircuit,(noise2Q[1], n, noise2Q[2]))
          end
        else
          # n -qubit Kraus operator
          noisegate = gate(noise2Q[1], length(nq); noise2Q[2]...)
          # if the single-qubit copy is return, use productnoise, and throw a warning
          size(noisegate,1) < 1<<length(nq) && error("$(length(nq))-qubit Kraus operators for the $(noise2Q[1]) noise not defined.\n")
          # correlated n-qubit noise
          push!(noisycircuit,(noise2Q[1], nq, noise2Q[2])) 
        end
      # 1-qubit gate
      else
        push!(noisycircuit,(noise1Q[1], nq, noise1Q[2]))  
      end
    end
    if idle
      busy_qubits = vcat([collect(g[2]) for g in layer]...)
      idle_qubits = filter(y -> y ∉ busy_qubits, 1:nqubits(circuit))
      for n in idle_qubits
        push!(noisycircuit, (noise1Q[1], n, noise1Q[2]))
      end
    end
  end
  return noisycircuit
end


#function apply_noise_to!(circuit::Vector, gatename::String, noise) 
#  length(size(circuit)) == 1 && (circuit = [circuit])
#  for layer in circuit
#    glist = findall(x -> x == gatename, first.(layer))
#    for k in 1:length(glist)
#      g = glist[k]
#      krauschannel = (first(noise),layer[glist[k]][2], last(noise))
#      @show layer[1]
#      @show krauschannel
#      splice!(layer, g:g-1, krauschannel) 
#      #glist .+= 1
#    end
#  end
#  @show circuit
#end
