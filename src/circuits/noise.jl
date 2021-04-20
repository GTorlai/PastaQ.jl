# Noise model gate definitions
function gate(::GateName"1qubit-pauli_error"; pX::Number = 0.0, pY::Number = 0.0, pZ::Number = 0.0)
  kraus = zeros(Complex{Float64},2,2,4)
  kraus[:,:,1] = √(1-pX-pY-pZ) * gate("Id")
  kraus[:,:,2] = √pX * gate("X")
  kraus[:,:,3] = √pY * gate("Y")
  kraus[:,:,4] = √pZ * gate("Z")
  return kraus 
end

function gate(::GateName"2qubit-pauli_error"; probs = prepend!(zeros(15),1))
  kraus = zeros(Complex{Float64},4,4,16)
  basis = vec(reverse.(Iterators.product(fill(["Id","X","Y","Z"],2)...)|>collect))
  for (k,ops) in enumerate(basis)
    kraus[:,:,k] = √probs[k] * kron(gate(ops[1]), gate(ops[2]))
  end
  return kraus 
end

function gate(::GateName"pauli_error", N::Int = 1; kwargs...) 
  N == 1 && return gate("1qubit-pauli_error"; kwargs...)
  N == 2 && return gate("2qubit-pauli_error"; kwargs...)
end


gate(::GateName"pauli_channel"; kwargs...) = gate("pauli_error"; kwargs...)


gate(::GateName"bit_flip"; p::Number) = 
  gate("pauli_error"; pX = p)[:,:,1:2]
  
gate(::GateName"phase_flip"; p::Number) = 
  gate("pauli_error"; pZ = p)[:,:,[1,4]]

gate(::GateName"bit_phase_flip"; p::Number) = 
  gate("pauli_error"; pY = p)[:,:,[1,3]]

gate(::GateName"phase_bit_flip"; kwargs...) = 
  gate("bit_phase_flip"; kwargs...)

function gate(::GateName"AD"; γ::Number)
  kraus = zeros(2,2,2)
  kraus[:,:,1] = [1 0
                  0 sqrt(1-γ)]
  kraus[:,:,2] = [0 sqrt(γ)
                  0 0]
  return kraus 
end

# To accept the gate name "amplitude_damping"
gate(::GateName"amplitude_damping"; kwargs...) = gate("AD"; kwargs...)

function gate(::GateName"PD"; γ::Number)
  kraus = zeros(2,2,2)
  kraus[:,:,1] = [1 0
                  0 sqrt(1-γ)]
  kraus[:,:,2] = [0 0
                  0 sqrt(γ)]
  return kraus 
end

# To accept the gate name "phase_damping"
gate(::GateName"phase_damping"; kwargs...) = gate("PD"; kwargs...)

# To accept the gate name "dephasing"
gate(::GateName"dephasing"; kwargs...) = gate("PD"; kwargs...)


gate(::GateName"1qDEP"; p::Number) = 
  gate("pauli_error"; pX = p/3.0, pY = p/3.0, pZ = p/3.0)

gate(::GateName"2qDEP"; p::Number) = 
  gate("pauli_error",2; probs = prepend!((p/15) * ones(15), 1-p))

function gate(::GateName"DEP", N::Int = 1; kwargs...)
  N == 1 && return gate("1qDEP"; kwargs...)
  N == 2 && return gate("2qDEP"; kwargs...)
end

# To accept the gate name "depolarizing"
gate(::GateName"depolarizing", N::Int = 1; kwargs...) = 
  gate("DEP", N; kwargs...)


function applynoise(circuit::Vector, noise::Tuple; kwargs...) 
  applynoise(circuit, (noise1Q = noise, noise2Q = noise); kwargs...)
end 

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
          if size(noisegate,1) < 1<<length(nq)
            println("WARNING: $(length(nq))-qubit Kraus operators for the $(noise2Q[1]) noise not defined.\nApplying instead the tensor-product of single-qubit noise")
            for n in nq
              push!(noisycircuit,(noise2Q[1], n, noise2Q[2]))
            end
          # correlated n-qubit noise
          else
            #for n in nq
            #  push!(noisycircuit,(noise2Q[1], n, noise2Q[2]))
            #end
            push!(noisycircuit,(noise2Q[1], nq, noise2Q[2])) 
          end
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
