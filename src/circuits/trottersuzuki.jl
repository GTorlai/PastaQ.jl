function getsites(g) 
  x = filter(x -> x isa Tuple, g)
  isempty(x) && return x
  return only(x)
end

sort_gates_by(g) = 
  TupleTools.sort(getsites(g))

function sort_gates_lt(g1, g2)
  if length(g1) ≠ length(g2)
    return length(g1) > length(g2)
  end
  return g1 < g2
end

sort_gates(gates) = 
  sort(gates; by=sort_gates_by, lt=sort_gates_lt)


function trotter1(H::OpSum; δt::Float64=0.1, δτ=im*δt)
  onequbitgates = Tuple[]
  twoqubitgates = Tuple[]
  
  for k in 1:length(H)
    coupling = ITensors.coef(H[k])
    O = ITensors.ops(H[k])
    length(O) > 1 && error("only a single operator allowed per term")
    localop = ITensors.name(O[1])
    support = ITensors.sites(O[1])
    params = ITensors.params(O[1])

    # single-qubit gate
    if length(support) == 1
      g = (localop, support[1], (params..., f = x -> exp(-δτ * coupling * x),)) 
      push!(onequbitgates, g)
    # multi-qubit gate
    else
      g = (localop, support, (params..., f = x -> exp(-δτ * coupling * x),)) 
      push!(twoqubitgates, g)
    end
  end
  sorted_two_qubit = sort_gates(twoqubitgates)
  sorted_one_qubit = onequbitgates[sortperm([s[2] for s in onequbitgates])]
  
  # TODO: place the one qubit gates of a site right after all two qubit gates 
  # involving that site have been done
  return vcat(sorted_two_qubit,sorted_one_qubit)
end

function trotter2(H::OpSum; δt::Float64=0.1, δτ=im*δt)
  onequbitgates = Tuple[]
  twoqubitgates = Tuple[]
  
  for k in 1:length(H)
    coupling = ITensors.coef(H[k])
    O = ITensors.ops(H[k])
    length(O) > 1 && error("only a single operator allowed per term")
    localop = ITensors.name(O[1])
    support = ITensors.sites(O[1])
    params = ITensors.params(O[1])

    # single-qubit gate
    if length(support) == 1
      g = (localop, support[1], (params..., f = x -> exp(-δτ * coupling * x),)) 
      push!(onequbitgates, g)
    # multi-qubit gate
    else
      g = (localop, support, (params..., f = x -> exp(-δτ/2 * coupling * x),)) 
      push!(twoqubitgates, g)
    end
  end
  sorted_two_qubit = sort_gates(twoqubitgates)
  sorted_one_qubit = onequbitgates[sortperm([s[2] for s in onequbitgates])]
  staircase = copy(sorted_two_qubit)
  append!(staircase, sorted_one_qubit)
  append!(staircase, reverse(sorted_two_qubit))
  return staircase
end


function trotterlayer(H::OpSum; order::Int = 2, kwargs...)
  order == 1 && return trotter1(H; kwargs...) 
  order == 2 && return trotter2(H; kwargs...) 
  error("Automated Trotter circuits with order > 2 not yet implemented")
end


function trottercircuit(H::OpSum, T::Float64; δt::Float64=0.1, δτ = im*δt, layered::Bool = true) 
  circuit = repeat([trotterlayer(H; δt = δt, δτ = δτ)], Int(ceil(T / abs(δτ))))
  layered && return circuit
  return vcat(circuit...)
end

