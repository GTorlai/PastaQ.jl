
function trotterlayer(H::OpSum; τ::Float64 = 0.1)
 
  onequbitgates = Tuple[]
  twoqubitgates = Tuple[]

  for k in 1:length(H)
    coupling = ITensors.coef(H[k])
    O = ITensors.ops(H[k])
    length(O) > 1 && error("only a single operator allowed per term")
    localop = ITensors.name(O[1])
    support = ITensors.sites(O[1])
    params = ITensors.params(O[1])

    #  TODO add kwargs of gate here
    # single-qubit gate
    if length(support) == 1
      g = (localop, support[1], (params..., f = x -> exp(-im * τ * coupling * x),)) 
      push!(onequbitgates, g)
    # multi-qubit gate
    else
      g = (localop, support, (params..., f = x -> exp(-im * τ/2 * coupling * x),)) 
      push!(twoqubitgates, g)
    end
  end
  
  n = max(nqubits(onequbitgates),nqubits(twoqubitgates))
  
  orderedlayer = Tuple[]
  for k in 1:n
    Q2 = [g[2] for g in twoqubitgates]
    mask = findall(x -> x == 1, [any(x -> x == 1, [q == k for q in Q]) for Q in Q2])
    for glocation in mask
      push!(orderedlayer, twoqubitgates[glocation])
    end
    for (i,glocation) in enumerate(mask)
      deleteat!(twoqubitgates, glocation-i+1)
    end
  end
  
  staircase = copy(orderedlayer)
  append!(staircase, onequbitgates)
  append!(staircase, reverse(orderedlayer))
  return staircase
end

function trottercircuit(H::OpSum, T::Float64; τ::Float64 = 0.1, layered::Bool = true) 
  circuit = repeat([trotterlayer(H; τ = τ)], Int(ceil(T / τ)))
  layered && return circuit
  return vcat(circuit...)
end



