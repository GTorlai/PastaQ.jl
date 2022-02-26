"""
    measurement_counts(samples::Matrix{Pair{String, Int}}; fillzeros::Bool = true)

Generate a dictionary containing the measurement counts for a set 
of projectors, given a set of single-shot samples with different
measurement bases (i.e. QST).
"""
function measurement_counts(samples::Matrix{Pair{String, Int}}; fillzeros::Bool = true)
  counts = Dict{Tuple,Dict}()
  N = size(samples,2)
  
  if N > 8
    error("Full QST restricted to N ≤ 8")
  end
  # Loop over each measurement in the data
  for n in 1:size(samples,1)
    # Extract the measurement basis and  and outcome
    basis    = Tuple(first.(samples[n,:]))
    outcome  = Tuple(last.(samples[n,:]))
    
    # If new basis, add dictionary
    if !haskey(counts,basis)
      counts[basis] = Dict{Tuple,Int64}()
    end
    # Record outcome
    if !haskey(counts[basis],outcome)
      counts[basis][outcome] = 0
    end
    counts[basis][outcome] += 1
  end
  
  if fillzeros
    # Fill the counts with zeros for bitstring never observed
    for (k1,v1) in counts
      for index in 0:1<<N-1
        bitstring = reverse(Tuple(digits.(index,base=2,pad=N)'))
        if !haskey(counts[k1],bitstring)
          counts[k1][bitstring] = 0
        end
      end
    end
  end
  return counts
end

measurement_counts(samples::Matrix{String}; kwargs...) = 
  measurement_counts(convertdatapoints(samples); kwargs...)

"""
    measurement_counts(data::Matrix{Pair{String,Pair{String, Int}}}; fillzeros::Bool = true)

Generate a dictionary containing the measurement counts for a set 
of input states and measurement projectors (i.e. QPT).
"""
function measurement_counts(data::Matrix{Pair{String,Pair{String, Int}}}; fillzeros::Bool = true)
  if size(data,2) > 8
    error("Full QPT restricted to N ≤ 4")
  end
  newdata = []
  input_states = first.(data)
  measurements = last.(data)
  bases = first.(measurements)
  outcomes = last.(measurements)
  for n in 1:size(bases,1)
    m = convertdatapoint(outcomes[n,:], bases[n,:])
    tmp = []
    for j in 1:size(bases,2)
      push!(tmp, input_states[n,j])
      push!(tmp, m[j])
    end
    push!(newdata, tmp)
  end
  return measurement_counts(permutedims(hcat(newdata...)); fillzeros = fillzeros) 
end


"""
    empirical_probabilities(counts::Dict{Tuple,Dict})

Compute the probabilities from a set of measurement counts.
"""
function empirical_probabilities(counts::Dict{Tuple,Dict})
  probs = Dict{Tuple,Dict{Tuple,Float64}}(deepcopy(counts))
  
  for basis in keys(probs)
    # Get count dictionary for 2^N projectors in this basis
    counts_in_basis = probs[basis]
    
    # Total number of counts
    tot_counts = sum(values(counts_in_basis))
    
    # Build probabilities
    for projector in keys(counts_in_basis)
      probs[basis][projector] /= tot_counts
    end
  end
  return probs
end

empirical_probabilities(samples::Array; fillzeros::Bool=true) = 
  empirical_probabilities(measurement_counts(samples;fillzeros=fillzeros))

"""
    projector_matrix(probs::AbstractDict; process::Bool = false, return_probs::Bool = false)

Return the projector matrix, where each row corresponds to the vectorized
projector into different measurement bases contained into an input 
probability dictionary.

If `return_probs=true`, return also the 1d vector of probabilities (i.e. the
1D flattening of the dictionary).
"""
function design_matrix(probs::AbstractDict; process::Bool = false, return_probs::Bool = false)
  A = []
  p̂ = []
  n = first(keys(probs))
  q = siteind("Qubit")
  for (basis, projectors) in probs
    for (outcome,probability) in projectors
      Π_list = []
      for j in 1:length(outcome)
        g = 0.5 * (array(gate("Id", q) + (1-2*outcome[j]) * gate(basis[j], q)))
        Πj = process && isodd(j) ? g' : g 
        push!(Π_list,Πj)
      end
      Π = reduce(kron,Π_list)
      push!(A,conj(vec(Π)))
      push!(p̂,probability)
    end
  end
  
  return_probs && return copy(transpose(hcat(A...))), float.(p̂)
  return copy(transpose(hcat(A...)))
end

"""
    make_PSD(ρ::AbstractArray{<:Number,2})

Transform a non-positive density operator into positive.
The output (positive) density matrix is obtained by minimizing
the 2-norm distance between the input density matrix and a positive
density matrix.
"""
function make_PSD(ρ::AbstractArray{<:Number,2})
  μ, μ⃗ = eigen(ρ; sortby=x->-real(x))
  a = 0.0
  μ = real(μ)
  λ = zeros(length(μ))
  i = length(μ)
  while (μ[i] + a/i < 0.0)
    a += μ[i]
    λ[i] = 0.0
    i -= 1
  end
  for j in reverse(1:i)
    λ[j] = μ[j] + a/i
  end

  ρ_PSD = zeros(ComplexF64,length(μ),length(μ))
  for k in 1:length(μ)
    ρ_PSD += λ[k] * (μ⃗[:,k] * μ⃗[:,k]')
  end
  return ρ_PSD
end

