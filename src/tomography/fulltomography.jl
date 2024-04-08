using Convex
using SCS
import MathOptInterface
const MOI = MathOptInterface

# Added a specific import statement for Convex, SCS and MathOptInterface. 
# The earlier implementation ran into a ModuleError here.

function tomography(
  probabilities::Dict{Tuple,<:Dict},
  sites::Vector{<:Index};
  method::String="linear_inversion",
  fillzeros::Bool=true,
  process::Bool=false,
  trρ::Number=1.0,
  max_iters::Int=10000,
  kwargs...,
)

  # Generate the projector matrix corresponding to the probabilities.
  A, p = design_matrix(probabilities; return_probs=true, process=process)

  if (method == "LI" || method == "linear_inversion")
    # Invert the Born rule and reshape
    ρ_vec = pinv(A) * p
    d = Int(sqrt(size(ρ_vec, 1)))
    ρ̂ = reshape(ρ_vec, (d, d))

    ρ̂ .= ρ̂ * (trρ / tr(ρ̂))
    # Make PSD
    ρ̂ = make_PSD(ρ̂)
  else
    d = Int(sqrt(size(A, 2)))
    N = Int(sqrt(d))
    n = N ÷ 2

    # Variational density matrix
    ρ = Convex.ComplexVariable(d, d)

    if (method == "LS" || method == "least_squares")
      # Minimize the cost function C = ||A ρ⃗ - p̂||²
      cost_function = Convex.norm(A * vec(ρ) - p)
    elseif (method == "ML" || method == "maximum_likelihood") 
      # Minimize the negative log likelihood:
      # Ensured uniformity in documentation and implementation of the maximum_likelihood case. 
      # the keyword was implemented as "MLS" in some places, "MLE" in some places. Now everything is consistent and two-lettered.
      cost_function = -p' * Convex.log(real(A * vec(ρ)) + 1e-10)
    else
      error("Tomography method not recognized
       Currently available methods: - LI  : linear inversion
                                      LS  : least squares
                                      ML : maximum likelihood")
    end

    # Contrained the trace and enforce positivity and hermitianity
    if process
      function tracepreserving(ρ)
        for j in 1:n
          subsystem_dims = [2 for _ in 1:(N + 1 - j)]
          ρ = Convex.partialtrace(ρ, j + 1, subsystem_dims)
        end
        return ρ
      end

      constraints = [
        Convex.tr(ρ) == (1 << n) * trρ
        Convex.isposdef(ρ)
        tracepreserving(ρ) == Matrix{Float64}(I, 1 << n, 1 << n)
        ρ == ρ'
      ]
    else
      constraints = [
        Convex.tr(ρ) == trρ
        Convex.isposdef(ρ)
        ρ == ρ'
      ]
    end
    # Use Convex.jl to solve the optimization
    problem = Convex.minimize(cost_function, constraints)
    optimizer = SCS.Optimizer()
    MOI.set(optimizer, MOI.RawOptimizerAttribute("verbose"), false)
    MOI.set(optimizer, MOI.RawOptimizerAttribute("max_iters"), max_iters)
    Convex.solve!(problem, optimizer; verbose=false)
    ρ̂ = ρ.value
  end
  return PastaQ.itensor(ρ̂, sites)
end

"""
    measurement_counts(samples::Matrix{Pair{String, Int}}; fillzeros::Bool = true)

Generate a dictionary containing the measurement counts for a set
of projectors, given a set of single-shot samples with different
measurement bases (i.e. QST).
"""
function measurement_counts(samples::Matrix{Pair{String,Int}}; fillzeros::Bool=true)
  counts = Dict{Tuple,Dict}()
  N = size(samples, 2)

  if N > 8
    error("Full QST restricted to N ≤ 8")
  end
  # Loop over each measurement in the data
  for n in 1:size(samples, 1)
    # Extract the measurement basis and  and outcome
    basis = Tuple(first.(samples[n, :]))
    outcome = Tuple(last.(samples[n, :]))

    # If new basis, add dictionary
    if !haskey(counts, basis)
      counts[basis] = Dict{Tuple,Int64}()
    end
    # Record outcome
    if !haskey(counts[basis], outcome)
      counts[basis][outcome] = 0
    end
    counts[basis][outcome] += 1
  end

  if fillzeros
    # Fill the counts with zeros for bitstring never observed
    for (k1, v1) in counts
      for index in 0:(1 << N - 1)
        bitstring = reverse(Tuple(digits.(index, base=2, pad=N)'))
        if !haskey(counts[k1], bitstring)
          counts[k1][bitstring] = 0
        end
      end
    end
  end
  return counts
end

function measurement_counts(samples::Matrix{String}; kwargs...)
  return measurement_counts(convertdatapoints(samples); kwargs...)
end

"""
    measurement_counts(data::Matrix{Pair{String,Pair{String, Int}}}; fillzeros::Bool = true)

Generate a dictionary containing the measurement counts for a set
of input states and measurement projectors (i.e. QPT).
"""
function measurement_counts(
  data::Matrix{Pair{String,Pair{String,Int}}}; fillzeros::Bool=true
)
  if size(data, 2) > 8
    error("Full QPT restricted to N ≤ 4")
  end
  newdata = []
  input_states = first.(data)
  measurements = last.(data)
  bases = first.(measurements)
  outcomes = last.(measurements)
  for n in 1:size(bases, 1)
    m = convertdatapoint(outcomes[n, :], bases[n, :])
    tmp = []
    for j in 1:size(bases, 2)
      push!(tmp, input_states[n, j])
      push!(tmp, m[j])
    end
    push!(newdata, tmp)
  end
  return measurement_counts(permutedims(hcat(newdata...)); fillzeros=fillzeros)
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

function empirical_probabilities(samples::Array; fillzeros::Bool=true)
  return empirical_probabilities(measurement_counts(samples; fillzeros=fillzeros))
end

"""
    projector_matrix(probs::AbstractDict; process::Bool = false, return_probs::Bool = false)

Return the projector matrix, where each row corresponds to the vectorized
projector into different measurement bases contained into an input
probability dictionary.

If `return_probs=true`, return also the 1d vector of probabilities (i.e. the
1D flattening of the dictionary).
"""
function design_matrix(probs::AbstractDict; process::Bool=false, return_probs::Bool=false)
  A = []
  p̂ = []
  n = first(keys(probs))
  st = SiteType("Qubit")
  for (basis, projectors) in probs
    for (outcome, probability) in projectors
      Π_list = []
      for j in 1:length(outcome)
        g = 0.5 * (gate("Id", st) + (1 - 2 * outcome[j]) * gate(basis[j], st))
        Πj = process && isodd(j) ? g' : g
        push!(Π_list, Πj)
      end
      Π = reduce(kron, Π_list)
      push!(A, conj(vec(Π)))
      push!(p̂, probability)
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
  μ, μ⃗ = eigen(ρ; sortby=x -> -real(x))
  a = 0.0
  μ = real(μ)
  λ = zeros(length(μ))
  i = length(μ)
  while (μ[i] + a / i < 0.0)
    a += μ[i]
    λ[i] = 0.0
    i -= 1
  end
  for j in reverse(1:i)
    λ[j] = μ[j] + a / i
  end

  ρ_PSD = zeros(ComplexF64, length(μ), length(μ))
  for k in 1:length(μ)
    ρ_PSD += λ[k] * (μ⃗[:, k] * μ⃗[:, k]')
  end
  return ρ_PSD
end
