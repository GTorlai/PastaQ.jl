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



function trotter1(H::OpSum, δτ::Number)
  onequbitgates = Tuple[]
  multiqubitgates = Tuple[]
  
  n = 1
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
      n = support[1] ≥ n ? support[1] : n
    # multi-qubit gate
    else
      g = (localop, support, (params..., f = x -> exp(-δτ * coupling * x),)) 
      push!(multiqubitgates, g)
      n = maximum(support) ≥ n ? maximum(support) : n
    end
  end
  
  sorted_multi_qubit = sort_gates(multiqubitgates)
  sorted_one_qubit = onequbitgates[sortperm([s[2] for s in onequbitgates])]
  
  # TODO: simplify this unison sorting loop
  tebd1 = Tuple[]
  g1_counter = 1; nmq = length(sorted_multi_qubit)
  gm_counter = 1; n1q = length(onequbitgates)
  for j in 1:n
    while (gm_counter ≤ nmq) && any(sorted_multi_qubit[gm_counter][2] .== j)
      push!(tebd1, sorted_multi_qubit[gm_counter])
      gm_counter += 1
    end
    while (g1_counter ≤ n1q) && any(sorted_one_qubit[g1_counter][2] .== j)
      push!(tebd1, sorted_one_qubit[g1_counter])
      g1_counter += 1
    end
  end
  #tebd1 = vcat(sorted_multi_qubit, sorted_one_qubit)
  return tebd1
end



"""
    trotter2(H::OpSum; δt::Float64=0.1, δτ=im*δt)

Generate a single layer of gates for one step of 2nd order TEBD.
"""
function trotter2(H::OpSum, δτ::Number)
  tebd1 = trotter1(H, δτ/2)
  tebd2 = copy(tebd1)
  append!(tebd2, reverse(tebd1))
  return tebd2
end

function trotter4(H::OpSum, δτ::Number)
  δτ1 = δτ / (4 - 4^(1/3)) 
  δτ2 = δτ - 4 * δτ1
  
  tebd2_δ1 = trotter2(H, δτ1)
  tebd2_δ2 = trotter2(H, δτ2)
  
  tebd4 = vcat(tebd2_δ1,tebd2_δ1)
  append!(tebd4, tebd2_δ2)
  append!(tebd4, vcat(tebd2_δ1,tebd2_δ1))
  return tebd4
end

"""
    trotterlayer(H::OpSum; order::Int = 2, kwargs...) 

Generate a single layer of gates for one step of TEBD.
"""
function trotterlayer(H::OpSum, δτ::Number; order::Int = 2)
  order == 1 && return trotter1(H, δτ) 
  order == 2 && return trotter2(H, δτ) 
  error("Automated Trotter circuits with order > 2 not yet implemented")
  # TODO: understand weird behaviour of trotter4
  #order == 4 && return trotter4(H, δτ) 
  #error("Automated Trotter circuits with order > 2 not yet implemented")
end


trottercircuit(H; kwargs...) = 
  _trottercircuit(H, get_times(;kwargs...); kwargs...)

trottercircuit(H::OpSum, T::Number; kwargs...) = 
  _trottercircuit(H, T, get_times(;kwargs...); kwargs...) 

function _trottercircuit(H::Vector{<:OpSum}, τs::Vector; order::Int = 2, layered::Bool = true, kwargs...)
  #if τs isa Vector{<:Complex}
  #  println("Running real-time evolution from t = $(imag(τs[1])) to t = $(imag(τs[end]))") 
  #else
  #  println("Running imaginary-time evolution from τ = im*$(τs[1]) to τ = im*$(τs[end])") 
  #end
  #@assert length(H) == (length(τs) -1)
  δτs = diff(τs)
  circuit = [trotterlayer(H[t], δτs[t]; order = order) for t in 1:length(δτs)] 
  layered && return circuit
  return vcat(circuit...)
end

_trottercircuit(H::Vector{<:OpSum}, δτ::Real; kwargs...) =
  _trottercircuit(H, collect(0.0:δτ:(length(H)*δτ)); kwargs...)

_trottercircuit(H::Vector{<:OpSum}, δτ::Complex; kwargs...) =
  _trottercircuit(H, im .* collect(0.0:imag(δτ):((length(H))*imag(δτ))); kwargs...)

_trottercircuit(H::OpSum, τs::Vector; kwargs...) = 
  _trottercircuit(repeat([H], length(τs)-1), τs; kwargs...) 

function _trottercircuit(H::OpSum, T::Number, δτ::Number; kwargs...)
  depth = abs(T / δτ)
  (depth-Int(floor(depth)) > 1e-5) && @warn "Incommensurate Trotter step!"
  depth = Int(floor(depth))
  return _trottercircuit(repeat([H], depth), δτ; kwargs...) 
end

get_times(; δt=nothing, δτ=nothing, ts=nothing, τs=nothing, kwargs...) = get_times(δt, δτ, ts, τs)

get_times(δt::Number,  δτ::Nothing, ts::Nothing,      τs::Nothing)   = im * δt 
get_times(δt::Nothing, δτ::Number,  ts::Nothing,      τs::Nothing)   = δτ  
get_times(δt::Nothing, δτ::Nothing, ts::Vector,       τs::Nothing)   = im .* ts 
get_times(δt::Nothing, δτ::Nothing, ts::Nothing,      τs::Vector)    = τs 
get_times(δt::Nothing, δτ::Nothing, ts::StepRangeLen, τs::Nothing)   = im .* collect(ts) 
get_times(δt::Nothing, δτ::Nothing, ts::Nothing,      τs::StepRangeLen) = collect(τs) 
 

