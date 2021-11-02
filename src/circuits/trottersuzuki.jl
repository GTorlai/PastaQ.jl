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
      # XXX Zygote
      #push!(onequbitgates, g)
      onequbitgates = vcat(onequbitgates, [g])
      n = support[1] ≥ n ? support[1] : n
    # multi-qubit gate
    else
      g = (localop, support, (params..., f = x -> exp(-δτ * coupling * x),)) 
      multiqubitgates = vcat(multiqubitgates, [g])
      #push!(multiqubitgates, g)
      n = maximum(support) ≥ n ? maximum(support) : n
    end
  end
  
  # XXX Zygote: Add this back (Zygote fails)
  #sorted_multi_qubit = sort_gates(multiqubitgates)
  sorted_multi_qubit = multiqubitgates
  sorted_one_qubit = onequbitgates[sortperm([s[2] for s in onequbitgates])]
  
  # TODO: simplify this unison sorting loop
  tebd1 = Tuple[]
  g1_counter = 1; nmq = length(sorted_multi_qubit)
  gm_counter = 1; n1q = length(onequbitgates)
  for j in 1:n
    while (gm_counter ≤ nmq) && any(sorted_multi_qubit[gm_counter][2] .== j)
      # XXX Zygote
      #push!(tebd1, sorted_multi_qubit[gm_counter])
      tebd1 = vcat(tebd1, [sorted_multi_qubit[gm_counter]])
      gm_counter += 1
    end
    while (g1_counter ≤ n1q) && any(sorted_one_qubit[g1_counter][2] .== j)
      # XXX Zygote
      #push!(tebd1, sorted_one_qubit[g1_counter])
      tebd1 = vcat(tebd1, [sorted_one_qubit[g1_counter]])
      g1_counter += 1
    end
  end
  return tebd1
end



"""
    trotter2(H::OpSum; δt::Float64=0.1, δτ=im*δt)

Generate a single layer of gates for one step of 2nd order TEBD.
"""
function trotter2(H::OpSum, δτ::Number)
  tebd1 = trotter1(H, δτ/2)
  # XXX Zygote
  #tebd2 = copy(tebd1)
  #append!(tebd2, reverse(tebd1))
  tebd2 = vcat(tebd1, reverse(tebd1))
  return tebd2
end

function trotter4(H::OpSum, δτ::Number)
  δτ1 = δτ / (4 - 4^(1/3)) 
  δτ2 = δτ - 4 * δτ1
  
  tebd2_δ1 = trotter2(H, δτ1)
  tebd2_δ2 = trotter2(H, δτ2)
  
  tebd4 = vcat(tebd2_δ1,tebd2_δ1)
  # XXX Zygote
  #append!(tebd4, tebd2_δ2)
  #append!(tebd4, vcat(tebd2_δ1,tebd2_δ1))
  tebd4 = vcat(tebd4, tebd2_δ2)
  tebd4 = vcat(tebd4, vcat(tebd2_δ1,tebd2_δ1))
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

function _trottercircuit(H::Vector{<:OpSum}, τs::Vector; order::Int = 2, layered::Bool = true, kwargs...)
  @assert length(H) == (length(τs) -1) || length(H) == length(τs)
  δτs = diff(τs)
  circuit = [trotterlayer(H[t], δτs[t]; order = order) for t in 1:length(δτs)] 
  # XXX Zygote
  #layered && return circuit
  return reduce(vcat, circuit)
end

function _trottercircuit(H::OpSum, τs::Vector; kwargs...)
  nlayers = length(τs) - 1
  # XXX Zygote
  #Hs = repeat([H], nlayers)
  Hs = [H for _ in 1:nlayers]
  _trottercircuit(Hs, τs; kwargs...) 
end

get_times(; δt=nothing, δτ=nothing, t=nothing, τ=nothing, ts=nothing, τs=nothing, kwargs...) = get_times(δt, δτ, t, τ, ts, τs)


get_times(δt::Nothing, δτ::Nothing, t::Nothing, τ::Nothing, ts::Vector,  τs::Nothing)   = im .* ts 
get_times(δt::Nothing, δτ::Nothing, t::Nothing, τ::Nothing, ts::Nothing, τs::Vector)    = τs 

function get_times(δt::Nothing, δτ::Number, t::Nothing, τ::Number, ts::Nothing, τs::Nothing)
  depth = abs(τ / δτ)
  (depth-Int(floor(depth)) > 1e-5) && @warn "Incommensurate Trotter step!"
  return collect(0.0:δτ:τ) 
end

get_times(δt::Number, δτ::Nothing, t::Number, τ::Nothing, ts::Nothing, τs::Nothing) =
  im .* get_times(δτ, δt, τ, t, ts, τs)

get_times(δt::Nothing, δτ::Nothing, t::Nothing, τ::Nothing, ts::AbstractRange, τs::Nothing)       = 
  get_times(δt, δτ, t, τ, collect(ts), τs) 

get_times(δt::Nothing, δτ::Nothing, t::Nothing, τ::Nothing, ts::Nothing,       τs::AbstractRange) = 
  get_times(δt, δτ, t, τ, ts, collect(τs))


get_times(δt::Number,  δτ::Nothing, t::Nothing, τ::Nothing, ts::Nothing, τs::Nothing)   =
  error("Total time `t` not set.")

get_times(δt::Nothing, δτ::Number,  t::Nothing, τ::Nothing, ts::Nothing, τs::Nothing)   =
  error("Total imaginary time `τ` not set")

get_times(δt::Nothing,  δτ::Nothing, t::Number, τ::Nothing, ts::Nothing, τs::Nothing)   =
  error("Trotter step `δt` not set.")

get_times(δt::Nothing,  δτ::Nothing, t::Nothing, τ::Number, ts::Nothing, τs::Nothing)   =
  error("Imaginary Trotter step `δτ` not set.")

