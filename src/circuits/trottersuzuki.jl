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


"""

WORKING WITH TUPLES  (TEMPORARY)
"""

# simplified version
function trotter1(H::Vector{<:Tuple}, δτ::Number)
  layer = Tuple[]
  for k in 1:length(H)
    length(H[k]) > 3 && error("Only the format (coupling, opname, support) currently allowed")
    coupling, localop, support = H[k]
    layer = vcat(layer, [(localop, support, (f = x -> exp(-δτ * coupling * x),))]) 
  end
  return layer 
end


"""
    trotter2(H::OpSum; δt::Float64=0.1, δτ=im*δt)

Generate a single layer of gates for one step of 2nd order TEBD.
"""
function trotter2(H::Vector{<:Tuple}, δτ::Number)
  tebd1 = trotter1(H, δτ/2)
  tebd2 = vcat(tebd1, reverse(tebd1))
  return tebd2
end

function trotter4(H::Vector{<:Tuple}, δτ::Number)
  δτ1 = δτ / (4 - 4^(1/3)) 
  δτ2 = δτ - 4 * δτ1
  
  tebd2_δ1 = trotter2(H, δτ1)
  tebd2_δ2 = trotter2(H, δτ2)
  
  tebd4 = vcat(tebd2_δ1,tebd2_δ1)
  tebd4 = vcat(tebd4, tebd2_δ2)
  tebd4 = vcat(tebd4, vcat(tebd2_δ1,tebd2_δ1))
  return tebd4
end

"""
    trotterlayer(H::OpSum; order::Int = 2, kwargs...) 

Generate a single layer of gates for one step of TEBD.
"""
function trotterlayer(H::Vector{<:Tuple}, δτ::Number; order::Int = 2)
  order == 1 && return trotter1(H, δτ) 
  order == 2 && return trotter2(H, δτ) 
  error("Automated Trotter circuits with order > 2 not yet implemented")
  # TODO: understand weird behaviour of trotter4
  #order == 4 && return trotter4(H, δτ) 
  #error("Automated Trotter circuits with order > 2 not yet implemented")
end

function _trottercircuit(H::Vector{<:Vector{Tuple}}, τs::Vector; order::Int = 2, layered::Bool = false, kwargs...)
  @assert length(H) == (length(τs) -1) || length(H) == length(τs)
  δτs = diff(τs)
  circuit = [trotterlayer(H[t], δτs[t]; order = order) for t in 1:length(δτs)] 
  layered && return circuit
  return reduce(vcat, circuit)
end


#XXX simplified version for Zygote
function _trottercircuit(H::Vector{<:Tuple}, τs::Vector; order::Int = 2, layered::Bool = false, kwargs...)
  nlayers = length(τs) - 1
  # XXX: Zygote: this breaks (?) 
  #circuit = [trotterlayer(H, τ; order = order) for τ in τs]
  #!layered && return reduce(vcat, circuit)
  #return circuit
  Δ = τs[2] - τs[1]
  layer = trotterlayer(H, Δ; order = order)
  layered && return [layer for _ in 1:nlayers]
  return reduce(vcat, [layer for _ in 1:nlayers])
end

trottercircuit(H; kwargs...) = 
  _trottercircuit(H, get_times(;kwargs...); kwargs...)



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


