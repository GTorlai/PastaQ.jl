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

function trotter1(δτ::Number, H::Vector{<:Tuple}; kwargs...)
  layer = Tuple[]
  for k in 1:length(H)
    length(H[k]) > 3 && error("Only the format (coupling, opname, support) currently allowed")
    coupling, Hdata... = H[k]
    opname = first(Hdata)
    layer=vcat(layer,[(x -> exp(-δτ * coupling * x), Hdata...)]) 
  end
  return layer 
end

function trotter1(δτ::Number, hilbert::Vector{<:Index}, H::Vector{<:Tuple}; lindbladians = [], atol = 1e-15, kwargs...)
  layer = buildcircuit(hilbert, trotter1(δτ, H))
  if !isempty(lindbladians)
    for lindblad in lindbladians
      rate, opname, site = lindblad
      !(site isa Int) && error("Only single-body lindblad operators allowed")
      s = hilbert[site]
      
      L = array(gate(opname, s))
      G = -im * δτ * rate * kron(conj(L), L)
      
      expG = reshape(exp(G),(size(L)..., size(L)...))
      expG = reshape(permutedims(expG, (1,3,2,4)), size(G))
      @assert ishermitian(expG)
      
      λ, U = eigen(expG)
      λsqrt = diagm(sqrt.(λ .+ atol))
      K = U * λsqrt 
      K = reshape(K, (size(L)..., size(K)[2]))
      krausind = Index(size(K)[3]; tags="kraus")
      T = ITensors.itensor(K, prime(s), ITensors.dag(s), krausind)
      layer = vcat(layer, [T])
      
      R = transpose(L) * conj(L)
      T = exp(0.5 * im * rate * δτ * op(R, s))
      layer = vcat(layer, [T])
    end
  end
  return layer 
end

"""
    trotter2(H::OpSum; δt::Float64=0.1, δτ=im*δt)
Generate a single layer of gates for one step of 2nd order TEBD.
"""
function trotter2(δτ::Number, args...; kwargs...)
  tebd1 = trotter1(δτ/2, args...; kwargs...)
  tebd2 = vcat(tebd1, reverse(tebd1))
  return tebd2
end

function trotter4(δτ::Number, args...; kwargs...)
  δτ1 = δτ / (4 - 4^(1/3)) 
  δτ2 = δτ - 4 * δτ1
  
  tebd2_δ1 = trotter2(δτ1, args...; kwargs...)
  tebd2_δ2 = trotter2(δτ2, args...; kwargs...)
  
  tebd4 = vcat(tebd2_δ1,tebd2_δ1)
  tebd4 = vcat(tebd4, tebd2_δ2)
  tebd4 = vcat(tebd4, vcat(tebd2_δ1,tebd2_δ1))
  return tebd4
end


"""
    trotterlayer(H::OpSum; order::Int = 2, kwargs...) 
Generate a single layer of gates for one step of TEBD.
"""
function trotterlayer(args...; order::Int = 2, kwargs...)
  order == 1 && return trotter1(args...; kwargs...) 
  order == 2 && return trotter2(args...; kwargs...) 
  error("Automated Trotter circuits with order > 2 not yet implemented")
  # TODO: understand weird behaviour of trotter4
  #order == 4 && return trotter4(H, δτ) 
end

function _trottercircuit(H::Vector{<:Vector{Tuple}}, τs::Vector; layered::Bool = false, lindbladians = [], kwargs...)
  !isempty(lindbladians) && error("Trotter simulation with Lindblad operators requires a set of indices")
  @assert length(H) == (length(τs) -1) || length(H) == length(τs)
  δτs = diff(τs)
  circuit = [trotterlayer(δτs[t], H[t]; kwargs...) for t in 1:length(δτs)] 
  layered && return circuit
  return reduce(vcat, circuit)
end

function _trottercircuit(hilbert::Vector{<:Index}, H::Vector{<:Vector{Tuple}}, τs::Vector; layered::Bool = false, kwargs...)
  @assert length(H) == (length(τs) -1) || length(H) == length(τs)
  δτs = diff(τs)
  circuit = [trotterlayer(δτs[t], hilbert, H[t]; kwargs...) for t in 1:length(δτs)] 
  layered && return circuit
  return reduce(vcat, circuit)
end
  
function _trottercircuit(H::Vector{<:Tuple}, τs::Vector; layered::Bool = false, lindbladians = [], kwargs...)
  !isempty(lindbladians) && error("Trotter simulation with Lindblad operators requires a set of indices")
  nlayers = length(τs) - 1
  Δ = τs[2] - τs[1]
  layer = trotterlayer(Δ, H; kwargs...)
  layered && return [layer for _ in 1:nlayers]
  return reduce(vcat, [layer for _ in 1:nlayers])
end

function _trottercircuit(hilbert::Vector{<:Index}, H::Vector{<:Tuple}, τs::Vector; layered::Bool = false, kwargs...)
  nlayers = length(τs) - 1
  Δ = τs[2] - τs[1]
  layer = trotterlayer(Δ, hilbert, H; kwargs...)
  layered && return [layer for _ in 1:nlayers]
  return reduce(vcat, [layer for _ in 1:nlayers])
end

trottercircuit(args...; kwargs...) = 
  _trottercircuit(args..., get_times(; kwargs...); kwargs...)



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

