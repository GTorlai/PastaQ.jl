"""
Observer is an implementation of an
observer object (<:AbstractObserver) which
implements custom measurements.
"""
struct Observer <: AbstractObserver
  measurements::Dict{String, Pair{<:Union{Nothing,Function, String, Tuple},<:Any}}
end

Observer() = Observer(Dict{String, Pair{<:Union{Nothing,Function, String, Tuple},<:Any}}())

measurement(observer::Observer, observable::String) = first(observer.measurements[observable])
results(observer::Observer, observable::String) = last(observer.measurements[observable])

"""
    Observer(measurements::Vector{<:Pair{String,<:Any}})
    Observer(measurement::Pair{String,<:Any})
    CircuitObserver(observables::Dict{String, <:Any})

Initialize an Observer given a set of measurements, expressed
as Pairs (measurement_name => measurement).

The following measurement are currently allowed:
- `("X",j)`       ⟶   `Tr[ρ X̂ⱼ]`
- `"X"`           ⟶   `Tr[ρ X̂ⱼ] ∀j ∈ [1,N]` 
- `("X",i,"Y",j)` ⟶   `Tr[ρ X̂ᵢ Ŷⱼ]` 
- `("X","Y")`     ⟶   `Tr[ρ X̂ᵢ Ŷⱼ] ∀i,j ∈ [1,N]`
- `f(ρ) = [...]   ⟶   Arbitrary fubction of the state
"""
Observer(measurements::Vector{<:Pair{String,<:Any}}) =  
  Observer(Dict(measurements))

Observer(measurement::Pair{String,<:Any}) = 
  Observer([measurement])

function Observer(observables::Dict{String, <:Any})
  measurements = Dict{String, Pair{<:Union{Nothing,Function, String, Tuple},<:Any}}()
  for observable in keys(observables)
    measurements[observable] = observables[observable] => []
  end
  return Observer(measurements)
end



"""
    function Observer(observables::Vector{<:Any})
    Observer(measurement::Union{String,Tuple,Function})

Initialize an Observer given a set of measurements, expressed
as a vector. The name of each measurement for the Observer data
structure will be assigned automatically as the name of the 
measurement field.
"""
function Observer(observables::Vector{<:Any})
  measurements = Dict{String, Pair{<:Union{Nothing,Function, String, Tuple},<:Any}}()
  for observable in observables
    name = _measurement_name(observable)
    measurements[name] = (observable isa Pair{<:String, Any} ? last(observable) : observable) => []
  end
  return Observer(measurements)
end

Observer(measurement::Union{String,Tuple,Function}) = 
  Observer([measurement])





"""
    Base.push!(observer::Observer, observable::Pair{String, <:Any})
    Base.push!(observer::Observer, observable::Union{String,Tuple,<:Function})

Add a measurement (either named or not) to an existing Observer.
"""
function Base.push!(observer::Observer, observable::Pair{String, <:Any})
  observer.measurements[first(observable)] = last(observable) => []
  return observer
end

function Base.push!(observer::Observer, observable::Union{String,Tuple,<:Function})
  name = _measurement_name(observable)
  observer.measurements[name] = (observable isa Pair{String, <:Any} ? last(observable) : observable) => []
  return observer
end


"""
Assign the name to a measurement.
"""
_measurement_name(measurement::String) = 
  measurement

_measurement_name(measurement::Tuple) = 
  prod(ntuple(n -> measurement[n] isa AbstractString ? measurement[n] : "("*string(measurement[n])*")", length(measurement)))

_measurement_name(measurement::Pair{String, <:Any}) = 
  first(measurement)

_measurement_name(measurement::Function) = 
  string(measurement)

_measurement_name(measurement::Pair{<:Function,<:Union{Any,Tuple{<:Any}}}) = 
  string(first(measurement))  

_has_customfunctions(observer::Observer) = 
  any(x -> isa(x,Function), values(observer.measurements))

"""
    measure!(observer::Observer, M::Union{MPS,MPO,LPDO}, ref_indices::Vector{<:Index})
    measure!(observer::Observer, M::Union{MPS,MPO,LPDO})

Perform the measurements contained in Observer and collect the results.
`ref_indices` are a set of indices corresponding to the original quantum state,
before any eventual permutation to realized long-range gates. These indices can
be used to correctly perform measurements of specific sub-system of the state.
"""
function measure!(observer::Observer, M::Union{MPS,MPO,LPDO}, reference_indices::Vector{<:Index})
  # loop over the measurements
  for ID in keys(observer.measurements)
    # get the measurement
    measurement = first(observer.measurements[ID])
    if measurement isa Function
      result = measurement(M)
    elseif !isnothing(measurement)
      result = measure(M, measurement, reference_indices)
    end
    if !isnothing(measurement)
      push!(observer.measurements[ID][2], result)
    end
  end
end

measure!(observer::Observer, M::Union{MPS,MPO,LPDO}) =  
  measure!(observer, M, hilbertspace(M))


Base.copy(observer::Observer) = Observer(copy(observer.measurements)) 

