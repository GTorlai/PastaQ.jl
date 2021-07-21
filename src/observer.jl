"""
Observer is an implementation of an
observer object (<:AbstractObserver) which
implements custom measurements.
"""
struct Observer <: AbstractObserver
  measurements::Dict{String,Pair{<:Union{Nothing,Function,String,Tuple},<:Any}}
end

Observer() = Observer(Dict{String,Pair{<:Union{Nothing,Function,String,Tuple},<:Any}}())

function measurement(observer::Observer, observable::String)
  return first(observer.measurements[observable])
end
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
Observer(measurements::Vector{<:Pair{String,<:Any}}) = Observer(Dict(measurements))

Observer(measurement::Pair{String,<:Any}) = Observer([measurement])

function Observer(observables::Dict{String,<:Any})
  measurements = Dict{String,Pair{<:Union{Nothing,Function,String,Tuple},<:Any}}()
  for observable in keys(observables)
    measurements[observable] = observables[observable] => []
  end
  return Observer(measurements)
end

"""
    Observer(observables::Vector{<:Any})
    Observer(measurement::Union{String,Tuple,Function})

Initialize an Observer given a set of measurements, expressed
as a vector. The name of each measurement for the Observer data
structure will be assigned automatically as the name of the 
measurement field.
"""
function Observer(observables::Vector{<:Any})
  measurements = Dict{String,Pair{<:Union{Nothing,Function,String,Tuple},<:Any}}()
  for observable in observables
    name = _measurement_name(observable)
    measurements[name] =
      (observable isa Pair{<:String,Any} ? last(observable) : observable) => []
  end
  return Observer(measurements)
end

Observer(measurement::Union{String,Tuple,Function}) = Observer([measurement])

"""
    Base.push!(observer::Observer, observable::Pair{String, <:Any})
    Base.push!(observer::Observer, observable::Union{String,Tuple,<:Function})

Add a measurement (either named or not) to an existing Observer.
"""
function Base.push!(observer::Observer, observable::Pair{String,<:Any})
  observer.measurements[first(observable)] = last(observable) => []
  return observer
end

function Base.push!(observer::Observer, observable::Union{String,Tuple,<:Function})
  name = _measurement_name(observable)
  observer.measurements[name] =
    (observable isa Pair{String,<:Any} ? last(observable) : observable) => []
  return observer
end

"""
Assign the name to a measurement.
"""
_measurement_name(measurement::String) = measurement

function _measurement_name(measurement::Tuple)
  return prod(
    ntuple(
      n -> if measurement[n] isa AbstractString
        measurement[n]
      else
        "(" * string(measurement[n]) * ")"
      end,
      length(measurement),
    ),
  )
end

_measurement_name(measurement::Pair{String,<:Any}) = first(measurement)

_measurement_name(measurement::Function) = string(measurement)

function _measurement_name(measurement::Pair{<:Function,<:Union{Any,Tuple{<:Any}}})
  return string(first(measurement))
end

function _has_customfunctions(observer::Observer)
  return any(x -> isa(x, Function), first.(values(observer.measurements)))
end

"""
    measure!(observer::Observer, M::Union{MPS,MPO,LPDO}, ref_indices::Vector{<:Index})
    measure!(observer::Observer, M::Union{MPS,MPO,LPDO})

Perform the measurements contained in Observer and collect the results.
`ref_indices` are a set of indices corresponding to the original quantum state,
before any eventual permutation to realized long-range gates. These indices can
be used to correctly perform measurements of specific sub-system of the state.
"""
function measure!(
  observer::Observer, M::Union{MPS,MPO,LPDO}, reference_indices::Vector{<:Index}
)
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

function measure!(observer::Observer, M::Union{MPS,MPO,LPDO})
  return measure!(observer, M, hilbertspace(M))
end

Base.copy(observer::Observer) = Observer(copy(observer.measurements))


"""
    configure!(observer::Union{Observer,Nothing},
               optimizer::Optimizer, 
               batchsize::Int,
               measurement_frequency::Int,
               train_data::Matrix,
               test_data::Union{Array,Nothing})

Configure the Observer for quantum tomography:
- Save the Optimizer parameters
- Save batchsize/measurement_frequency/dataset_size
- Initialize train_loss (test_loss) measurements
"""
function configure!(
  observer::Union{Observer,Nothing},
  optimizer::Optimizer,
  batchsize::Int,
  measurement_frequency::Int,
  train_data::Matrix,
  test_data::Union{Array,Nothing},
)
  if isnothing(observer)
    observer = Observer()
  end

  params = Dict{String,Any}()
  # grab the optimizer parameters
  params["optimizer"] = Dict{Symbol,Any}()
  params["optimizer"][:name] = string(typeof(optimizer))
  #params[string(typeof(optimizer))] = Dict{Symbol,Any}()
  for par in fieldnames(typeof(optimizer))
    if !(getfield(optimizer, par) isa Vector{<:ITensor})
      params["optimizer"][par] = getfield(optimizer, par)
    end
  end

  # batchsize 
  params["batchsize"] = batchsize
  # storing this can help to back out simulation time and observables evolution
  params["measurement_frequency"] = measurement_frequency
  
  params["dataset_size"] = isnothing(test_data) ? size(train_data, 1) : size(train_data, 1) + size(test_data, 1)

  observer.measurements["parameters"] = (nothing => params)

  observer.measurements["train_loss"] = (nothing => [])
  if !isnothing(test_data)
    observer.measurements["test_loss"] = (nothing => [])
  end

  return observer
end

"""
    update!(observer::Observer,
            normalized_model::Union{MPS,MPO,LPDO},
            best_model::LPDO,
            simulation_time::Float64,
            train_loss::Float64,
            test_loss::Union{Nothing,Float64})

Update the observer for quantum tomography.
Perform measuremenst and record data.
"""
function update!(
  observer::Observer,
  normalized_model::Union{MPS,MPO,LPDO},
  best_model::LPDO,
  simulation_time::Float64,
  train_loss::Float64,
  test_loss::Union{Nothing,Float64}
)
  observer.measurements["simulation_time"] = nothing => simulation_time
  push!(observer.measurements["train_loss"][2], train_loss)
  if !isnothing(test_loss)
    push!(observer.measurements["test_loss"][2], test_loss)
  end
  if normalized_model isa LPDO{MPS}
    measure!(observer, normalized_model.X)
  else
    measure!(observer, normalized_model)
  end
  return observer
end

