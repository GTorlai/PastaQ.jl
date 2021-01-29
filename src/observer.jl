"""
CircuitObserver is an implementation of an
observer object (<:AbstractObserver) which
implements custom measurements to perform
at each layer of the circuit evolution.
"""
struct Observer <: AbstractObserver
  measurements::Dict{String, Union{Function, String, Tuple}}
  results::Dict{String, Any}
end

Observer() = Observer(Dict{String, Union{Function, String, Tuple}}(),Dict{String, Function}())

"""
    CircuitObserver(observables::Dict{String, <:Any})

Generate an observer given a list of measurements passed as a dictionary. 
"""
function Observer(measurements::Dict{String, <:Any}) 
  res = Dict{String, Any}()
  for obs in keys(measurements)
    res[obs] = []
  end
  return Observer(measurements, res)
end

Observer(measurement::Pair{String, <:Any}) = 
  Observer([measurement])

Observer(measurements::Vector{<:Pair{String, <:Any}}) = 
  Observer(Dict(measurements))


function Observer(measurements::Vector{<:Any})
  res = Dict{String, Any}()
  named_measurements = Dict{String, Union{Function, String, Tuple}}()
  for measurement in measurements
    name = measurement_name(measurement)
    named_measurements[name] = (measurement isa Pair ? last(measurement) : measurement)
    res[name] = []
  end
  return Observer(named_measurements, res)
end

Observer(measurement::Union{String,Tuple,Function}) = 
  Observer([measurement])


function Base.push!(observer::Observer, measurement::Union{Pair,String,Tuple,Function})
  name = measurement_name(measurement)
  observer.measurements[name] = (measurement isa Pair ? last(measurement) : measurement)
  observer.results[name] = []
  return observer
end

function Base.push!(observer::Observer, measurements...)
  for measurement in measurements
    Base.push!(observer, measurement)
  end
end

measurement_name(measurement::String) = 
  measurement

measurement_name(measurement::Tuple) = 
  prod(ntuple(n -> measurement[n] isa AbstractString ? measurement[n] : "("*string(measurement[n])*")", length(measurement)))

measurement_name(measurement::Pair{String, <:Any}) = 
  first(measurement)

measurement_name(measurement::Function) = 
  string(measurement)

has_customfunctions(observer::Observer) = 
  any(x -> isa(x,Function), values(observer.measurements))


function save(observer::Observer, output_path::String)
  h5rewrite(output_path) do file
    write(file,"results", observer.results["parameters"])
  end
end



