"""
TomographyObserver is an implementation of an
observer object (<:AbstractObserver) which
implements custom measurements.
"""
struct TomographyObserver <: AbstractObserver
  fidelity::Vector{Float64}
  train_loss::Vector{Float64}
  test_loss::Vector{Float64}
  trace_preserving_distance::Vector{Float64}
  frobenius_distance::Vector{Float64}
  fidelity_bound::Vector{Float64}
  function TomographyObserver()
    return new([],[],[],[],[],[])
  end
end

train_loss(obs::TomographyObserver) = obs.train_loss
test_loss(obs::TomographyObserver) = obs.test_loss
trace_preserving_distance(obs::TomographyObserver) = 
  obs.trace_preserving_distance
fidelity(o::TomographyObserver) = o.fidelity
fidelity_bound(o::TomographyObserver) = o.fidelity_bound
frobenius_distance(o::TomographyObserver) = o.frobenius_distance

function measure!(obs::TomographyObserver;
                  train_loss=nothing,
                  test_loss=nothing,
                  trace_preserving_dist=nothing,
                  frob_dist=nothing,
                  F=nothing,Fbound=nothing)
  
  ## Record negative log likelihood
  if !isnothing(train_loss)
    push!(obs.train_loss,train_loss)
  end
  if !isnothing(test_loss)
    push!(obs.test_loss,test_loss)
  end
  # Measure fidelity
  if !isnothing(trace_preserving_dist)
    push!(trace_preserving_distance(obs),trace_preserving_dist)
  end
  # Measure fidelity
  if !isnothing(F)
    push!(fidelity(obs),F)
  end
  # Measure fidelity
  if !isnothing(frob_dist)
    push!(frobenius_distance(obs),frob_dist)
  end
  # Measure fidelity
  if !isnothing(Fbound)
    push!(fidelity_bound(obs),Fbound)
  end
end

function saveobserver(obs::TomographyObserver,
                      fout::String; model=nothing)
  h5rewrite(fout,"w") do file
    write(file,"train_loss", obs.train_loss)
    write(file,"test_loss",  obs.test_loss)
    write(file,"trace_preserving_distance",  obs.trace_preserving_distance)
    write(file,"fidelity", obs.fidelity)
    write(file,"frobenius_distance", obs.frobenius_distance)
    write(file,"fidelity_bound", obs.fidelity_bound)
    if !isnothing(model)
      write(file, "model", model)
    end
  end
end






struct CircuitObserver <: AbstractObserver
  observables::Dict{String, Any}
  functions::Dict{String, Function}
  results::Dict{String, Any}
end

ITensors.measurements(observer::CircuitObserver, metric::String) = observer.results[metric]


function CircuitObserver(observables::Dict{String, <:Any}) 
  res = Dict{String, Any}()
  for obs in keys(observables)
    res[obs] = []
  end
  return CircuitObserver(observables,Dict{String, Function}(), res)
end

CircuitObserver(observable::Pair{String, <:Any}) = 
  CircuitObserver([observable])

CircuitObserver(observables::Vector{<:Pair{String, <:Any}}) = 
  CircuitObserver(Dict(observables))


function CircuitObserver(functions::Dict{String, <:Function}) 
  res = Dict{String, Any}()
  observables = Dict{String, Any}()
  for f in keys(functions)
    res[f] = []
  end
  return CircuitObserver(observables,functions, res)
end

CircuitObserver(f::Pair{String, <:Function}) = 
  CircuitObserver([f])

CircuitObserver(functions::Vector{<:Pair{String, <:Function}}) = 
  CircuitObserver(Dict(functions))


function CircuitObserver(functions::Dict{String, <:Function}, observables::Dict{String, <:Any})
  res = Dict{String, Any}()
  for obs in keys(observables)
    res[obs] = []
  end
  for f in keys(functions)
    res[f] = []
  end
  return CircuitObserver(observables, functions, res)
end

CircuitObserver(observables::Dict{String, <:Any}, functions::Dict{String, <:Function}) = 
  CircuitObserver(functions,observables)


CircuitObserver(functions::Vector{<:Pair{String, <:Function}}, observables::Vector{<:Pair{String, <:Any}}) = 
  CircuitObserver(Dict(functions),Dict(observables))

CircuitObserver(observables::Vector{<:Pair{String, <:Any}}, functions::Vector{<:Pair{String, <:Function}}) = 
  CircuitObserver(functions, observables)

CircuitObserver(f::Pair{String, <:Function}, observable::Pair{String, <:Any}) =
  CircuitObserver([f],[observable])

CircuitObserver(observable::Pair{String, <:Any}, f::Pair{String, <:Function}) = 
  CircuitObserver([f],[observable])

CircuitObserver(functions::Vector{<:Pair{String, <:Function}}, observable::Pair{String, <:Any}) =
  CircuitObserver(functions, [observable])

CircuitObserver(f::Pair{String, <:Function}, observables::Vector{<:Pair{String, <:Any}}) =
  CircuitObserver([f],observables)




# TODO: double check this one here!!!
function CircuitObserver(unnamed_observables::Vector{<:Any})#Union{String,Tuple}})
  res = Dict{String, Any}()
  observables = Dict{String, Any}()
  for observable in unnamed_observables
    name = (observable isa String ? observable*"(n)" : "")
    if observable isa Tuple
      for c in observable
        name *= string(c)
      end
    end
    observables[name] = observable
    res[name] = []
  end
  return CircuitObserver(observables, Dict{String, Function}(),res)
end

CircuitObserver(unnamed_observable::Union{String,Tuple}) = CircuitObserver([unnamed_observable])

function CircuitObserver(unnamed_functions::Vector{<:Function})
  res = Dict{String, Any}()
  functions = Dict{String, Function}()
  functions[string(unnamed_functions[1])] = unnamed_functions[1]
  res[string(unnamed_functions[1])] = []
  if length(unnamed_functions) > 1
    for j in 2:length(unnamed_functions)
      functions[string(unnamed_functions[j])] = unnamed_functions[j]
      res[string(unnamed_functions[j])] = []
    end
  end
  return CircuitObserver(Dict{String, Any}(),functions, res) 
end

CircuitObserver(unnamed_f::Function) = 
  CircuitObserver([unnamed_f])



function CircuitObserver(unnamed_observables::Vector{<:Any},unnamed_functions::Vector{<:Function})
  dict_obs = CircuitObserver(unnamed_observables)
  dict_f   = CircuitObserver(unnamed_functions)

  res = merge(dict_obs.results,dict_f.results)
  return CircuitObserver(dict_obs.observables, dict_f.functions, res)
end

CircuitObserver(unnamed_functions::Vector{<:Function}, unnamed_observables::Vector{<:Any}) = 
  CircuitObserver(unnamed_observables, unnamed_functions)


#CircuitObserver(unnamed_function::Function, unnamed_observables::Vector{<:Any}) =
#  CircuitObserver([unnamed_function],unnamed_observables)
#CircuitObserver(unnamed_observables::Vector{<:Any}, unnamed_function::Function) =
#  CircuitObserver(unnamed_function,unnamed_function)
#CircuitObserver(unnamed_functions::Vector{<:Function},unnamed_observable::Any) = 
#  CircuitObserver(unnamed_functions, [unnamed_observable])
#CircuitObserver(unnamed_observable::Any, unnamed_functions::Vector{<:Function}) = 
#  CircuitObserver(unnamed_functions, unnamed_observable)
#CircuitObserver(unnamed_function::Function, unnamed_observable::Any) =
#  CircuitObserver([unnamed_function], [unnamed_observable])
#CircuitObserver(unnamed_observable::Any, unnamed_function::Function) =
#  CircuitObserver(unnamed_function,unnamed_observable)



function measure!(observer::CircuitObserver, M::Union{MPS,MPO})
  if !isempty(observer.observables)
    for obs_name in keys(observer.observables)
      res = measure(M, observer.observables[obs_name])
      push!(observer.results[obs_name], res)
    end
  end
  if !isempty(observer.functions)
    for obs_name in keys(observer.functions)
      result = observer.functions[obs_name](M)
      push!(observer.results[obs_name], result) 
    end
  end
end






"""
Measure 1-body operator
"""

# at a given site
function measure(ψ::MPS, measurement::Tuple{String,Int})
  site = measurement[2]
  ϕ = orthogonalize!(copy(ψ), site)
  ϕs = ϕ[site]
  obs_op = gate(measurement[1], firstsiteind(ϕ, site))
  T = noprime(ϕs * obs_op)
  return real((dag(T) * ϕs)[])
end

# for a set of sites (passed as a vector)
function measure(ψ::MPS, measurement::Tuple{String,Array{Int}})
  result = []
  sites = measurement[2]
  ϕ = copy(ψ)
  for site in sites
    orthogonalize!(ϕ, site)
    ϕs = ϕ[site]
    obs_op = gate(measurement[1], firstsiteind(ϕ, site))
    T = noprime(ϕs * obs_op)
    push!(result, real((dag(T) * ϕs)[]))
  end
  return result
end

# for a range of sites
measure(ψ::MPS, measurement::Tuple{String,AbstractRange}) = 
  measure(ψ, (measurement[1], Array(measurement[2])))

# for every sites
measure(ψ::MPS, measurement::String) = 
  measure(ψ::MPS, (measurement, 1:length(ψ)))



#function measure(ψ::MPS, measurement::Tuple{String, Int})
