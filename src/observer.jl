"""
CircuitObserver is an implementation of an
observer object (<:AbstractObserver) which
implements custom measurements to perform
at each layer of the circuit evolution.
"""
struct CircuitObserver <: AbstractObserver
  measurements::Dict{String, Union{Function, String, Tuple}}
  results::Dict{String, Any}
end

"""
    ITensors.measurements(observer::CircuitObserver, metric::String)

Retrieve the results of a set of measurements
"""
ITensors.measurements(observer::CircuitObserver, metric::String) = observer.results[metric]


CircuitObserver() = CircuitObserver(Dict{String, Union{Function, String, Tuple}}(),Dict{String, Function}())
"""
    CircuitObserver(observables::Dict{String, <:Any})

Generate an observer given a list of measurements passed as a dictionary. 
"""
function CircuitObserver(measurements::Dict{String, <:Any}) 
  res = Dict{String, Any}()
  for obs in keys(measurements)
    res[obs] = []
  end
  return CircuitObserver(measurements, res)
end

CircuitObserver(measurement::Pair{String, <:Any}) = 
  CircuitObserver([measurement])

CircuitObserver(measurements::Vector{<:Pair{String, <:Any}}) = 
  CircuitObserver(Dict(measurements))


function CircuitObserver(measurements::Vector{<:Any})
  res = Dict{String, Any}()
  named_measurements = Dict{String, Union{Function, String, Tuple}}()
  for measurement in measurements
    name = measurement_name(measurement)
    named_measurements[name] = (measurement isa Pair ? last(measurement) : measurement)
    res[name] = []
  end
  return CircuitObserver(named_measurements, res)
end

CircuitObserver(measurement::Union{String,Tuple,Function}) = 
  CircuitObserver([measurement])



function measurement_name(measurement::Union{String,<:Tuple})
  name = (measurement isa String ? measurement*"(n)" : "")
  if measurement isa Tuple
    for c in measurement
      name *= string(c)
    end
  end
  return name
end

measurement_name(measurement::Pair{String, <:Any}) = 
  first(measurement)

measurement_name(measurement::Function) = 
  string(measurement)

has_customfunctions(observer::CircuitObserver) = 
  any(x -> isa(x,Function),values(observer.measurements))



function measure!(observer::CircuitObserver, M::Union{MPS,MPO}, ref_indices::Vector{<:Index})
  for measurement in keys(observer.measurements)
    if observer.measurements[measurement] isa Function
      res = observer.measurements[measurement](M)
    else
      res = measure(M, observer.measurements[measurement], ref_indices)
    end
    push!(observer.results[measurement], res)
  end
end

measure!(observer::CircuitObserver, M::Union{MPS,MPO}) = 
  measure!(observer, M, siteinds(M))



"""
    measure(ψ::MPS, measurement::Tuple{String,Int}, s::Vector{<:Index})
    measure(ψ::MPS, measurement::Tuple{String,Int})
  
Perform a measurement of a 1-local operator on an MPS ψ. The operator
is identifyed by a String (corresponding to a `gate`) and a site.
If an additional set of indices `s` is provided, the correct site is 
extracted by comparing the MPS with the index order in `s`.
"""
function measure(ψ::MPS, measurement::Tuple{String,Int}, s::Vector{<:Index})
  site0 = measurement[2]
  site = findsite(ψ,s[site0])
  ϕ = orthogonalize!(copy(ψ), site)
  ϕs = ϕ[site]
  obs_op = gate(measurement[1], firstsiteind(ϕ, site))
  T = noprime(ϕs * obs_op)
  return real((dag(T) * ϕs)[])
end

measure(ψ::MPS, measurement::Tuple{String,Int}) = 
  measure(ψ, measurement, siteinds(ψ))

"""
    measure(ψ::MPS, measurement::Tuple{String,Array{Int}}, s::Vector{<:Index})
    measure(ψ::MPS, measurement::Tuple{String,Array{Int}})
    measure(ψ::MPS, measurement::Tuple{String,AbstractRange}, s::Vector{<:Index})
    measure(ψ::MPS, measurement::Tuple{String,AbstractRange})
    measure(ψ::MPS, measurement::String, s::Vector{<:Index})
    measure(ψ::MPS, measurement::String)

Perform a measurement of a 1-local operator on an MPS ψ on a set of sites passed 
as a vector. If an additional set of indices `s` is provided, the correct site is 
extracted by comparing the MPS with the index order in `s`.
"""
function measure(ψ::MPS, measurement::Tuple{String,Array{Int}}, s::Vector{<:Index})
  result = []
  sites0 = measurement[2]
  ϕ = copy(ψ)
  for site0 in sites0
    site = findsite(ϕ,s[site0])
    orthogonalize!(ϕ, site)
    ϕs = ϕ[site]
    obs_op = gate(measurement[1], firstsiteind(ϕ, site))
    T = noprime(ϕs * obs_op)
    push!(result, real((dag(T) * ϕs)[]))
  end
  return result
end

measure(ψ::MPS, measurement::Tuple{String,Array{Int}}) = 
   measure(ψ,measurement,siteinds(ψ))

# for a range of sites
measure(ψ::MPS, measurement::Tuple{String,AbstractRange}, s::Vector{<:Index}) = 
  measure(ψ, (measurement[1], Array(measurement[2])), s)

measure(ψ::MPS, measurement::Tuple{String,AbstractRange}) = 
  measure(ψ, (measurement[1], Array(measurement[2])), siteinds(ψ))

# for every sites
measure(ψ::MPS, measurement::String, s::Vector{<:Index}) = 
  measure(ψ::MPS, (measurement, 1:length(ψ)),s)

## for every sites
measure(ψ::MPS, measurement::String) = 
  measure(ψ::MPS, (measurement, 1:length(ψ)), siteinds(ψ))


# at a given site
"""
    measure(ψ::MPS, measurement::Tuple{String,Int,String,Int}, s::Vector{<:Index})


Perform a measurement of a 2-body tensor-product operator on an MPS ψ. The two operators
are defined by Strings (for op name) and the sites. If an additional set of indices `s` is provided, the correct site is 
extracted by comparing the MPS with the index order in `s`.
"""
function measure(ψ::MPS, measurement::Tuple{String,Int,String,Int}, s::Vector{<:Index})
  obsA  = measurement[1]
  obsB  = measurement[3]
  siteA0 = measurement[2]
  siteB0 = measurement[4]
  siteA = findsite(ψ,s[siteA0])
  siteB = findsite(ψ,s[siteB0])
  
  if siteA > siteB
    obsA, obsB  = obsB, obsA
    siteA,siteB = siteB,siteA
  end
  ϕ = orthogonalize!(copy(ψ), siteA)
  ϕdag = prime(dag(ϕ),tags="Link")
  
  if siteA == siteB
    C = ϕ[siteA] * gate(obsA, firstsiteind(ϕ, siteA))
    C = noprime(C,tags="Site") * gate(obsA, firstsiteind(ϕ, siteA)) 
    C = noprime(C,tags="Site") * noprime(ϕdag[siteA])
    return real(C[])
  end
  if siteA == 1
    C = ϕ[siteA] * gate(obsA, firstsiteind(ϕ, siteA))
    C = noprime(C,tags="Site") * ϕdag[siteA]
  else
    C = prime(ϕ[siteA],commonind(ϕ[siteA],ϕ[siteA-1])) * gate(obsA, firstsiteind(ϕ, siteA))
    C = noprime(C,tags="Site") * ϕdag[siteA]
  end
  for j in siteA+1:siteB-1
    C = C * ϕ[j]
    C = C * ϕdag[j]
  end
  if siteB == length(ϕ)
    C = C * ϕ[siteB] * gate(obsB, firstsiteind(ϕ, siteB))
    C = noprime(C,tags="Site") * ϕdag[siteB]
  else
    C = C * prime(ϕ[siteB],commonind(ϕ[siteB],ϕ[siteB+1])) * gate(obsB, firstsiteind(ϕ, siteB))
    C = noprime(C,tags="Site") * ϕdag[siteB]
  end
  return real(C[])
end

measure(ψ::MPS, measurement::Tuple{String,Int,String,Int}) = 
  measure(ψ, measurement, siteinds(ψ))

function measure(ψ::MPS, measurement::Tuple{String,String}, s::Vector{<:Index})
  N = length(ψ)
  C = Matrix{Float64}(undef,N,N)
  for siteA in 1:N
    for siteB in 1:N
      m = (measurement[1],siteA,measurement[2],siteB)
      result = measure(ψ, m, s)
      C[siteA,siteB] = result
    end
  end
  return C
end

# for every sites
measure(ψ::MPS, measurement::Tuple{String,String}) = 
  measure(ψ::MPS, measurement, siteinds(ψ))



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




