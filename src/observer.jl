const TomographyMeasurement = Vector{Vector{Float64}}

"""
TomographyObserver is an implementation of an
observer object (<:AbstractObserver) which
implements custom measurements.
"""
struct TomographyObserver <: AbstractObserver
  ops::Vector{String}
  sites::Vector{<:Index}
  measurements::Dict{String,TomographyMeasurement}
  fidelities::Vector{Float64}
  NLL::Vector{Float64}

  function TomographyObserver(M::Union{MPS,MPO})
    sites = (typeof(M)==MPS ? siteinds(M) : firstsiteinds(M))
    return TomographyObserver(sites)
  end
  
  function TomographyObserver(sites::Vector{<:Index})
    return new([],sites,Dict{String,TomographyMeasurement}(),[],[])
  end
  
  function TomographyObserver(ops::Vector{String},
                              sites::Vector{<:Index})
    measurements = Dict(o => TomographyMeasurement() for o in ops)
    return new(ops,sites,measurements,[],[])
  end
  
  function TomographyObserver(ops::Vector{String},
                              M::Union{MPS,MPO})
    sites = (typeof(M)==MPS ? siteinds(M) : firstsiteinds(M))
    return TomographyObserver(ops,sites)
  end
end

measurements(o::TomographyObserver) = o.measurements
fidelities(o::TomographyObserver)   = o.fidelities
sites(obs::TomographyObserver)      = obs.sites
ops(obs::TomographyObserver)        = obs.ops
NLL(obs::TomographyObserver)        = obs.NLL

function measurelocalops!(obs::TomographyObserver,
                          wf::ITensor,
                          i::Int)
  for o in ops(obs)
    m = dot(wf, noprime(gate(o,sites(obs),i)*wf))
    imag(m)>1e-8 && (@warn "encountered finite imaginary part when measuring $o")
    measurements(obs)[o][end][i]=real(m)
  end
end

function measure!(obs::TomographyObserver,ψ0::MPS;
                  nll=nothing,target=nothing,istargetlpdo::Bool=true)
  
  ψ = copy(ψ0)
  N = length(ψ)
  
  for o in ops(obs)
    push!(measurements(obs)[o],zeros(N))
  end

  # Measure 1-local operators
  for j in 1:N
    orthogonalize!(ψ,j)
    measurelocalops!(obs,ψ[j],j)
  end

  # Record negative log likelihood
  if !isnothing(nll)
    push!(NLL(obs),nll)
  end
  
  # Measure fidelity
  if !isnothing(target)
    F = fidelity(ψ,target)
    push!(fidelities(obs),F)
  end
end

function writeobserver(obs::TomographyObserver,fout::String; M=nothing)
  h5open(fout,"w") do file
    # TODO save measurements
    #write(file,"measurements",obs.measurements)
    write(file,"fidelity",obs.fidelities)
    write(file,"nll",obs.NLL)
    if !isnothing(M)
      write(file,"model",M)
    end
  end
end

