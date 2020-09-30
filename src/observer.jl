"""
TomographyObserver is an implementation of an
observer object (<:AbstractObserver) which
implements custom measurements.
"""
struct TomographyObserver <: AbstractObserver
  fidelity::Vector{Float64}
  negative_loglikelihood::Vector{Float64}
  frobenius_distance::Vector{Float64}
  fidelity_bound::Vector{Float64}
  function TomographyObserver()
    return new([],[],[],[])
  end
end

negative_loglikelihood(obs::TomographyObserver) =
  obs.negative_loglikelihood

fidelity(o::TomographyObserver) = o.fidelity
fidelity_bound(o::TomographyObserver) = o.fidelity_bound
frobenius_distance(o::TomographyObserver) = o.frobenius_distance

function measure!(obs::TomographyObserver;
                  NLL=nothing,frob_dist=nothing,
                  F=nothing,Fbound=nothing)
  
  # Record negative log likelihood
  if !isnothing(NLL)
    push!(negative_loglikelihood(obs),NLL)
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
                      fout::String; M=nothing)
  h5rewrite(fout,"w") do file
    write(file,"nll", obs.negative_loglikelihood)
    write(file,"fidelity", obs.fidelity)
    write(file,"frobenius_distance", obs.frobenius_distance)
    write(file,"fidelity_bound", obs.fidelity_bound)
    write(file,"nll", obs.negative_loglikelihood)
    if !isnothing(M)
      write(file, "model", M)
    end
  end
end

