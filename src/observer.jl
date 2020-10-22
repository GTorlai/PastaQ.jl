"""
TomographyObserver is an implementation of an
observer object (<:AbstractObserver) which
implements custom measurements.
"""
struct TomographyObserver <: AbstractObserver
  fidelity::Vector{Float64}
  train_loss::Vector{Float64}
  test_loss::Vector{Float64}
  TP_distance::Vector{Float64}
  frobenius_distance::Vector{Float64}
  fidelity_bound::Vector{Float64}
  function TomographyObserver()
    return new([],[],[],[],[],[])
  end
end

train_loss(obs::TomographyObserver) = obs.train_loss
test_loss(obs::TomographyObserver) = obs.test_loss
TP_distance(obs::TomographyObserver) = obs.TP_distance
fidelity(o::TomographyObserver) = o.fidelity
fidelity_bound(o::TomographyObserver) = o.fidelity_bound
frobenius_distance(o::TomographyObserver) = o.frobenius_distance

function measure!(obs::TomographyObserver;
                  train_loss=nothing,
                  test_loss=nothing,
                  TP_dist=nothing,
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
  if !isnothing(TP_dist)
    push!(TP_distance(obs),TP_dist)
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
    write(file,"TP_distance",  obs.TP_distance)
    write(file,"fidelity", obs.fidelity)
    write(file,"frobenius_distance", obs.frobenius_distance)
    write(file,"fidelity_bound", obs.fidelity_bound)
    if !isnothing(model)
      write(file, "model", model)
    end
  end
end

