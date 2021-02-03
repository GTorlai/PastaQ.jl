function trace_outputsites(L::LPDO)
  N = length(L)
  
  Φ = ITensor[]

  tmp = noprime(ket(L,1),tags="Output") * bra(L,1)
  Cdn = combiner(commonind(tmp,L.X[2]),commonind(tmp,L.X[2]'))
  push!(Φ,tmp * Cdn)

  for j in 2:N-1
    tmp = noprime(ket(L,j),tags="Output") * bra(L,j)
    Cup = Cdn
    Cdn = combiner(commonind(tmp,L.X[j+1]),commonind(tmp,L.X[j+1]'))
    push!(Φ,tmp * Cup * Cdn)
  end
  tmp = noprime(ket(L,N),tags="Output") * bra(L,N)
  Cup = Cdn
  push!(Φ,tmp * Cup)
  return MPO(Φ)
end

function splitobserverargs!(observer::Observer) 
  for obs in keys(observer.measurements)
    observable = first(observer.measurements[obs])
    if !isnothing(observable)
      arg = last(observable)
      if arg isa MPO
        observer.measurements[obs] = (observer.measurements[obs][1][1] => makeChoi(observer.measurements[obs][1][2])) => []
      end
    end
  end
  return observer
end

function configure!(observer::Union{Observer,Nothing},
                    optimizer::Optimizer, 
                    batchsize::Int,
                    measurement_frequency::Int,
                    train_data::Matrix,
                    test_data::Union{Array,Nothing})
   
  if isnothing(observer)
    observer = Observer()
  end

  params = Dict{String,Any}()
  # grab the optimizer parameters
  params[string(typeof(optimizer))] = Dict{Symbol,Any}() 
  for par in fieldnames(typeof(optimizer))
    if !(getfield(optimizer,par) isa Vector{<:ITensor})
      params[string(typeof(optimizer))][par] = getfield(optimizer,par)
    end
  end
  # batchsize 
  params["batchsize"] = batchsize
  # storing this can help to back out simulation time and observables evolution
  params["measurement_frequency"] = measurement_frequency
  params["dataset_size"] = size(train_data,1)

  observer.measurements["parameters"] = (nothing => params)
  
  observer.measurements["train_loss"] = (nothing => [])
  if !isnothing(test_data)
    observer.measurements["test_loss"] = (nothing => [])
  end
  
  return observer
end

function update!(observer::Observer,
                 normalized_model::LPDO,
                 best_model::LPDO,
                 simulation_time::Float64,
                 train_loss::Float64,
                 test_loss::Union{Nothing,Float64})
  
  observer.measurements["simulation_time"] = nothing => simulation_time 
  push!(observer.measurements["train_loss"][2], train_loss)
  if !isnothing(test_loss)
    push!(observer.measurements["test_loss"][2], test_loss)
  end
  
  measure!(observer, normalized_model)
end


printmetric(name::String, metric::Int) = @printf("%s = %d  ",name,metric)
printmetric(name::String, metric::Float64) = @printf("%s = %-4.4f  ",name,metric)
printmetric(name::String, metric::AbstractArray) = 
  @printf("%s = [...]  ",name)

function printmetric(name::String, metric::Complex)
  if imag(metric) < 1e-8
    @printf("%s = %-4.4f  ",name,real(metric))
  else
    @printf("%s = %.4f±i%-4.4f  ",name,real(metric),imag(metric))
  end
end


function printobserver(epoch::Int, observer::Observer, print_metrics::Union{Bool,String,AbstractArray})
  
  (print_metrics isa Bool) && !print_metrics && return
  
  @printf("%-4d  ",epoch)
  @printf("⟨logP⟩ = %-4.4f  ", results(observer,"train_loss")[end]) 
  if haskey(observer.measurements,"test_loss")
    @printf("(%.4f)  ",results(observer,"test_loss")[end])
  end
  if !isempty(print_metrics)
    if print_metrics isa String
      printmetric(print_metrics, results(observer,print_metrics)[end])  
    else
      for metric in print_metrics
        printmetric(metric, results(observer,metric)[end])
      end
    end
  end
  println()
end

