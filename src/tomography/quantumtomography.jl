function tomography(probabilities::Dict{Tuple,<:Dict}, sites::Vector{<:Index}; 
                    method::String="linear_inversion", 
                    fillzeros::Bool=true, 
                    process::Bool = false,
                    trρ::Number = 1.0,
                    max_iters::Int=10000,
                    kwargs...)
  
  # Generate the projector matrix corresponding to the probabilities.
  A, p = design_matrix(probabilities; return_probs = true, process = process)
  
  if (method == "LI"  || method == "linear_inversion")
    # Invert the Born rule and reshape
    ρ_vec = pinv(A) * p
    d = Int(sqrt(size(ρ_vec,1)))
    ρ̂ = reshape(ρ_vec,(d,d))
    
    ρ̂ .= ρ̂ * (trρ / tr(ρ̂))
    # Make PSD
    ρ̂ = make_PSD(ρ̂)
  else
    d = Int(sqrt(size(A,2)))
    N = Int(sqrt(d))
    n = N ÷ 2
    
    # Variational density matrix
    ρ = Convex.ComplexVariable(d,d)
    
    if (method == "LS"  || method == "least_squares") 
      # Minimize the cost function C = ||A ρ⃗ - p̂||²
      cost_function = Convex.norm(A * vec(ρ) - p) 
    elseif (method == "MLE" || method == "maximum_likelihood")
      # Minimize the negative log likelihood:
      cost_function = - p' * Convex.log(real(A * vec(ρ)) + 1e-10)
    else
      error("Tomography method not recognized
       Currently available methods: - LI  : linear inversion
                                      LS  : least squares
                                      MLS : maximum likelihood")
    end

    # Contrained the trace and enforce positivity and hermitianity 
    if process 
      function tracepreserving(ρ)
        for j in 1:n
          subsystem_dims = [2 for _ in 1:(N+1-j)]
          ρ = Convex.partialtrace(ρ,j+1,subsystem_dims)
        end
        return ρ
      end
      
      constraints = [Convex.tr(ρ) == (1<<n) * trρ
                    Convex.isposdef(ρ)
                    tracepreserving(ρ) == Matrix{Float64}(I,1<<n,1<<n)
                    ρ == ρ']
    else
      constraints = [Convex.tr(ρ) == trρ 
                     Convex.isposdef(ρ) 
                     ρ == ρ']
    end
    # Use Convex.jl to solve the optimization
    problem = Convex.minimize(cost_function,constraints)
    Convex.solve!(problem, () -> SCS.Optimizer(verbose=false,max_iters=max_iters),verbose=false)
    ρ̂ = ρ.value
  end
  # TODO: CHECK
  return itensor(ρ̂, reverse(sites)', ITensors.dag(reverse(sites)))
end


tomography(data::Matrix{Pair{String, Int}}; method::String="linear_inversion", fillzeros::Bool=true, kwargs...) = 
  tomography(empirical_probabilities(data; fillzeros=fillzeros), siteinds("Qubit", size(data,2)); method = method, kwargs...)

tomography(data::Matrix{Pair{String, Int}}, sites::Vector{<:Index}; method::String="linear_inversion", fillzeros::Bool=true, kwargs...) = 
  tomography(empirical_probabilities(data; fillzeros=fillzeros), sites; method = method, kwargs...)

tomography(data::Matrix{Pair{String,Pair{String, Int}}}; kwargs...) = 
  tomography(data, siteinds("Qubit", size(data,2)); kwargs...)

function tomography(data::Matrix{Pair{String,Pair{String, Int}}}, sites::Vector{<:Index}; method::String="linear_inversion", fillzeros::Bool=true, kwargs...) 
  sites_in  = addtags.(sites, "Input")
  sites_out = addtags.(sites, "Output")
  process_sites = Index[]
  for j in 1:size(data,2) 
    push!(process_sites, sites_in[j]) 
    push!(process_sites, sites_out[j]) 
  end
  tomography(empirical_probabilities(data; fillzeros=fillzeros), process_sites; method = method, process = true, kwargs...)
end






"""
    tomography(train_data::Matrix{Pair{String,Pair{String, Int}}}, L::LPDO;
               optimizer::Optimizer,
               observer! = nothing,
               batchsize::Int64 = 100,
               epochs::Int64 = 1000,
               kwargs...)

Run quantum process tomography using a variational model `L` to fit `train_data`.
The model can be either a unitary circuit (MPO) or a Choi matrix (LPDO).

# Arguments:
- `train_data`: pairs of preparation/ (basis/outcome): `("X+"=>"X"=>0, "Z-"=>"Y"=>1, "Y+"=>"Z"=>0, …)`.
 - `L`: variational model (MPO/LPDO).
 - `optimizer`: algorithm used to update the model parameters.
 - `observer!`: if provided, keep track of training metrics.
 - `batch_size`: number of samples used for one gradient update.
 - `epochs`: number of training iterations.
 - `target`: target quantum process (if provided, compute fidelities).
 - `test_data`: data for computing cross-validation.
 - `outputpath`: if provided, save metrics on file.
"""
function tomography(
  train_data::AbstractMatrix, L::LPDO; (observer!)=nothing, kwargs...
)
  # Read arguments
  opt = get(kwargs, :optimizer, Optimisers.Descent(0.01))
  batchsize::Int64 = get(kwargs, :batchsize, 100)
  epochs::Int64 = get(kwargs, :epochs, 1000)
  trace_preserving_regularizer = get(kwargs, :trace_preserving_regularizer, 0.0)
  observe_step::Int64 = get(kwargs, :observe_step, 1)
  test_data = get(kwargs, :test_data, nothing)
  outputpath = get(kwargs, :fout, nothing)
  print_metrics = get(kwargs, :print_metrics, [])
  outputpath = get(kwargs, :outputpath, nothing)
  outputlevel = get(kwargs, :outputlevel, 1)
  savestate = get(kwargs, :savestate, false)
  earlystop = get(kwargs, :earlystop, false)

  model = copy(L)
  isqpt = train_data isa Matrix{Pair{String,Pair{String,Int}}}
  localnorm = isqpt ? 2.0 : 1.0
  
  # observer is not passed but earlystop is called
  observer! = (isnothing(observer!) && earlystop) ? Observer() : observer!
  
  # observer is defined
  if !isnothing(observer!)
    observer!["train_loss"] = nothing 
    if !isnothing(test_data)
      observer!["test_loss"] = nothing
    end
    # add the standard early stop function to the observer
    if earlystop
      stop_if(; loss::Vector) = stopif_loss(; loss = loss, ϵ = 1e-3, min_iter = 10)
      observer!["earlystop"] = stopif
    end 
  end

  # initialize optimizer
  st = PastaQ.state(opt, model)
  optimizer = (opt, st)
  
  @assert size(train_data, 2) == length(model)
  !isnothing(test_data) && @assert size(test_data)[2] == length(model)

  batchsize = min(size(train_data)[1], batchsize)
  num_batches = Int(floor(size(train_data)[1] / batchsize))

  tot_time = 0.0
  best_model = nothing
  best_testloss = 1000.0
  test_loss = nothing

  # Training iterations
  for ep in 1:epochs
    ep_time = @elapsed begin
      train_data = train_data[shuffle(1:end), :]
      train_loss = 0.0

      # Sweep over the data set
      for b in 1:num_batches
        batch = train_data[((b - 1) * batchsize + 1):(b * batchsize), :]

        normalized_model = copy(model)
        sqrt_localnorms = []
        normalize!(normalized_model; 
                   (sqrt_localnorms!)=sqrt_localnorms, 
                   localnorm=localnorm)
        
        grads, loss = gradients(
          normalized_model,
          batch;
          sqrt_localnorms=sqrt_localnorms,
          trace_preserving_regularizer=trace_preserving_regularizer,
        )

        nupdate = ep * num_batches + b
        train_loss += loss / Float64(num_batches)
        update!(model, grads, optimizer)
      end
    end # end @elapsed
    !isnothing(observer!) && push!(last(observer!["train_loss"]), train_loss)
    tot_time += ep_time
    
    # measurement stage
    if ep % observe_step == 0
      normalized_model = copy(model)
      sqrt_localnorms = []
      normalize!(normalized_model; 
                 (sqrt_localnorms!)=sqrt_localnorms, 
                 localnorm=localnorm)
      
      if !isnothing(test_data)
        test_loss = nll(normalized_model, test_data)
        !isnothing(observer!) && push!(last(observer!["test_loss"]), test_loss)
        if test_loss ≤ best_testloss
          best_testloss = test_loss
          best_model = copy(normalized_model)
        end
      else
        best_model = copy(normalized_model)
      end

      # update observer
      if !isnothing(observer!)
        loss = (!isnothing(test_data) ? results(observer!, "test_loss") : 
                                        results(observer!, "train_loss"))
        
        model_to_observe = (isqpt && (normalized_model isa LPDO{MPS}) ? choi_mps_to_unitary_mpo(normalized_model) : 
                                  !isqpt && (normalized_model isa LPDO{MPS}) ? normalized_model.X :
                                                                               normalized_model)
        update!(observer!, model_to_observe; train_loss = train_loss,
                                             test_loss  = test_loss)
      end

      # printing
      if outputlevel ≥ 1 
        @printf("%-4d  ", ep)
        @printf("⟨logP⟩ = %-4.4f  ", train_loss)
        if !isnothing(test_data) 
          @printf("(%.4f)  ", test_loss)
        end
        # TODO: add the trace preserving cost function here for QPT
        !isnothing(observer!) && printobserver(observer!, print_metrics)
        @printf("elapsed = %-4.3fs", ep_time)
        println()
      end
      # saving
      if !isnothing(outputpath)
        observerpath = outputpath * "_observer.jld2"
        save(observerpath, observer!)
        if savestate
          if isqpt
            model_to_be_saved = model isa LPDO{MPS} ? choi_mps_to_unitary_mpo(best_model) : best_model
          else
            model_to_be_saved = model isa LPDO{MPS} ? best_model.X : best_model
          end
          statepath = outputpath * "_state.h5"
          h5rewrite(statepath) do fout
            write(fout, "state", model_to_be_saved)
          end
        end
      end
    end
    !isnothing(observer!) && haskey(observer!.data,"earlystop") && results(observer!, "earlystop")[end] && break
  end
  return best_model
end

# QST
tomography(data::Matrix{Pair{String,Int}}, ψ::MPS; kwargs...) = 
  tomography(data, LPDO(ψ); kwargs...).X

tomography(train_data::Vector{<:Vector{Pair{String,Int}}}, args...; kwargs...) = 
  tomography(permutedims(hcat(train_data...)), args...; kwargs...)


# QPT
tomography(data::Matrix{Pair{String,Pair{String,Int}}}, U::MPO; kwargs...) = 
  choi_mps_to_unitary_mpo(tomography(data, LPDO(unitary_mpo_to_choi_mps(U)); kwargs...))

tomography(train_data::Vector{<:Vector{Pair{String,Pair{String,Int}}}}, args...; kwargs...) = 
  tomography(permutedims(hcat(train_data...)), args...; kwargs...)



"""
EARLY STOPPING FUNCTIONS
"""

#stopif_fidelity(M1, M2; ϵ::Number, kwargs...) 
#  fidelity(M1,M2) ≤ ϵ

function stopif_loss(; loss::Vector, ϵ::Number, min_iter::Number)
  length(loss) < min_iter+1 && return false
  avgloss = StatsBase.mean(loss[end-size:end])
  Δ = StatsBase.sem(historyloss[end-size:end])
  return Δ/avgloss < ϵ
end
