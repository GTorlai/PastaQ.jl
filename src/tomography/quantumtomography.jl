@doc raw"""
    tomography(data::Matrix{Pair{String, Int}}, sites::Vector{<:Index};
               method::String = "linear_inversion",
               fillzeros::Bool = true,
               trρ::Number = 1.0,
               max_iters::Int = 10000,
               kwargs...)

    tomography(data::Matrix{Pair{String,Pair{String, Int}}}, sites::Vector{<:Index};
               method::String="linear_inversion", kwargs...)

Run full quantum tomography for a set of input measurement data `data`. If the input data
consists of a list of `Pair{String, Int}`, it is interpreted as quantum state tomography. Each
data point is a single measurement outcome, e.g. `"X" => 1` to refer to a measurement in the `X` basis
with binary outcome `1`. If instead the input data is a collection of `Pair{String,Pair{String, Int}}`,
it is interpreted as quantum process tomography, with each data-point corresponding to having input a
given state to the channel, followed by a measurement in a basis, e.g.  `"X+" => ("Z" => 0)` referring to an
input ``|+\rangle`` state, followed by a measurement in the `Z` basis with outcome `0`.

There are three methods to perform tomography (we show state tomography here as an example):
1. `method = "linear_inversion"` (or `"LI"`): optimize a variational density matrix ``\rho`` (or Choi matrix)1


2. `method = "least_squares"` (or `"LS"`):

3. `method = "maximum_likelihood"` (or `"ML"`):
"""
function tomography(
  data::Matrix{Pair{String,Int}};
  method::String="linear_inversion",
  fillzeros::Bool=true,
  kwargs...,
)
  return tomography(
    empirical_probabilities(data; fillzeros=fillzeros),
    siteinds("Qubit", size(data, 2));
    method=method,
    kwargs...,
  )
end

function tomography(
  data::Matrix{Pair{String,Int}},
  sites::Vector{<:Index};
  method::String="linear_inversion",
  fillzeros::Bool=true,
  kwargs...,
)
  return tomography(
    empirical_probabilities(data; fillzeros=fillzeros), sites; method=method, kwargs...
  )
end

function tomography(data::Matrix{Pair{String,Pair{String,Int}}}; kwargs...)
  return tomography(data, siteinds("Qubit", size(data, 2)); kwargs...)
end

function tomography(
  data::Matrix{Pair{String,Pair{String,Int}}},
  sites::Vector{<:Index};
  method::String="linear_inversion",
  fillzeros::Bool=true,
  kwargs...,
)
  sites_in = addtags.(sites, "Input")
  sites_out = addtags.(sites, "Output")
  process_sites = Index[]
  for j in 1:size(data, 2)
    push!(process_sites, sites_in[j])
    push!(process_sites, sites_out[j])
  end
  return tomography(
    empirical_probabilities(data; fillzeros=fillzeros),
    process_sites;
    method=method,
    process=true,
    kwargs...,
  )
end

@doc raw"""
    tomography(train_data::AbstractMatrix, L::LPDO;
               optimizer::Optimizer,
               observer! = nothing,
               kwargs...)

Run quantum process tomography using a variational model ``L`` to fit `train_data`. Depending
on the type of `train_data`, run state or process tomography (see full tomography docs for the data format).

For quantum state tomography, optimize the average Kullbach-Leibler (KL) divergence for projective
measurements in a set of bases:

```math
C(\theta) = -\frac{1}{|D|}\sum_{k=1}^{|D|} \log P(x_k^{(b)})
```
where the cost function is computed as
```math
C(\theta) = -\frac{1}{|D|}\sum_{k=1}^{|D|} \log |\langle x_k|U_b|\psi(\theta)\rangle|^2
```
for input MPS variational wavefunction, and
```math
C(\theta) = -\frac{1}{|D|}\sum_{k=1}^{|D|} \log \langle x_k|U_b \rho(\theta) U_b^\dagger|x_k\rangle
```
for input LPDO variational density operators. Here ``U_b`` is the depth-1 unitary that rotates
the variational state into the measurement basis ``b`` for outcome ``x^{(b)}``.


For quantum process tomography, optimize the  KL divergence for the process probability distribution
```math
C(\theta) = -\frac{1}{|D|}\sum_{k=1}^{|D|} \log P(x_k^{(b)}|\xi)
```
where ``\xi`` is the input state to the channel. The cost function is computed as
```math
C(\theta) = -\frac{1}{|D|}\sum_{k=1}^{|D|} \log |\langle x_k|U_b|\tilde\Phi(\theta)\rangle|^2
```
for input MPO variational unitary operator (trated as a MPS ``|\Phi\rangle`` after appropriate
vectorization), and
```math
C(\theta) = -\frac{1}{|D|}\sum_{k=1}^{|D|} \log \langle x|U_b \tilde\Lambda(\theta) U_b^\dagger|x\rangle
```
for input LPDO variational Choi matrix. Here we refer to ``\tilde\Phi`` and ``\tilde\Lambda`` respectively as the
projection of the unitary operator or Choi matrix into the input state ``|\xi\rangle`` to the channel.

Keyword arguments:
- `train_data`: pairs of preparation/ (basis/outcome): `("X+"=>"X"=>0, "Z-"=>"Y"=>1, "Y+"=>"Z"=>0, …)`.
- `L`: variational model (MPO/LPDO).
- `optimizer`: optimizer object for stochastic optimization (from `Optimisers.jl`).
- `observer!`: if provided, keep track of training metrics (from `Observers.jl`).
- `batch_size`: number of samples used for one gradient update.
- `epochs`: number of training iterations.
- `target`: target quantum state/process for distance measures (e.g. fidelity).
- `test_data`: data for computing cross-validation.
- `observer_step = 1`: how often the Observer is called to record measurements.
- `outputlevel = 1`: amount of information printed on screen during training.
- `outputpath`: if provided, save metrics on file.
- `savestate`: if `true`, save the variational state on file.
- `print_metrics = []`: print these metrics on screen during training.
- `earlystop`: if `true`, use pre-defined early-stop function. A function can also be passed
   as `earlystop`, in which case if the function (evaluated at each iteration) returns `true`,
   the training is halted.
"""
function tomography(train_data::AbstractMatrix, L::LPDO; (observer!)=nothing, kwargs...)
  # Read arguments
  opt = get(kwargs, :optimizer, Optimisers.Descent(0.01))
  batchsize::Int64 = get(kwargs, :batchsize, 100)
  epochs::Int64 = get(kwargs, :epochs, 1000)
  trace_preserving_regularizer = get(kwargs, :trace_preserving_regularizer, 0.0)
  observe_step::Int64 = get(kwargs, :observe_step, 1)
  test_data = get(kwargs, :test_data, nothing)
  print_metrics = get(kwargs, :print_metrics, [])
  outputpath = get(kwargs, :outputpath, nothing)
  outputlevel = get(kwargs, :outputlevel, 1)
  savestate = get(kwargs, :savestate, false)
  earlystop = get(kwargs, :earlystop, false)

  model = copy(L)
  isqpt = train_data isa Matrix{Pair{String,Pair{String,Int}}}
  localnorm = isqpt ? 2.0 : 1.0

  # observer is not passed but earlystop is called
  observer! = (isnothing(observer!) || earlystop) ? observer() : observer!

  # observer is defined
  if !isnothing(observer!)
    insert_function!(observer!, "train_loss" => identity)
    if !isnothing(test_data)
      insert_function!(observer!, "test_loss" => identity)
    end
    # add the standard early stop function to the observer
    if earlystop
      function stop_if(; loss::Vector)
        return stoptomography_ifloss(; loss=loss, ϵ=1e-3, min_iter=50, window=50)
      end
      insert_function!(observer!, "earlystop" => stop_if)
    end
  end

  # initialize optimizer
  optimizer = setup(opt, model)

  @assert size(train_data, 2) == length(model)
  !isnothing(test_data) && @assert size(test_data)[2] == length(model)

  batchsize = min(size(train_data)[1], batchsize)
  num_batches = Int(floor(size(train_data)[1] / batchsize))

  tot_time = 0.0
  observe_time = 0.0
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
        normalize!(
          normalized_model; (sqrt_localnorms!)=sqrt_localnorms, localnorm=localnorm
        )

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
    ## TODO: Replace this with `update!`.
    !isnothing(observer!) && push!(observer![!, "train_loss"], train_loss)
    observe_time += ep_time

    # measurement stage
    if ep % observe_step == 0
      normalized_model = copy(model)
      sqrt_localnorms = []
      normalize!(normalized_model; (sqrt_localnorms!)=sqrt_localnorms, localnorm=localnorm)

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
        loss = (
          if !isnothing(test_data)
            observer![!, "test_loss"]
          else
            observer![!, "train_loss"]
          end
        )

        model_to_observe = (
          if isqpt && (normalized_model isa LPDO{MPS})
            choi_mps_to_unitary_mpo(normalized_model)
          elseif !isqpt && (normalized_model isa LPDO{MPS})
            normalized_model.X
          else
            normalized_model
          end
        )
        update!(
          observer!, model_to_observe; train_loss=train_loss, test_loss=test_loss, loss=loss
        )
        tot_time += observe_time
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
        @printf("elapsed = %-4.3fs", observe_time)
        observe_time = 0.0
        println()
      end
      # saving
      if !isnothing(outputpath)
        observerpath = outputpath * "_observer.jld2"
        save(observerpath; observer!)
        if savestate
          if isqpt
            model_to_be_saved =
              model isa LPDO{MPS} ? choi_mps_to_unitary_mpo(best_model) : best_model
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
    !isnothing(observer!) &&
      hasproperty(observer!, "earlystop") &&
      observer![end, "earlystop"] &&
      break
  end
  return best_model
end

# QST
function tomography(data::Matrix{Pair{String,Int}}, ψ::MPS; kwargs...)
  return tomography(data, LPDO(ψ); kwargs...).X
end

function tomography(train_data::Vector{<:Vector{Pair{String,Int}}}, args...; kwargs...)
  return tomography(permutedims(hcat(train_data...)), args...; kwargs...)
end

# QPT
function tomography(data::Matrix{Pair{String,Pair{String,Int}}}, U::MPO; kwargs...)
  return choi_mps_to_unitary_mpo(
    tomography(data, LPDO(unitary_mpo_to_choi_mps(U)); kwargs...)
  )
end

function tomography(
  train_data::Vector{<:Vector{Pair{String,Pair{String,Int}}}}, args...; kwargs...
)
  return tomography(permutedims(hcat(train_data...)), args...; kwargs...)
end

"""
EARLY STOPPING FUNCTIONS
"""

#stopif_fidelity(M1, M2; ϵ::Number, kwargs...)
#  fidelity(M1,M2) ≤ ϵ

function stoptomography_ifloss(; loss::Vector, ϵ::Number, min_iter::Number, window::Number)
  length(loss) < min_iter + 1 && return false
  length(loss) < window && return false
  avgloss = StatsBase.mean(loss[(end - window):end])
  Δ = StatsBase.sem(loss[(end - window):end])
  return Δ / avgloss < ϵ
end
