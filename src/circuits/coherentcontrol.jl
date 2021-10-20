
"""
    control_fourierseries(t::Number, θ::Vector;
                          amplitude::Number = 1.0,
                          maxfrequency::Number = 1.0,
                          nharmonics::Number = 1)

"""
function control_fourierseries(t::Number, θ::Vector;
                               amplitude::Number = 1.0,
                               maxfrequency::Number = 1.0)
  
  nharmonics = floor(length(θ))÷2
  f = isodd(length(θ)) ? 0.5 * θ[end] : 0.0
  
  for n in 1:length(θ)÷2
    ω = 2π * n * (maxfrequency / nharmonics)
    f += θ[2*n-1] * cos(ω*t) + θ[2*n] * sin(ω*t)
  end
  return amplitude * tanh(f)
end

function control_sweep(t::Number, θ::Vector)
  @assert length(θ) == 6
  offset, amplitude = θ[1],θ[2]/2
  ton,τon,toff,τoff = θ[3:end]
  f = offset + amplitude * (tanh((t-ton)/τon) - tanh((t-toff)/τoff))
  return f
end



function _drivinghamiltonian(H₀::OpSum, drives::Vector{<:Pair}, t::Float64; kwargs...)
  H = copy(H₀)
  for drive in drives
    drive_function = first(drive)
    if drive_function isa Function
      ft = drive_function(t)
    else
      @assert length(drive_function) > 1
      f, θ... = drive_function
      ft = f(t,θ...)
    end
    μs = last(drive)
    μs = μs isa Tuple ? [μs] : μs 
    for μ in μs 
      driveop, support = μ 
      H += ft, driveop, support, (∇ = true,)
    end
  end
  return H
  # TODO merge the terms once the new gate interface is operational.
  #return ITensors.sortmergeterms!(H)
end

_drivinghamiltonian(H₀::OpSum, drive::Pair, args...; kwargs...) =  
  _drivinghamiltonian(H₀, [drive], args...; kwargs...)

"""
    gradients(ψ₀::Vector{MPS},ψtarget::Vector{MPS},circuit::Vector{<:Vector{<:Any}},
              drives::Union{Pair, Vector},δt::Number, cmap::Vector; kwargs...)
Compute the gradients for the optimization in optimal coherent control. 
Given a set of input states {ψ₀} and a set of desired target states {ψtarget}, the
goal is to discover a parametric drive that realize such dynamics at a fixed time T.
"""
function gradients(ψs₀::Vector{MPS}, 
                   ϕs₀::Vector{MPS},
                   circuit::Vector{<:Vector{<:Any}},
                   drives::Union{Pair,Vector{<:Pair}}, 
                   ts::Vector, 
                   cmap::Vector; 
                   kwargs...)
 
  ψs = copy(ψs₀)
  ϕs = copy(ϕs₀)

  @assert length(ψs) == length(ϕs) 
  
  nthreads = Threads.nthreads()
  
  d = length(ψs)
  
  drives = drives isa Pair ? [drives] : drives
  
  dagcircuit = dag(circuit)
  
  Odag = 0.0
  ∇C = [[zeros(Complex, length(first(drives[k])[2])) for k in 1:length(drives)] for _ in 1:nthreads]
  
  Threads.@threads for j in 1:d
    @assert siteinds(ψs[j]) == siteinds(ϕs[j])
    nthread = Threads.threadid()

    ψL = runcircuit(ϕs[j], dagcircuit; kwargs...)
    ψR = copy(ψs[j])
    
    Odag += conj(inner(ψL,ψR)) / d
    # loop over circuit layers
    for m in 1:length(circuit)
      t = ts[m]
      δt = ts[m+1]-ts[m]
      layer = circuit[m]
      
      # loop over parametric driving gates on the layer
      gcnt = 1
      for g in cmap[m]
        driveop, support, pars = layer[g]
        if !((m==1) && g == 1)
          # here I assume that the gate is on the form exp(im * operator)
          ψR = runcircuit(ψR, layer[gcnt:g]; kwargs...)
          ψL = runcircuit(ψL, layer[gcnt:g]; kwargs...) 
          gcnt = g+1
        end

        for (i,drive) in enumerate(drives)
          μs = last(drive)
          μs = μs isa Tuple ? [μs] : μs
          modeinds = findall(x -> x == (driveop, support), μs)
          for j in modeinds
            f, θ = first(drive)
            grads = Zygote.gradient(Zygote.Params([θ])) do
              f(t, θ)
            end
            # TODO remove the exponent from the params
            Ψ = runcircuit(ψR, (driveop, support); kwargs...)
            ∇F_wrt_T = inner(ψL, Ψ)
            ∇C[nthread][i] += δt * ∇F_wrt_T * grads[θ] / d
          end
        end 
      end
      ψR = runcircuit(ψR, layer[gcnt:end]; kwargs...)
      ψL = runcircuit(ψL, layer[gcnt:end]; kwargs...) 
    end
  end
  ∇tot = [zeros(Complex, length(first(drives[k])[2])) for k in 1:length(drives)]
  for nthread in 1:nthreads
    ∇tot += ∇C[nthread]
  end
  F = real(Odag * conj(Odag))
  return F, imag.(Odag .* ∇tot)
end

gradients(ψ₀::MPS, ψtarget::MPS, args...; kwargs...) = 
  gradients([ψ₀], [ψtarget], args...; kwargs...)



"""
maximmize fidelity
"""
function optimize!(drives0::Vector{<:Pair}, 
                   H::OpSum,
                   ψs::Vector{MPS},
                   ϕs::Vector{MPS};
                   (observer!) = nothing,
                   kwargs...)
  
  ts          = get(kwargs, :ts, nothing)
  δt          = get(kwargs, :δt, nothing)
  T           = get(kwargs, :T, nothing)
  optimizer   = get(kwargs, :optimizer, Optimisers.Descent(0.01))
  epochs      = get(kwargs, :epochs, 100)
  maxdim      = get(kwargs, :maxdim, 10_000)
  cutoff      = get(kwargs, :cutoff, 1E-15)
  outputpath  = get(kwargs, :outputpath, nothing)
  outputlevel = get(kwargs, :outputlevel, 1)
  #earlystop  = get(kwargs, :earlystop, false)
  print_metrics = get(kwargs, :print_metrics, [])
 
  drives = deepcopy(drives0)
  
  # set the time sequence
  if isnothing(ts)
    (isnothing(δt) || isnothing(T)) && error("Time sequence not defined")
    ts = 0.0:δt:T
  end
  ts = collect(ts) 
  
  if !isnothing(observer!)
    observer!["drives"] = nothing
    push!(last(observer!["drives"]), [])
    observer!["loss"]   = nothing
    observer!["∇avg"]   = nothing
  end

  Hts = [_drivinghamiltonian(H, drives, t) for t in ts]
  circuit = trottercircuit(Hts; ts =  ts)
  cmap = circuitmap(circuit)
  
  θ = [last(first(drive)) for drive in drives]
  st = Optimisers.state(optimizer, θ)
  
  tot_time = 0.0
  for ep in 1:epochs
    ep_time = @elapsed begin
      Hts = [_drivinghamiltonian(H, drives, t) for t in ts]
      circuit = trottercircuit(Hts; ts =  ts)
      
      # evolve layer by layer and record infidelity estimator
      F, ∇ = gradients(ψs, ϕs, circuit, drives, ts, cmap; cutoff = cutoff, maxdim = maxdim)
      
      θ = [last(first(drive)) for drive in drives]
      st, θ′ = Optimisers.update(optimizer, st, θ, -∇)
      drives = [(first(first(drives[k])), θ′[k]) => last(drives[k]) for k in 1:length(drives)]
    end
    ∇avg = StatsBase.mean(abs.(vcat(∇...)))
    
    if !isnothing(observer!)
      last(observer!["drives"])[1] = drives
      push!(last(observer!["loss"]), 1-F)
      push!(last(observer!["∇avg"]), ∇avg)
      update!(observer!, ψs, ϕs)
    end
    
    if !isnothing(outputpath)
      observerpath = outputpath * "_observer.jld2"
      save(observerpath, observer!)
    end

    if outputlevel > 0
      @printf("iter = %d  infidelity = %.5E  ", ep, 1 - F); flush(stdout)
      @printf("⟨∇⟩ = %.3E  ", ∇avg); flush(stdout)
      printobserver(observer!, print_metrics)
      @printf(" elapsed = %.3f",ep_time); flush(stdout)
      println()
    end
  end
  drives0[:] = drives
  return drives0
end

optimize!(drives::Union{Pair,Vector{<:Pair}}, H::OpSum, ψ::MPS, ϕ::MPS) = 
  optimize!(drives, H, [ψ], [ϕ])

optimize!(drive::Pair, H, ψ, ϕ) = 
  optimize!([drive], H, ψ, ϕ)

