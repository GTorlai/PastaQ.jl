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
function gradients(ψs::Vector{MPS}, 
                   ϕs::Vector{MPS},
                   circuit::Vector{<:Vector{<:Any}},
                   drives::Union{Pair, Vector}, 
                   ts::Vector, 
                   cmap::Vector; 
                   kwargs...)
 
  @assert length(ψs) == length(ϕs) 
  nthreads = Threads.nthreads()
  depth = length(circuit)
  d = length(ψs)
  
  drives = drives isa Pair ? [drives] : drives
  
  dagcircuit = dag(circuit)
  Odag = 0.0
  for j in 1:d
    @assert siteinds(ψs[j]) == siteinds(ϕs[j])
    ϕ = runcircuit(ϕs[j], dagcircuit; kwargs...)
    Odag += conj(inner(ϕ, ψs[j])) / d
  end
  
  F = Odag * conj(Odag)
  ∇C = [[zeros(Complex, length(first(drives[k])[2])) for k in 1:length(drives)] for _ in 1:nthreads]
  
  Threads.@threads for j in 1:d
    nthread = Threads.threadid()

    ψL = runcircuit(ϕs[j], dagcircuit; kwargs...)
    ψR = copy(ψs[j])
    # loop over circuit layers
    for m in 1:depth
      t = ts[m]
      δt = ts[m+1]-ts[m]
      layer = circuit[m]
      
      # loop over parametric driving gates on the layer
      gcnt = 1
      for g in cmap[m]
        driveop, support, pars = layer[g]
        if !((m==1) && g == 1)
          ψR = runcircuit(ψR, layer[gcnt:g]; kwargs...)
          ψL = runcircuit(ψL, layer[gcnt:g]; kwargs...) 
          gcnt = g+1
        end
        for (i,drive) in enumerate(drives)
          μs = last(drive)
          μs = μs isa Tuple ? [μs] : μs 
          for μ in μs 
            if μ == (driveop, support)
              f, θ = first(drive)
              grads = Zygote.gradient(Zygote.Params([θ])) do
                f(t, θ)
              end
              # TODO remove the exponent from the params
              Ψ = runcircuit(ψR, (driveop, support); kwargs...)
              ∇F_wrt_T = inner(ψL, Ψ)
              ∇C[nthread][i] += δt * ∇F_wrt_T * Odag * grads[θ] / d
            end
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
  return Odag * conj(Odag), imag.(∇tot)
end

gradients(ψ₀::MPS, ψtarget::MPS, args...; kwargs...) = 
  gradients([ψ₀], [ψtarget], args...; kwargs...)



#"""
#maximmize fidelity
#"""
#function maximize!(θ₀::Vector{<:Number},
#                   makecircuit::Function,
#                   drives::Union{Pair,Vector{<:Pair}},
#                   ψ₀::Union{MPS,Vector{MPS}}, ψtarget::Union{MPS,Vector{MPS}},
#                   δt::Number;
#                   optimizer = Optimisers.Descent(0.01),
#                   epochs::Int = 100,
#                   maxdim::Int64 = 10_000,
#                   cutoff::Float64 = 1E-12,)
#  
#  # identity trainable parameters
#  circuit = makecircuit(θ₀, drives)
#  cmap = circuitmap(circuit) 
#  
#  st = Optimisers.state(optimizer, θ₀)
#  θ = copy(θ₀)
#  for ep in 1:epochs
#    circuit = makecircuit(θ, drives) 
#    F, ∇ = gradients(θ, ψ₀, ψtarget, circuit, drives, δt, cmap; maxdim = maxdim, cutoff = cutoff)
#    ∇avg = StatsBase.mean(abs.(∇))
#    #Optimise.update!(optimizer, θ, -∇)
#    st, θ′ = Optimisers.update!(optimizer, st, θ, ∇)
#    @printf("iter = %d  F = %.5E  ⟨∇⟩ = %.3E ",ep,real(F), ∇avg)
#    println()
#  end
#end
#



