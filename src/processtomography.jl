gradlogZ(C::Choi{LPDO{MPS}}; localnorms = nothing) = 
  gradlogZ(C.M.X; localnorms = localnorms)

gradlogZ(C::Choi{LPDO{MPO}}; sqrt_localnorms = nothing) = 
  gradlogZ(C.M; sqrt_localnorms = sqrt_localnorms)

  
function gradnll(C::Choi{LPDO{MPS}},
                 data_in::Array,
                 data_out::Array;
                 localnorms = nothing)
  ψ = C.M.X
  N = length(ψ)

  s_in  = [firstind(ψ[j],tags="Input") for j in 1:length(ψ)]
  s_out = [firstind(ψ[j],tags="Output") for j in 1:length(ψ)]

  links = [linkind(ψ, n) for n in 1:N-1]

  ElT = eltype(ψ[1])

  nthreads = Threads.nthreads()

  L = [Vector{ITensor{1}}(undef, N) for _ in 1:nthreads]
  Lψ = [Vector{ITensor}(undef, N) for _ in 1:nthreads]

  R = [Vector{ITensor{1}}(undef, N) for _ in 1:nthreads]
  Rψ = [Vector{ITensor}(undef, N) for _ in 1:nthreads]
  
  P = [Vector{ITensor}(undef, N) for _ in 1:nthreads]

  for nthread in 1:nthreads
    for n in 1:N-1
      L[nthread][n] = ITensor(ElT, undef, links[n])
      Lψ[nthread][n] = ITensor(ElT, undef, s_in[n],s_out[n], links[n])
    end
    Lψ[nthread][N] = ITensor(ElT, undef, s_in[N],s_out[N])

    for n in N:-1:2
      R[nthread][n] = ITensor(ElT, undef, links[n-1])
      Rψ[nthread][n] = ITensor(ElT, undef, links[n-1], s_in[n],s_out[n])
    end
    Rψ[nthread][1] = ITensor(ElT, undef, s_in[1],s_out[1])

    for n in 1:N
      P[nthread][n] = ITensor(ElT, undef, s_in[n],s_out[n])
    end
  end

  if isnothing(localnorms)
    localnorms = ones(N)
  end

  ψdag = dag(ψ)

  gradients = [[ITensor(ElT, inds(ψ[j])) for j in 1:N] for _ in 1:nthreads]

  grads = [[ITensor(ElT, undef, inds(ψ[j])) for j in 1:N] for _ in 1:nthreads]

  loss = zeros(nthreads)
 
  Threads.@threads for n in 1:size(data_in)[1]

    nthread = Threads.threadid()

    x_in = data_in[n,:] 
    x_out = data_out[n,:] 
    
    """ LEFT ENVIRONMENTS """
    P[nthread][1] = dag(inputstate(x_in[1],s_in[1])) * inputstate(x_out[1],s_out[1])
    L[nthread][1] .= ψdag[1] .* P[nthread][1]
    for j in 2:N-1
      P[nthread][j] = dag(inputstate(x_in[j],s_in[j])) * inputstate(x_out[j],s_out[j])
      Lψ[nthread][j] .= L[nthread][j-1] .* ψdag[j]
      L[nthread][j] .=  Lψ[nthread][j] .* P[nthread][j]
    end
    P[nthread][N] = dag(inputstate(x_in[N],s_in[N])) * inputstate(x_out[N],s_out[N])
    Lψ[nthread][N] .= L[nthread][N-1] .* ψdag[N]
    ψx = (Lψ[nthread][N] * P[nthread][N])[]
    prob = abs2(ψx)
    loss[nthread] -= log(prob)/size(data_in)[1]
    
    #""" RIGHT ENVIRONMENTS """
    R[nthread][N] .= ψdag[N] .* P[nthread][N]
    for j in reverse(2:N-1)
      Rψ[nthread][j] .= ψdag[j] .* R[nthread][j+1]
      R[nthread][j] .= Rψ[nthread][j] .* P[nthread][j]
    end

    """ GRADIENTS """
    # TODO: fuse into one call to mul!
    grads[nthread][1] .= P[nthread][1] .* R[nthread][2]
    gradients[nthread][1] .+= (1 / (localnorms[1] * ψx)) .* grads[nthread][1]
    for j in 2:N-1
      Rψ[nthread][j] .= L[nthread][j-1] .* P[nthread][j]    
      # TODO: fuse into one call to mul!
      grads[nthread][j] .= Rψ[nthread][j] .* R[nthread][j+1]
      gradients[nthread][j] .+= (1 / (localnorms[j] * ψx)) .* grads[nthread][j]
    end
    grads[nthread][N] .= L[nthread][N-1] .* P[nthread][N]
    gradients[nthread][N] .+= (1 / (localnorms[N] * ψx)) .* grads[nthread][N]
  end
  
  for nthread in 1:nthreads
    for g in gradients[nthread]
      g .= (-2/size(data_in)[1]) .* g
    end
  end

  gradients_tot = [ITensor(ElT, inds(ψ[j])) for j in 1:N]
  loss_tot = 0.0
  for nthread in 1:nthreads
    gradients_tot .+= gradients[nthread]
    loss_tot += loss[nthread]
  end

  return gradients_tot, loss_tot
end



#gradnll(ψ::MPS, data::Array; localnorms = nothing, choi::Bool = false) = 
#  gradnll(LPDO(ψ), data; sqrt_localnorms = localnorms, choi = choi)
#
#"""
#    PastaQ.gradnll(lpdo::LPDO{MPO}, data::Array; sqrt_localnorms = nothing, choi::Bool=false)
#
#Compute the gradients of the cross-entropy between the LPDO probability 
#distribution of the empirical data distribution for a set of projective 
#measurements in different local bases. The probability of a single 
#data-point `σ = (σ₁,σ₂,…)` is :
#
#`P(σ) = ⟨σ|Û ρ Û†|σ⟩ = |⟨σ|Û M M† Û†|σ⟩ = |⟨σ|Û M`   
#
#where `Û` is the depth-1 local circuit implementing the basis rotation.
#The cross entropy function is
#
#`nll ∝ -∑ᵢlog P(σᵢ)`
#
#where `∑ᵢ` runs over the measurement data. Returns the gradients:
#
#`∇ᵢ = - ∂ᵢ⟨log P(σ))⟩_data`
#
#If `choi=true`, the probability is then obtaining by transposing the 
#input state, which is equivalent to take the conjugate of the eigenstate projector.
#"""
#function gradnll(L::LPDO{MPO}, data::Array;
#                 sqrt_localnorms = nothing, choi::Bool = false)
#  lpdo = L.X
#  N = length(lpdo)
#
#  s = firstsiteinds(lpdo)  
#  
#  links = [linkind(lpdo, n) for n in 1:N-1]
#  
#  kraus = Index[]
#  for j in 1:N
#    push!(kraus,firstind(lpdo[j], "Purifier"))
#  end
#
#  ElT = eltype(lpdo[1])
#  
#  nthreads = Threads.nthreads()
#
#  L     = [Vector{ITensor{2}}(undef, N) for _ in 1:nthreads]
#  Llpdo = [Vector{ITensor}(undef, N) for _ in 1:nthreads]
#  Lgrad = [Vector{ITensor}(undef,N) for _ in 1:nthreads]
#
#  R     = [Vector{ITensor{2}}(undef, N) for _ in 1:nthreads]
#  Rlpdo = [Vector{ITensor}(undef, N) for _ in 1:nthreads]
#  
#  Agrad = [Vector{ITensor}(undef, N) for _ in 1:nthreads]
#  
#  T  = [Vector{ITensor}(undef,N) for _ in 1:nthreads] 
#  Tp = [Vector{ITensor}(undef,N) for _ in 1:nthreads]
#  
#  grads     = [Vector{ITensor}(undef,N) for _ in 1:nthreads] 
#  gradients = [Vector{ITensor}(undef,N) for _ in 1:nthreads]
#  
#  for nthread in 1:nthreads
#
#    for n in 1:N-1
#      L[nthread][n] = ITensor(ElT, undef, links[n]',links[n])
#    end
#    for n in 2:N-1
#      Llpdo[nthread][n] = ITensor(ElT, undef, kraus[n],links[n]',links[n-1])
#    end
#    for n in 1:N-2
#      Lgrad[nthread][n] = ITensor(ElT,undef,links[n],kraus[n+1],links[n+1]')
#    end
#    Lgrad[nthread][N-1] = ITensor(ElT,undef,links[N-1],kraus[N])
#
#    for n in N:-1:2
#      R[nthread][n] = ITensor(ElT, undef, links[n-1]',links[n-1])
#    end 
#    for n in N-1:-1:2
#      Rlpdo[nthread][n] = ITensor(ElT, undef, links[n-1]',kraus[n],links[n])
#    end
#  
#    Agrad[nthread][1] = ITensor(ElT, undef, kraus[1],links[1]',s[1])
#    for n in 2:N-1
#      Agrad[nthread][n] = ITensor(ElT, undef, links[n-1],kraus[n],links[n]',s[n])
#    end
#
#    T[nthread][1] = ITensor(ElT, undef, kraus[1],links[1])
#    Tp[nthread][1] = prime(T[nthread][1],"Link")
#    for n in 2:N-1
#      T[nthread][n] = ITensor(ElT, undef, kraus[n],links[n],links[n-1])
#      Tp[nthread][n] = prime(T[nthread][n],"Link")
#    end
#    T[nthread][N] = ITensor(ElT, undef, kraus[N],links[N-1])
#    Tp[nthread][N] = prime(T[nthread][N],"Link")
#  
#    grads[nthread][1] = ITensor(ElT, undef,links[1],kraus[1],s[1])
#    gradients[nthread][1] = ITensor(ElT,links[1],kraus[1],s[1])
#    for n in 2:N-1
#      grads[nthread][n] = ITensor(ElT, undef,links[n],links[n-1],kraus[n],s[n])
#      gradients[nthread][n] = ITensor(ElT,links[n],links[n-1],kraus[n],s[n])
#    end
#    grads[nthread][N] = ITensor(ElT, undef,links[N-1],kraus[N],s[N])
#    gradients[nthread][N] = ITensor(ElT, links[N-1],kraus[N],s[N])
#  end
#  
#  if isnothing(sqrt_localnorms)
#    sqrt_localnorms = ones(N)
#  end
#  
#  loss = zeros(nthreads)
#
#  Threads.@threads for n in 1:size(data)[1]
#
#    nthread = Threads.threadid()
#
#    x = data[n,:]
#    
#    """ LEFT ENVIRONMENTS """
#    if choi
#      T[nthread][1] .= lpdo[1] .* inputstate(x[1],s[1])
#      L[nthread][1] .= prime(T[nthread][1],"Link") .* dag(T[nthread][1])
#    else
#      T[nthread][1] .= lpdo[1] .* dag(inputstate(x[1],s[1]))
#      L[nthread][1] .= prime(T[nthread][1],"Link") .* dag(T[nthread][1])
#    end
#    for j in 2:N-1
#      if isodd(j) & choi
#        T[nthread][j] .= lpdo[j] .* inputstate(x[j],s[j])
#      else
#        T[nthread][j] .= lpdo[j] .* dag(inputstate(x[j],s[j]))
#      end
#      Llpdo[nthread][j] .= prime(T[nthread][j],"Link") .* L[nthread][j-1]
#      L[nthread][j] .= Llpdo[nthread][j] .* dag(T[nthread][j])
#    end
#    T[nthread][N] .= lpdo[N] .* dag(inputstate(x[N],s[N]))
#    prob = L[nthread][N-1] * prime(T[nthread][N],"Link")
#    prob = prob * dag(T[nthread][N])
#    prob = real(prob[])
#    loss[nthread] -= log(prob)/size(data)[1]
#    
#    """ RIGHT ENVIRONMENTS """
#    R[nthread][N] .= prime(T[nthread][N],"Link") .* dag(T[nthread][N])
#    for j in reverse(2:N-1)
#      Rlpdo[nthread][j] .= prime(T[nthread][j],"Link") .* R[nthread][j+1] 
#      R[nthread][j] .= Rlpdo[nthread][j] .* dag(T[nthread][j])
#    end
#    
#    """ GRADIENTS """
#    if choi
#      Tp[nthread][1] .= prime(lpdo[1],"Link") .* inputstate(x[1],s[1])
#      Agrad[nthread][1] .=  Tp[nthread][1] .* dag(inputstate(x[1],s[1]))
#    else
#      Tp[nthread][1] .= prime(lpdo[1],"Link") .* dag(inputstate(x[1],s[1]))
#      Agrad[nthread][1] .=  Tp[nthread][1] .* inputstate(x[1],s[1])
#    end
#    grads[nthread][1] .= R[nthread][2] .* Agrad[nthread][1]
#    gradients[nthread][1] .+= (1 / (sqrt_localnorms[1] * prob)) .* grads[nthread][1]
#    for j in 2:N-1
#      if isodd(j) & choi
#        Tp[nthread][j] .= prime(lpdo[j],"Link") .* inputstate(x[j],s[j])
#        Lgrad[nthread][j-1] .= L[nthread][j-1] .* Tp[nthread][j]
#        Agrad[nthread][j] .= Lgrad[nthread][j-1] .* dag(inputstate(x[j],s[j]))
#      else
#        Tp[nthread][j] .= prime(lpdo[j],"Link") .* dag(inputstate(x[j],s[j]))
#        Lgrad[nthread][j-1] .= L[nthread][j-1] .* Tp[nthread][j]
#        Agrad[nthread][j] .= Lgrad[nthread][j-1] .* inputstate(x[j],s[j])
#      end
#      grads[nthread][j] .= R[nthread][j+1] .* Agrad[nthread][j] 
#      gradients[nthread][j] .+= (1 / (sqrt_localnorms[j] * prob)) .* grads[nthread][j]
#    end
#    Tp[nthread][N] .= prime(lpdo[N],"Link") .* dag(inputstate(x[N],s[N]))
#    Lgrad[nthread][N-1] .= L[nthread][N-1] .* Tp[nthread][N]
#    grads[nthread][N] .= Lgrad[nthread][N-1] .* inputstate(x[N],s[N])
#    gradients[nthread][N] .+= (1 / (sqrt_localnorms[N] * prob)) .* grads[nthread][N]
#  end
#  
#  for nthread in 1:nthreads
#    for g in gradients[nthread]
#      g .= (-2/size(data)[1]) .* g
#    end
#  end
#  
#  gradients_tot = Vector{ITensor}(undef,N) 
#  gradients_tot[1] = ITensor(ElT,links[1],kraus[1],s[1])
#  for n in 2:N-1
#    gradients_tot[n] = ITensor(ElT,links[n],links[n-1],kraus[n],s[n])
#  end
#  gradients_tot[N] = ITensor(ElT, links[N-1],kraus[N],s[N])
#  
#  loss_tot = 0.0
#  for nthread in 1:nthreads
#    gradients_tot .+= gradients[nthread]
#    loss_tot += loss[nthread]
#  end
#  
#  return gradients_tot, loss_tot
#end
#

"""
    PastaQ.gradients(L::LPDO, data::Array; sqrt_localnorms = nothing, choi::Bool = false)
    PastaQ.gradients(ψ::MPS, data::Array; localnorms = nothing, choi::Bool = false)

Compute the gradients of the cost function:
`C = log(Z) - ⟨log P(σ)⟩_data`

If `choi=true`, add the Choi normalization `trace(Λ)=d^N` to the cost function.
"""
function gradients(C::Choi, data_in::Array, data_out::Array;
                   localnorms = nothing)
  g_logZ,logZ = gradlogZ(C; localnorms = localnorms)
  g_nll, nll  = gradnll(C, data_in, data_out; localnorms = localnorms) 
  
  grads = g_logZ + g_nll
  loss = logZ + nll
  loss -= length(C) * log(2) 
  return grads,loss
end

#gradients(ψ::MPS, data::Array; localnorms = nothing, choi::Bool = false) = 
#  gradients(LPDO(ψ), data; sqrt_localnorms = localnorms, choi = choi)
#
#
#"""
#    tomography(data::Array, L::LPDO; optimizer::Optimizer, kwargs...)
#    tomography(data::Array, ψ::MPS; optimizer::Optimizer, kwargs...)
#
#Run quantum state tomography using a the starting state `model` on `data`.
#
## Arguments:
#  - `model`: starting LPDO state.
#  - `data`: training data set of projective measurements.
#  - `batchsize`: number of data-points used to compute one gradient iteration.
#  - `epochs`: total number of full sweeps over the dataset.
#  - `target`: target quantum state underlying the data
#  - `choi`: if true, compute probability using Choi matrix
#  - `observer!`: pass an observer object (like `TomographyObserver()`) to keep track of measurements and fidelities.
#  - `outputpath`: write training metrics on file 
#"""
#function tomography(data::Matrix{Pair{String, Pair{String, Int}}}, L::LPDO;
#                    optimizer::Optimizer,
#                    mixed::Bool=false,
#                    observer! = nothing,
#                    kwargs...)
#  target = get(kwargs,:target,nothing)
#  #mixed::Bool = get(kwargs,:mixed,false)
#
#  optimizer = copy(optimizer)
#  #
#  # TEMPORARY WRAPPER FOR UNSPLIT PROCESS TOMOGRAPHY
#  #
#  # This function take a model `L` and a `target` (is provided) in a unsplit
#  # representation, and run tomography with the split algorithm. Returns the unsplit result.
#  #
#
#  # Target LPDO are currently not supported
#  if target isa Choi{MPO}
#    target = target.M
#  elseif target isa Choi{LPDO{MPO}}
#    target = target.M.X
#  end
#  
#  @assert (target isa MPS) || (target isa MPO)
#  
#  if !mixed
#    #
#    # 1. Noiseless channel (unitary circuit)
#    #
#
#    # Split the variational state: MPO -> MPS (x2 sites)
#    U = L.X
#    model = LPDO(splitunitary(U))
#
#    # Split the target state: MPO -> MPS (x2 sites)
#    target = splitunitary(target)
#
#    # Run process tomography
#    V = _tomography(data, model;
#                    optimizer = optimizer,
#                    observer! = observer!,
#                    kwargs...,
#                    target = target)
#
#    # Unsplit the learned state: MPS -> MPO (÷2 sites) 
#    return unsplitunitary(V.X)
#  else
#    #
#    # 2. Noisy channel (choi matrix)
#    #
#
#    # Split the target choi matrix: MPO -> MPS (x2 sites)
#    target = splitchoi(target).M
#    model = splitchoi(Choi(L))
#
#    # Run process tomography
#    Λ = _tomography(data, model.M;
#                    optimizer = optimizer,
#                    observer! = observer!,
#                    kwargs...,
#                    target = target)
#
#    # Split the target choi matrix: MPO -> MPS (x2 sites)
#    return unsplitchoi(Choi(Λ))
#  end
#end
#
#function tomography(data::Matrix{Pair{String, Pair{String, Int}}}, U::MPO;
#                    optimizer::Optimizer, mixed::Bool=false, kwargs...) 
#  return tomography(data, LPDO(U); optimizer = optimizer, mixed = mixed, kwargs...)
#end
#
#function tomography(data::Matrix{Pair{String, Pair{String, Int}}}, C::Choi;
#                    optimizer::Optimizer, mixed::Bool=true, kwargs...)
#  return tomography(data, C.M; optimizer = optimizer, mixed = mixed,  kwargs...)
#end
#
#function _tomography(data::Matrix{Pair{String, Int}}, L::LPDO;
#                     optimizer::Optimizer,
#                     observer! = nothing,
#                     kwargs...)
#  # Read arguments
#  use_localnorm::Bool = get(kwargs,:use_localnorm,true)
#  use_globalnorm::Bool = get(kwargs,:use_globalnorm,false)
#  batchsize::Int64 = get(kwargs,:batchsize,500)
#  epochs::Int64 = get(kwargs,:epochs,1000)
#  target = get(kwargs,:target,nothing)
#  choi::Bool = get(kwargs,:choi,false)
#  outputpath = get(kwargs,:fout,nothing)
#
#  optimizer = copy(optimizer)
#
#  if use_localnorm && use_globalnorm
#    error("Both use_localnorm and use_globalnorm are set to true, cannot use both local norm and global norm.")
#  end
#  
#  # Convert data to projectors
#  #data = "state" .* data
#  data = convertdatapoints(data; state = true)
#
#  model = copy(L)
#  F = nothing
#  Fbound = nothing
#  frob_dist = nothing
#  
#  if batchsize > size(data)[1]
#    error("Batch size larger than dataset size")
#  end
#
#  # Number of training batches
#  num_batches = Int(floor(size(data)[1]/batchsize))
#  
#  tot_time = 0.0
#  # Training iterations
#  for ep in 1:epochs
#    ep_time = @elapsed begin
#  
#    data = data[shuffle(1:end),:]
#    
#    avg_loss = 0.0
#
#    # Sweep over the data set
#    for b in 1:num_batches
#      batch = data[(b-1)*batchsize+1:b*batchsize,:]
#      # Local normalization
#      if use_localnorm
#        modelcopy = copy(model)
#        sqrt_localnorms = []
#        normalize!(modelcopy; sqrt_localnorms! = sqrt_localnorms)
#        grads,loss = gradients(modelcopy, batch, sqrt_localnorms = sqrt_localnorms, choi = choi)
#      # Global normalization
#      elseif use_globalnorm
#        normalize!(model)
#        grads,loss = gradients(model,batch,choi=choi)
#      # Unnormalized
#      else
#        grads,loss = gradients(model,batch,choi=choi)
#      end
#
#      nupdate = ep * num_batches + b
#      avg_loss += loss/Float64(num_batches)
#      update!(model,grads,optimizer;step=nupdate)
#    end
#    end # end @elapsed
#    
#    print("Ep = $ep  ")
#    @printf("Loss = %.5E  ",avg_loss)
#    if !isnothing(target)
#      if ((model.X isa MPO) & (target isa MPO)) 
#        frob_dist = frobenius_distance(model,target)
#        Fbound = fidelity_bound(model,target)
#        @printf("Trace distance = %.3E  ",frob_dist)
#        @printf("Fidelity bound = %.3E  ",Fbound)
#        if (length(model) <= 8)
#          disable_warn_order!()
#          F = fullfidelity(model,target)
#          reset_warn_order!()
#          @printf("Fidelity = %.3E  ",F)
#        end
#      else
#        F = fidelity(model,target)
#        @printf("Fidelity = %.3E  ",F)
#      end
#    end
#    @printf("Time = %.3f sec",ep_time)
#    print("\n")
#
#    # Measure
#    if !isnothing(observer!)
#      measure!(observer!;
#               NLL = avg_loss,
#               F = F,
#               Fbound = Fbound,
#               frob_dist = frob_dist)
#      # Save on file
#      if !isnothing(outputpath)
#        saveobserver(observer, outputpath; M = model)
#      end
#    end
#    
#
#    tot_time += ep_time
#  end
#  @printf("Total Time = %.3f sec\n",tot_time)
#  normalize!(model)
#
#  return model
#end
#
#_tomography(data::Matrix{Pair{String, Int}}, C::Choi; optimizer::Optimizer, mixed::Bool=false, kwargs...) =
# _tomography(data, C.M; optimizer = optimizer, mixed = mixed, kwargs...)
#
#_tomography(data::Matrix{Pair{String, Int}}, ψ::MPS; optimizer::Optimizer, mixed::Bool, kwargs...) =
#  _tomography(data, LPDO(ψ); optimizer = optimizer, mixed = mixed, kwargs...)
#
#
##Run quantum process tomography on measurement data `data` using `model` as s variational ansatz.
##
##The data is reshuffled so it takes the format: `(input1,output1,input2,output2,…)`.
#function _tomography(data::Matrix{Pair{String, Pair{String, Int}}},
#                     L::LPDO;
#                     optimizer::Optimizer,
#                     kwargs...)
#  N = size(data, 2)
#  nsamples = size(data, 1)
#  inputs0 = first.(data)
#  bases = first.(last.(data))
#  outcomes = last.(last.(data))
#  data_combined = Matrix{Pair{String, Int}}(undef, nsamples, 2*N)
#  for n in 1:nsamples
#    inputstate = convertdatapoint(inputs0[n,:])
#    for j in 1:N
#      data_combined[n, 2*j-1] = inputstate[j] 
#      data_combined[n, 2*j] = last(data[n, j])
#    end
#  end
#  return _tomography(data_combined, L;
#                     optimizer = optimizer,
#                     choi = true,
#                     kwargs...)
#end
#
function nll(C::Choi{LPDO{MPS}}, data_in::Array, data_out::Array)
  ψ = C.M.X
  N = length(ψ)
  @assert N==size(data_in)[2]
  @assert N==size(data_out)[2]
  @assert size(data_in)[1] == size(data_out)[1]
  loss = 0.0
  s_in  = [firstind(ψ[j],tags="Input") for j in 1:length(ψ)]
  s_out = [firstind(ψ[j],tags="Output") for j in 1:length(ψ)]
  
  for n in 1:size(data_in)[1]
    x_in  = data_in[n,:]
    x_out = data_out[n,:]

    ψx = dag(ψ[1]) * dag(inputstate(x_in[1],s_in[1]))
    ψx = ψx * inputstate(x_out[1],s_out[1])
    #ψx = dag(ψ[1]) * dag(inputstate(x[1],s_in[1]))
    #ψx = (choi ? dag(ψ[1]) * dag(inputstate(x[1],s[1])) :
    #             dag(ψ[1]) * inputstate(x[1],s[1]))
    for j in 2:N
      ψ_r = dag(ψ[j]) * dag(inputstate(x_in[j],s_in[j]))
      ψ_r = ψ_r *inputstate(x_out[j],s_out[j])
      #ψ_r = (isodd(j) & choi ? ψ_r = dag(ψ[j]) * dag(inputstate(x[j],s[j])) :
      #                         ψ_r = dag(ψ[j]) * inputstate(x[j],s[j]))
      ψx = ψx * ψ_r
    end
    prob = abs2(ψx[])
    loss -= log(prob)/size(data_in)[1]
  end
  return loss
end

#nll(ψ::MPS, args...; kwargs...) = nll(LPDO(ψ), args...; kwargs...)
#
#"""
#    PastaQ.nll(lpdo::LPDO, data::Array; choi::Bool = false)
#
#Compute the negative log-likelihood using an LPDO ansatz
#over a dataset `data`:
#
#`nll ∝ -∑ᵢlog P(σᵢ)`
#
#If `choi=true`, the probability is then obtaining by transposing the 
#input state, which is equivalent to take the conjugate of the eigenstate projector.
#"""
#function nll(L::LPDO{MPO}, data::Array; choi::Bool = false)
#  lpdo = L.X
#  N = length(lpdo)
#  loss = 0.0
#  s = firstsiteinds(lpdo)
#  for n in 1:size(data)[1]
#    x = data[n,:]
#
#    # Project LPDO into the measurement eigenstates
#    Φdag = dag(copy(lpdo))
#    for j in 1:N
#      Φdag[j] = (isodd(j) & choi ? Φdag[j] = Φdag[j] * dag(inputstate(x[j],s[j])) :
#                                   Φdag[j] = Φdag[j] * inputstate(x[j],s[j]))
#    end
#    
#    # Compute overlap
#    prob = inner(Φdag,Φdag)
#    loss -= log(real(prob))/size(data)[1]
#  end
#  return loss
#end
#
