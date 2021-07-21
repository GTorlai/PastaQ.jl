"""
    nll(L::LPDO{MPS},data::Matrix{Pair{String,Pair{String, Int}}}) 

Compute the negative log-likelihood of process data using a MPO ansatz
for the vectorized unitary operator.
"""
function nll(L::LPDO{MPS}, data::Matrix{Pair{String,Pair{String,Int}}})
  data_in = first.(data)
  data_out = convertdatapoints(last.(data))

  ψ = L.X
  N = length(ψ)
  loss = 0.0
  s_in = [firstind(ψ[j]; tags="Input") for j in 1:length(ψ)]
  s_out = [firstind(ψ[j]; tags="Output") for j in 1:length(ψ)]

  for n in 1:size(data_in)[1]
    x_in = data_in[n, :]
    x_out = data_out[n, :]

    ψx = dag(ψ[1]) * dag(state(x_in[1], s_in[1]))
    ψx = ψx * state(x_out[1], s_out[1])
    for j in 2:N
      ψ_r = dag(ψ[j]) * dag(state(x_in[j], s_in[j]))
      ψ_r = ψ_r * state(x_out[j], s_out[j])
      ψx = ψx * ψ_r
    end
    prob = abs2(ψx[])
    loss -= log(prob) / size(data_in)[1]
  end
  return loss
end

nll(ψ::MPS, data::Matrix{Pair{String,Pair{String,Int}}}) = nll(LPDO(ψ), data)

"""
    nll(L::LPDO{MPO},data::Matrix{Pair{String,Pair{String, Int}}})

Compute the negative log-likelihood of process data using a LPDO ansatz
for the Choi matrix.
"""
function nll(L::LPDO{MPO}, data::Matrix{Pair{String,Pair{String,Int}}})
  # if the MPO in LPDO{MPO} is a unitary MPO (instead of a MPO with Kraus index)
  # then transform the MPO to MPS and run the nll on that
  #if !hastags(L.X[1], default_purifier_tags)
  #  return nll(_UnitaryMPOtoMPS(copy(L.X)), data)
  #end

  data_in = first.(data)
  data_out = convertdatapoints(last.(data))

  ρ = L.X
  N = length(ρ)
  loss = 0.0
  s_in = [firstind(ρ[j]; tags="Input") for j in 1:N]
  s_out = [firstind(ρ[j]; tags="Output") for j in 1:N]

  for n in 1:size(data_in)[1]
    x_in = data_in[n, :]
    x_out = data_out[n, :]
    ρdag = dag(copy(ρ))
    for j in 1:N
      ρdag[j] = ρdag[j] * dag(state(x_in[j], s_in[j]))
      ρdag[j] = ρdag[j] * state(x_out[j], s_out[j])
    end
    prob = inner(ρdag, ρdag)
    loss -= log(real(prob)) / size(data_in)[1]
  end
  return loss
end

"""
    TP(L::LPDO)

Γ = 1/√D * √(Tr[Φ²] - 2*Tr[Φ] + D)
"""
function TP(L::LPDO)
  Λ = copy(L)
  normalize!(Λ; localnorm=2)
  Φ = tr(Λ, "Output")

  D = 2^length(Φ)
  @assert D ≈ tr(Φ)
  Γ = (1 / sqrt(D)) * sqrt(inner(Φ, Φ) - D)
  return real(Γ)
end

"""
    gradnll(L::LPDO{MPS},
            data::Matrix{Pair{String,Pair{String, Int}}};
            sqrt_localnorms = nothing)

Compute the gradients of the negative log-likelihood for process data
using a MPS representation of the vectorized unitary operation.
"""
function gradnll(
  L::LPDO{MPS}, data::Matrix{Pair{String,Pair{String,Int}}}; sqrt_localnorms=nothing
)
  data_in = first.(data)
  data_out = convertdatapoints(last.(data))

  ψ = L.X
  N = length(ψ)

  s_in = [firstind(ψ[j]; tags="Input") for j in 1:length(ψ)]
  s_out = [firstind(ψ[j]; tags="Output") for j in 1:length(ψ)]

  links = [linkind(ψ, n) for n in 1:(N - 1)]

  ElT = eltype(ψ[1])

  nthreads = Threads.nthreads()

  L = [Vector{ITensor}(undef, N) for _ in 1:nthreads]
  Lψ = [Vector{ITensor}(undef, N) for _ in 1:nthreads]

  R = [Vector{ITensor}(undef, N) for _ in 1:nthreads]
  Rψ = [Vector{ITensor}(undef, N) for _ in 1:nthreads]

  P = [Vector{ITensor}(undef, N) for _ in 1:nthreads]

  for nthread in 1:nthreads
    for n in 1:(N - 1)
      L[nthread][n] = ITensor(ElT, undef, links[n])
      Lψ[nthread][n] = ITensor(ElT, undef, s_in[n], s_out[n], links[n])
    end
    Lψ[nthread][N] = ITensor(ElT, undef, s_in[N], s_out[N])

    for n in N:-1:2
      R[nthread][n] = ITensor(ElT, undef, links[n - 1])
      Rψ[nthread][n] = ITensor(ElT, undef, links[n - 1], s_in[n], s_out[n])
    end
    Rψ[nthread][1] = ITensor(ElT, undef, s_in[1], s_out[1])

    for n in 1:N
      P[nthread][n] = ITensor(ElT, undef, s_in[n], s_out[n])
    end
  end

  if isnothing(sqrt_localnorms)
    sqrt_localnorms = ones(N)
  end

  ψdag = dag(ψ)

  gradients = [[ITensor(ElT, inds(ψ[j])) for j in 1:N] for _ in 1:nthreads]

  grads = [[ITensor(ElT, undef, inds(ψ[j])) for j in 1:N] for _ in 1:nthreads]

  loss = zeros(nthreads)

  Threads.@threads for n in 1:size(data_in)[1]
    nthread = Threads.threadid()

    x_in = data_in[n, :]
    x_out = data_out[n, :]

    """ LEFT ENVIRONMENTS """
    P[nthread][1] = dag(state(x_in[1], s_in[1])) * state(x_out[1], s_out[1])
    L[nthread][1] .= ψdag[1] .* P[nthread][1]
    for j in 2:(N - 1)
      P[nthread][j] = dag(state(x_in[j], s_in[j])) * state(x_out[j], s_out[j])
      Lψ[nthread][j] .= L[nthread][j - 1] .* ψdag[j]
      L[nthread][j] .= Lψ[nthread][j] .* P[nthread][j]
    end
    P[nthread][N] = dag(state(x_in[N], s_in[N])) * state(x_out[N], s_out[N])
    Lψ[nthread][N] .= L[nthread][N - 1] .* ψdag[N]
    ψx = (Lψ[nthread][N] * P[nthread][N])[]
    prob = abs2(ψx)
    loss[nthread] -= log(prob) / size(data_in)[1]

    #""" RIGHT ENVIRONMENTS """
    R[nthread][N] .= ψdag[N] .* P[nthread][N]
    for j in reverse(2:(N - 1))
      Rψ[nthread][j] .= ψdag[j] .* R[nthread][j + 1]
      R[nthread][j] .= Rψ[nthread][j] .* P[nthread][j]
    end

    """ GRADIENTS """
    # TODO: fuse into one call to mul!
    grads[nthread][1] .= P[nthread][1] .* R[nthread][2]
    gradients[nthread][1] .+= (1 / (sqrt_localnorms[1] * ψx)) .* grads[nthread][1]
    for j in 2:(N - 1)
      Rψ[nthread][j] .= L[nthread][j - 1] .* P[nthread][j]
      # TODO: fuse into one call to mul!
      grads[nthread][j] .= Rψ[nthread][j] .* R[nthread][j + 1]
      gradients[nthread][j] .+= (1 / (sqrt_localnorms[j] * ψx)) .* grads[nthread][j]
    end
    grads[nthread][N] .= L[nthread][N - 1] .* P[nthread][N]
    gradients[nthread][N] .+= (1 / (sqrt_localnorms[N] * ψx)) .* grads[nthread][N]
  end

  for nthread in 1:nthreads
    for g in gradients[nthread]
      g .= (-2 / size(data_in)[1]) .* g
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

"""
    gradnll(L::LPDO{MPO},
            data::Matrix{Pair{String,Pair{String, Int}}};
            sqrt_localnorms = nothing)

Compute the negative log-likelihood of process data using a LPDO ansatz
for the Choi matrix.
"""
function gradnll(
  L::LPDO{MPO}, data::Matrix{Pair{String,Pair{String,Int}}}; sqrt_localnorms=nothing
)
  data_in = first.(data)
  data_out = convertdatapoints(last.(data))

  ρ = L.X
  N = length(ρ)

  s_in = [firstind(ρ[j]; tags="Input") for j in 1:length(ρ)]
  s_out = [firstind(ρ[j]; tags="Output") for j in 1:length(ρ)]

  links = [linkind(ρ, n) for n in 1:(N - 1)]

  ElT = eltype(ρ[1])

  kraus = Index[]
  for j in 1:N
    push!(kraus, firstind(ρ[j], default_purifier_tags))
  end

  nthreads = Threads.nthreads()

  L = [Vector{ITensor}(undef, N) for _ in 1:nthreads]
  Lρ = [Vector{ITensor}(undef, N) for _ in 1:nthreads]
  Lgrad = [Vector{ITensor}(undef, N) for _ in 1:nthreads]

  R = [Vector{ITensor}(undef, N) for _ in 1:nthreads]
  Rρ = [Vector{ITensor}(undef, N) for _ in 1:nthreads]

  Agrad = [Vector{ITensor}(undef, N) for _ in 1:nthreads]

  T = [Vector{ITensor}(undef, N) for _ in 1:nthreads]
  Tp = [Vector{ITensor}(undef, N) for _ in 1:nthreads]

  grads = [Vector{ITensor}(undef, N) for _ in 1:nthreads]
  gradients = [Vector{ITensor}(undef, N) for _ in 1:nthreads]

  P = [Vector{ITensor}(undef, N) for _ in 1:nthreads]

  for nthread in 1:nthreads
    for n in 1:(N - 1)
      L[nthread][n] = ITensor(ElT, undef, links[n]', links[n])
    end
    for n in 2:(N - 1)
      Lρ[nthread][n] = ITensor(ElT, undef, kraus[n], links[n]', links[n - 1])
    end
    for n in 1:(N - 2)
      Lgrad[nthread][n] = ITensor(ElT, undef, links[n], kraus[n + 1], links[n + 1]')
    end
    Lgrad[nthread][N - 1] = ITensor(ElT, undef, links[N - 1], kraus[N])

    for n in N:-1:2
      R[nthread][n] = ITensor(ElT, undef, links[n - 1]', links[n - 1])
    end
    for n in (N - 1):-1:2
      Rρ[nthread][n] = ITensor(ElT, undef, links[n - 1]', kraus[n], links[n])
    end

    Agrad[nthread][1] = ITensor(ElT, undef, kraus[1], links[1]', s_in[1], s_out[1])
    for n in 2:(N - 1)
      Agrad[nthread][n] = ITensor(
        ElT, undef, links[n - 1], kraus[n], links[n]', s_in[n], s_out[n]
      )
    end

    T[nthread][1] = ITensor(ElT, undef, kraus[1], links[1])
    Tp[nthread][1] = prime(T[nthread][1], "Link")
    for n in 2:(N - 1)
      T[nthread][n] = ITensor(ElT, undef, kraus[n], links[n], links[n - 1])
      Tp[nthread][n] = prime(T[nthread][n], "Link")
    end
    T[nthread][N] = ITensor(ElT, undef, kraus[N], links[N - 1])
    Tp[nthread][N] = prime(T[nthread][N], "Link")

    grads[nthread][1] = ITensor(ElT, undef, links[1], kraus[1], s_in[1], s_out[1])
    gradients[nthread][1] = ITensor(ElT, links[1], kraus[1], s_in[1], s_out[1])
    for n in 2:(N - 1)
      grads[nthread][n] = ITensor(
        ElT, undef, links[n], links[n - 1], kraus[n], s_in[n], s_out[n]
      )
      gradients[nthread][n] = ITensor(
        ElT, links[n], links[n - 1], kraus[n], s_in[n], s_out[n]
      )
    end
    grads[nthread][N] = ITensor(ElT, undef, links[N - 1], kraus[N], s_in[N], s_out[N])
    gradients[nthread][N] = ITensor(ElT, links[N - 1], kraus[N], s_in[N], s_out[N])

    for n in 1:N
      P[nthread][n] = ITensor(ElT, undef, s_in[n], s_out[n])
    end
  end

  if isnothing(sqrt_localnorms)
    sqrt_localnorms = ones(N)
  end

  loss = zeros(nthreads)

  Threads.@threads for n in 1:size(data_in)[1]
    nthread = Threads.threadid()

    x_in = data_in[n, :]
    x_out = data_out[n, :]

    """ LEFT ENVIRONMENTS """
    P[nthread][1] = dag(state(x_in[1], s_in[1])) * state(x_out[1], s_out[1])
    P[nthread][1] = dag(P[nthread][1])
    T[nthread][1] .= ρ[1] .* P[nthread][1]
    L[nthread][1] .= prime(T[nthread][1], "Link") .* dag(T[nthread][1])
    for j in 2:(N - 1)
      P[nthread][j] = dag(state(x_in[j], s_in[j])) * state(x_out[j], s_out[j])
      P[nthread][j] = dag(P[nthread][j])
      T[nthread][j] .= ρ[j] .* P[nthread][j]
      Lρ[nthread][j] .= prime(T[nthread][j], "Link") .* L[nthread][j - 1]
      L[nthread][j] .= Lρ[nthread][j] .* dag(T[nthread][j])
    end
    P[nthread][N] = dag(state(x_in[N], s_in[N])) * state(x_out[N], s_out[N])
    P[nthread][N] = dag(P[nthread][N])
    T[nthread][N] .= ρ[N] .* P[nthread][N]
    prob = L[nthread][N - 1] * prime(T[nthread][N], "Link")
    prob = prob * dag(T[nthread][N])
    prob = real(prob[])
    loss[nthread] -= log(prob) / size(data_in)[1]

    """ RIGHT ENVIRONMENTS """
    R[nthread][N] .= prime(T[nthread][N], "Link") .* dag(T[nthread][N])
    for j in reverse(2:(N - 1))
      Rρ[nthread][j] .= prime(T[nthread][j], "Link") .* R[nthread][j + 1]
      R[nthread][j] .= Rρ[nthread][j] .* dag(T[nthread][j])
    end

    """ GRADIENTS """

    Tp[nthread][1] .= prime(ρ[1], "Link") .* P[nthread][1]
    Agrad[nthread][1] .= Tp[nthread][1] .* dag(P[nthread][1])
    grads[nthread][1] .= R[nthread][2] .* Agrad[nthread][1]
    gradients[nthread][1] .+= (1 / (sqrt_localnorms[1] * prob)) .* grads[nthread][1]
    for j in 2:(N - 1)
      Tp[nthread][j] .= prime(ρ[j], "Link") .* P[nthread][j]
      Lgrad[nthread][j - 1] .= L[nthread][j - 1] .* Tp[nthread][j]
      Agrad[nthread][j] .= Lgrad[nthread][j - 1] .* dag(P[nthread][j])
      grads[nthread][j] .= R[nthread][j + 1] .* Agrad[nthread][j]
      gradients[nthread][j] .+= (1 / (sqrt_localnorms[j] * prob)) .* grads[nthread][j]
    end
    Tp[nthread][N] .= prime(ρ[N], "Link") .* P[nthread][N]
    Lgrad[nthread][N - 1] .= L[nthread][N - 1] .* Tp[nthread][N]
    grads[nthread][N] .= Lgrad[nthread][N - 1] .* dag(P[nthread][N])
    gradients[nthread][N] .+= (1 / (sqrt_localnorms[N] * prob)) .* grads[nthread][N]
  end

  for nthread in 1:nthreads
    for g in gradients[nthread]
      g .= (-2 / size(data_in)[1]) .* g
    end
  end

  gradients_tot = Vector{ITensor}(undef, N)
  gradients_tot[1] = ITensor(ElT, links[1], kraus[1], s_in[1], s_out[1])
  for n in 2:(N - 1)
    gradients_tot[n] = ITensor(ElT, links[n], links[n - 1], kraus[n], s_in[n], s_out[n])
  end
  gradients_tot[N] = ITensor(ElT, links[N - 1], kraus[N], s_in[N], s_out[N])

  loss_tot = 0.0
  for nthread in 1:nthreads
    gradients_tot .+= gradients[nthread]
    loss_tot += loss[nthread]
  end

  return gradients_tot, loss_tot
end

"""
    gradTP(L::LPDO, gradlogZ::Vector{<:ITensor}, 
           logZ::Float64; sqrt_localnorms = nothing)

Compute the gradients of the trace-preserving regularization.
"""
function gradTP(
  L::LPDO, gradlogZ::Vector{<:ITensor}, logZ::Float64; sqrt_localnorms=nothing
)
  N = length(L)
  D = 2^N

  gradients_TrΦ², trΦ² = grad_TrΦ²(L; sqrt_localnorms=sqrt_localnorms)

  trΦ = exp(logZ)
  @assert D ≈ trΦ
  Γ = (1 / sqrt(D)) * sqrt(trΦ² - D)

  gradients = Vector{ITensor}(undef, N)

  for j in 1:N
    grad = gradients_TrΦ²[j] - 2 * trΦ² * gradlogZ[j]
    gradients[j] = (1 / D) * grad / (2.0 * Γ)
  end
  return gradients, Γ
end

function grad_TrΦ²(L::LPDO{MPS}; sqrt_localnorms=nothing)
  N = length(L)
  Ψ = copy(L.X)
  Ψdag = dag(Ψ)

  if isnothing(sqrt_localnorms)
    sqrt_localnorms = ones(N)
  end

  L = Vector{ITensor}(undef, N - 1)
  R = Vector{ITensor}(undef, N)

  L[1] = Ψdag[1] * prime(prime(Ψ[1], "Link"), "Input")
  L[1] = L[1] * prime(prime(Ψdag[1]), "Link")
  L[1] = L[1] * prime(prime(Ψ[1], "Output"), 3, "Link")

  for j in 2:(N - 1)
    L[j] = L[j - 1] * Ψdag[j]
    L[j] = L[j] * prime(prime(Ψ[j], "Link"), "Input")
    L[j] = L[j] * prime(prime(Ψdag[j]), "Link")
    L[j] = L[j] * prime(prime(Ψ[j], "Output"), 3, "Link")
  end
  trΦ² = L[N - 1] * Ψdag[N]
  trΦ² = trΦ² * prime(prime(Ψ[N], "Link"), "Input")
  trΦ² = trΦ² * prime(prime(Ψdag[N]), "Link")
  trΦ² = trΦ² * prime(prime(Ψ[N], "Output"), 3, "Link")
  trΦ² = real(trΦ²[])

  R[N] = Ψdag[N] * prime(prime(Ψ[N], "Link"), "Input")
  R[N] = R[N] * prime(prime(Ψdag[N]), "Link")
  R[N] = R[N] * prime(prime(Ψ[N], "Output"), 3, "Link")

  for j in reverse(2:(N - 1))
    R[j] = R[j + 1] * Ψdag[j]
    R[j] = R[j] * prime(prime(Ψ[j], "Link"), "Input")
    R[j] = R[j] * prime(prime(Ψdag[j]), "Link")
    R[j] = R[j] * prime(prime(Ψ[j], "Output"), 3, "Link")
  end

  gradients = Vector{ITensor}(undef, N)
  tmp = prime(Ψ[1], 3, "Link") * R[2]
  tmp = tmp * prime(prime(Ψdag[1], 2, "Link"), "Input")
  gradients[1] = (prime(prime(Ψ[1], "Link"), "Input") * tmp) / (sqrt_localnorms[1])

  for j in 2:(N - 1)
    tmp = prime(Ψ[j], 3, "Link") * L[j - 1]
    tmp = tmp * prime(prime(Ψdag[j], 2, "Link"), "Input")
    tmp = prime(prime(Ψ[j], "Link"), "Input") * tmp
    gradients[j] = (tmp * R[j + 1]) / (sqrt_localnorms[j])
  end
  tmp = prime(Ψ[N], 3, "Link") * L[N - 1]
  tmp = prime(prime(Ψdag[N], 2, "Link"), "Input") * tmp
  gradients[N] = (prime(prime(Ψ[N], "Link"), "Input") * tmp) / (sqrt_localnorms[N])

  return 4 * gradients, trΦ²
end

function grad_TrΦ²(Λ::LPDO{MPO}; sqrt_localnorms=nothing)
  N = length(Λ)

  if isnothing(sqrt_localnorms)
    sqrt_localnorms = ones(N)
  end

  L = Vector{ITensor}(undef, N - 1)
  R = Vector{ITensor}(undef, N)
  L[1] = bra(Λ, 1) * noprime(ket(Λ, 1); tags="Output")
  L[1] = L[1] * prime(bra(Λ, 1)', "Link")
  L[1] = L[1] * prime(prime(noprime(ket(Λ, 1), "Input"), 2, "Link"), default_purifier_tags)
  for j in 2:(N - 1)
    L[j] = L[j - 1] * bra(Λ, j)
    L[j] = L[j] * noprime(ket(Λ, j); tags="Output")
    L[j] = L[j] * prime(bra(Λ, j)', "Link")
    L[j] =
      L[j] * prime(prime(noprime(ket(Λ, j), "Input"), 2, "Link"), default_purifier_tags)
  end
  trΦ² = L[N - 1] * bra(Λ, N)
  trΦ² = trΦ² * noprime(ket(Λ, N); tags="Output")
  trΦ² = trΦ² * prime(bra(Λ, N)', "Link")
  trΦ² = trΦ² * prime(prime(noprime(ket(Λ, N), "Input"), 2, "Link"), default_purifier_tags)
  trΦ² = real(trΦ²[])

  R[N] = bra(Λ, N) * noprime(ket(Λ, N); tags="Output")
  R[N] = R[N] * prime(bra(Λ, N)', "Link")
  R[N] = R[N] * prime(prime(noprime(ket(Λ, N), "Input"), 2, "Link"), default_purifier_tags)

  for j in reverse(2:(N - 1))
    R[j] = R[j + 1] * bra(Λ, j)
    R[j] = R[j] * noprime(ket(Λ, j); tags="Output")
    R[j] = R[j] * prime(bra(Λ, j)', "Link")
    R[j] =
      R[j] * prime(prime(noprime(ket(Λ, j), "Input"), 2, "Link"), default_purifier_tags)
  end

  gradients = Vector{ITensor}(undef, N)
  tmp = prime(noprime(ket(Λ, 1)), 3, "Link") * R[2]
  tmp = tmp * prime(prime(bra(Λ, 1), "Input"), 2, "Link")
  gradients[1] = (noprime(ket(Λ, 1), "Output") * tmp) / (sqrt_localnorms[1])

  for j in 2:(N - 1)
    tmp = prime(noprime(ket(Λ, j)), 3, "Link") * L[j - 1]
    tmp = tmp * prime(prime(bra(Λ, j), "Input"), 2, "Link")
    tmp = noprime(ket(Λ, j), "Output") * tmp
    gradients[j] = (tmp * R[j + 1]) / (sqrt_localnorms[j])
  end
  tmp = prime(noprime(ket(Λ, N)), 3, "Link") * L[N - 1]
  tmp = prime(prime(bra(Λ, N), "Input"), 2, "Link") * tmp
  gradients[N] = (noprime(ket(Λ, N), "Output") * tmp) / (sqrt_localnorms[N])

  return 4 * gradients, trΦ²
end

"""
    PastaQ.gradients(L::LPDO, data::Array; sqrt_localnorms = nothing)
    PastaQ.gradients(ψ::MPS, data::Array; localnorms = nothing)

Compute the gradients of the cost function:
`C = log(Z) - ⟨log P(σ)⟩_data + TP`
"""
function gradients(
  L::LPDO,
  data::Matrix{Pair{String,Pair{String,Int}}};
  sqrt_localnorms=nothing,
  trace_preserving_regularizer=nothing,
)
  g_logZ, logZ = gradlogZ(L; sqrt_localnorms=sqrt_localnorms)
  g_nll, NLL = gradnll(L, data; sqrt_localnorms=sqrt_localnorms)
  g_TP, TP_distance = gradTP(L, g_logZ, logZ; sqrt_localnorms=sqrt_localnorms)

  grads = g_logZ + g_nll
  loss = logZ + NLL

  # trace-preserving regularization
  if !isnothing(trace_preserving_regularizer)
    grads += trace_preserving_regularizer * g_TP
  end
  return grads, loss
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
  train_data::Matrix{Pair{String,Pair{String,Int}}}, L::LPDO; (observer!)=nothing, kwargs...
)

  # Read arguments
  optimizer::Optimizer = get(kwargs, :optimizer, SGD(; η=0.01))
  batchsize::Int64 = get(kwargs, :batchsize, 100)
  epochs::Int64 = get(kwargs, :epochs, 1000)
  trace_preserving_regularizer = get(kwargs, :trace_preserving_regularizer, 0.0)
  measurement_frequency::Int64 = get(kwargs, :measurement_frequency, 1)
  test_data = get(kwargs, :test_data, nothing)
  outputpath = get(kwargs, :fout, nothing)
  print_metrics = get(kwargs, :print_metrics, [])
  outputpath = get(kwargs, :outputpath, nothing)
  outputlevel = get(kwargs, :outputlevel, 1)
  savemodel = get(kwargs, :savemodel, false)

  # configure the observer. if no observer is provided, create an empty one
  observer! = configure!(
    observer!, optimizer, batchsize, measurement_frequency, train_data, test_data
  )

  optimizer = copy(optimizer)
  model = copy(L)

  @assert size(train_data, 2) == length(model)
  if !isnothing(test_data)
    @assert size(test_data)[2] == length(model)
    best_test_loss = 1_000
  end

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
        normalize!(normalized_model; (sqrt_localnorms!)=sqrt_localnorms, localnorm=2)

        grads, loss = gradients(
          normalized_model,
          batch;
          sqrt_localnorms=sqrt_localnorms,
          trace_preserving_regularizer=trace_preserving_regularizer,
        )

        nupdate = ep * num_batches + b
        train_loss += loss / Float64(num_batches)
        update!(model, grads, optimizer; step=nupdate)
      end
    end # end @elapsed
    tot_time += ep_time
    # measurement stage
    if ep % measurement_frequency == 0
      normalized_model = copy(model)
      sqrt_localnorms = []
      normalize!(normalized_model; (sqrt_localnorms!)=sqrt_localnorms, localnorm=2)
      if !isnothing(test_data)
        test_loss = nll(normalized_model, test_data)
        if test_loss ≤ best_testloss
          best_testloss = test_loss
          best_model = copy(normalized_model)
        end
      else
        best_model = copy(model)
      end

      if model isa LPDO{MPS}
        update!(
          observer!,
          choi_mps_to_unitary_mpo(normalized_model),
          best_model,
          tot_time,
          train_loss,
          test_loss,
        )
      else
        update!(observer!, normalized_model, best_model, tot_time, train_loss, test_loss)
      end
      # printing
      if outputlevel ≥ 1 
        @printf("%-4d  ", ep)
        @printf("⟨logP⟩ = %-4.4f  ", results(observer!, "train_loss")[end])
        if !isnothing(test_data) 
          @printf("(%.4f)  ", results(observer!, "test_loss")[end])
        end
        printobserver(observer!, print_metrics)
        @printf("elapsed = %-4.3fs", ep_time)
        println()
      end
      # saving
      if !isnothing(outputpath)
        model_to_be_saved = (!savemodel ? nothing :
                             model isa LPDO{MPS} ? choi_mps_to_unitary_mpo(best_model) : best_model)
        savetomographyobserver(observer!, outputpath; model = model_to_be_saved)
      end
    end
  end
  return best_model
end

function tomography(data::Matrix{Pair{String,Pair{String,Int}}}, U::MPO; kwargs...)
  return choi_mps_to_unitary_mpo(
    tomography(data, LPDO(unitary_mpo_to_choi_mps(U)); kwargs...)
  )
end
