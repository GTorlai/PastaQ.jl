
"""
    qubits(N::Int; mixed::Bool=false)
    
    qubits(sites::Vector{<:Index}; mixed::Bool=false)


Initialize qubits to:
- An MPS wavefunction `|ψ⟩` if `mixed = false`
- An MPO density matrix `ρ` if `mixed = true`
"""
qubits(N::Int; mixed::Bool=false) = qubits(siteinds("Qubit", N); mixed=mixed)

function qubits(sites::Vector{<:Index}; mixed::Bool=false)
  @warn "Method `qubits` is deprecated, use `productstate` or `productoperator` instead."
  ψ = productMPS(sites, "0")
  mixed && return MPO(ψ)
  return ψ
end

"""
    qubits(M::Union{MPS,MPO,LPDO}; mixed::Bool=false)

Initialize qubits on the Hilbert space of a reference state,
given as `MPS`, `MPO` or `LPDO`.
"""
qubits(M::Union{MPS,MPO,LPDO}; mixed::Bool=false) = qubits(hilbertspace(M); mixed=mixed)

"""
    qubits(N::Int, states::Vector{String}; mixed::Bool=false)

    qubits(sites::Vector{<:Index}, states::Vector{String};mixed::Bool = false)

Initialize the qubits to a given single-qubit product state.
"""
function qubits(N::Int, states::Vector{String}; mixed::Bool=false)
  return qubits(siteinds("Qubit", N), states; mixed=mixed)
end

function qubits(sites::Vector{<:Index}, states::Vector{String}; mixed::Bool=false)
  @warn "Method `qubits` is deprecated, use `productstate` or `productoperator` instead."
  N = length(sites)
  @assert N == length(states)

  ψ = productMPS(sites, "0")

  if N == 1
    s1 = sites[1]
    state1 = state(states[1])
    if eltype(state1) <: Complex
      ψ[1] = complex(ψ[1])
    end
    for j in 1:dim(s1)
      ψ[1][s1 => j] = state1[j]
    end
    mixed && return MPO(ψ)
    return ψ
  end

  # Set first site
  s1 = sites[1]
  l1 = linkind(ψ, 1)
  state1 = state(states[1])
  if eltype(state1) <: Complex
    ψ[1] = complex(ψ[1])
  end
  for j in 1:dim(s1)
    ψ[1][s1 => j, l1 => 1] = state1[j]
  end

  # Set sites 2:N-1
  for n in 2:(N - 1)
    sn = sites[n]
    ln_1 = linkind(ψ, n - 1)
    ln = linkind(ψ, n)
    state_n = state(states[n])
    if eltype(state_n) <: Complex
      ψ[n] = complex(ψ[n])
    end
    for j in 1:dim(sn)
      ψ[n][sn => j, ln_1 => 1, ln => 1] = state_n[j]
    end
  end

  # Set last site N
  sN = sites[N]
  lN_1 = linkind(ψ, N - 1)
  state_N = state(states[N])
  if eltype(state_N) <: Complex
    ψ[N] = complex(ψ[N])
  end
  for j in 1:dim(sN)
    ψ[N][sN => j, lN_1 => 1] = state_N[j]
  end

  mixed && return MPO(ψ)
  return ψ
end

