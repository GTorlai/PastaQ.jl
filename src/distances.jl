# TODO: make a faster custom version that doesn't convert to MPO first
ITensors.inner(ρ::LPDO{MPO}, σ::LPDO{MPO}) = inner(MPO(ρ), MPO(σ))

ITensors.inner(ρ::LPDO{MPS}, σ::LPDO{MPS}) = abs(inner(ρ.X, σ.X))^2

ITensors.inner(ρ::LPDO{MPO}, σ::MPO) = inner(MPO(ρ), σ)

ITensors.inner(ρ::LPDO{MPS}, σ::MPO) = inner(ρ.X, σ, ρ.X)

ITensors.inner(ρ::MPO, σ::LPDO{MPO}) = inner(ρ, MPO(σ))

ITensors.inner(ρ::MPO, σ::LPDO{MPS}) = inner(σ.X, ρ, σ.X)

"""
    fidelity(ρ::ITensor, σ::ITensor)

Compute the quantum fidelity between two ITensors, which are treated as density operators
from the unprimed to the primed indices (if they are matrix-like).

Matrix-like ITensors should be Hermitian and non-negative.
"""
function fidelity(ρ::ITensor, σ::ITensor)
  @assert order(ρ) == order(σ)
  ρ ./= tr(ρ)
  σ ./= tr(σ)
  F = product(product(sqrt(ρ), σ), sqrt(ρ))
  F = real(tr(sqrt(F)))^2
  return F
end

"""
Quantum state fidelity between two wavefunctions.

F = |⟨ψ|ϕ⟩|²
"""
function fidelity(ψ::MPS, ϕ::MPS)
  log_F̃ = 2.0 * real(loginner(ψ, ϕ))
  log_K = 2.0 * (lognorm(ψ) + lognorm(ϕ))
  fidelity = exp(log_F̃ - log_K)
  return fidelity
end

"""
Quantum state fidelity between an MPS wavefunction and a 
density operator.

F = ⟨ψ|ρ|ψ⟩
"""
function fidelity(ψ::MPS, ρ::MPO)
  # TODO: replace with:
  # log_F̃ = loginner(ψ, ρ, ψ)
  # log_K = 2 * lognorm(ψ) + logtr(ρ) 
  log_F̃ = log(abs(inner(ψ', ρ, ψ)))

  # TODO Check if trace is real
  #@assert imag(tr(ρ)) ≈ 0.0 atol=1e-5
  log_K = 2 * lognorm(ψ) + log(real(tr(ρ)))
  fidelity = exp(log_F̃ - log_K)
  return fidelity
end

fidelity(ρ::MPO, ψ::MPS) = fidelity(ψ, ρ)

"""
Quantum state fidelity between an MPS wavefunction and a 
LPDO density operator.

F = ⟨ψ|ϱ|ψ⟩=|X†|ψ⟩|²
"""
function fidelity(Ψ::MPS, ϱ::LPDO{MPO})
  # TODO: fix exponential scaling
  #proj = bra(ϱ) * Ψ
  proj = *(bra(ϱ), Ψ; method = "naive")
  K = abs2(norm(Ψ)) * tr(ϱ)
  return inner(proj, proj) / K
end

"""
Quantum state/process fidelity between two MPO density matrices.

F = (tr[√(√A B √A)])² 

If `process = true`, get process fidelities:
1. Quantum process fidelity between two MPO unitary operators.
   F = |⟨⟨A|B⟩⟩|²
2. Quantum process fidelity between a MPO unitary and MPO Choi matrix
   F = ⟨⟨A|B|A⟩⟩
3. Quantum process fidelity between two MPO Choi matrices
   F = (Tr[√(√A B √A)])² 
"""
function fidelity(A::MPO, B::MPO; process::Bool=false)
  ischoiA = ischoi(A)
  ischoiB = ischoi(B)
  # if quantum state fidelity:
  if process && (!ischoiA || !ischoiB)
    A = ischoiA ? A : unitary_mpo_to_choi_mps(A)
    B = ischoiB ? B : unitary_mpo_to_choi_mps(B)
    return fidelity(A, B)
  end
  # quantum state/process fidelity between two MPO density matrices 
  # or two Choi matrices
  return fidelity(prod(A), prod(B))
end

"""
1. Quantum process fidelity between a unitary MPO and a Choi LPDO.
   F = ⟨ψ|ϱ|ψ⟩=|X†|ψ⟩|²
2. Quantum process fidelity bewteen a Choi MPO and a Choi LPDO
"""
function fidelity(A::MPO, B::LPDO{MPO}; process::Bool=false)
  #1: Choi MPO   -  Choi LPDO
  (process && !ischoi(A)) && return fidelity(unitary_mpo_to_choi_mps(A), B)
  return fidelity(A, MPO(B))
end
fidelity(B::LPDO{MPO}, A::MPO; kwargs...) = fidelity(A, B; kwargs...)

"""
Quantum fidelity between two LPDO density matrices.

F = (tr[√(√A B √A)])² 
"""
fidelity(A::LPDO{MPO}, B::LPDO{MPO}; kwargs...) = fidelity(MPO(A), MPO(B); kwargs...)

"""
    frobenius_distance(ρ::Union{MPO, LPDO}, σ::Union{MPO, LPDO})

Compute the trace norm of the difference between two LPDOs and MPOs:

`T(ρ,σ) = sqrt(trace[(ρ̃-σ̃)†(ρ̃-σ̃)])`

where `ρ̃` and `σ̃` are the normalized density matrices.
"""
function frobenius_distance(ρ::Union{MPO,LPDO{MPO}}, σ::Union{MPO,LPDO{MPO}})
  ρ̃ = copy(ρ)
  σ̃ = copy(σ)
  normalize!(ρ̃)
  normalize!(σ̃)
  distance = real(inner(ρ̃, ρ̃))
  distance += real(inner(σ̃, σ̃))
  distance -= 2 * real(inner(ρ̃, σ̃))
  return sqrt(distance)
end

frobenius_distance(ψ::MPS, ρ::Union{MPO,LPDO{MPO}}) = frobenius_distance(MPO(ψ), ρ)

frobenius_distance(ρ::Union{MPO,LPDO{MPO}}, ψ::MPS) = frobenius_distance(ψ, ρ)

frobenius_distance(ψ::MPS, ϕ::MPS) = frobenius_distance(MPO(ψ), MPO(ϕ))

"""
    fidelity_bound(ρ::Union{MPO, LPDO}, σ::Union{MPO, LPDO})

Compute the the following lower bound of the fidelity:

`F̃(ρ,σ) = trace[ρ̃† σ̃]`

where `ρ̃` and `σ̃` are the normalized density matrices.

The bound becomes tight when the target state is nearly pure.
"""
function fidelity_bound(ρ::Union{MPO,LPDO{MPO}}, σ::Union{MPO,LPDO{MPO}})
  ρ̃ = copy(ρ)
  σ̃ = copy(σ)
  normalize!(ρ̃)
  normalize!(σ̃)
  return real(inner(ρ̃, σ̃))
end

fidelity_bound(ψ::MPS, ρ::Union{MPO,LPDO{MPO}}) = fidelity(ψ, ρ)

fidelity_bound(ρ::Union{MPO,LPDO{MPO}}, ψ::MPS) = fidelity(ρ, ψ)

fidelity_bound(ψ::MPS, ϕ::MPS) = fidelity(ψ, ϕ)
