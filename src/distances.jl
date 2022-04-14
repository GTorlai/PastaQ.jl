# TODO: make a faster custom version that doesn't convert to MPO first
ITensors.inner(ρ::LPDO{MPO}, σ::LPDO{MPO}) = inner(MPO(ρ), MPO(σ))

ITensors.inner(ρ::LPDO{MPS}, σ::LPDO{MPS}) = abs(inner(ρ.X, σ.X))^2

ITensors.inner(ρ::LPDO{MPO}, σ::MPO) = inner(MPO(ρ), σ)

ITensors.inner(ρ::LPDO{MPS}, σ::MPO) = inner(ρ.X, σ, ρ.X)

ITensors.inner(ρ::MPO, σ::LPDO{MPO}) = inner(ρ, MPO(σ))

ITensors.inner(ρ::MPO, σ::LPDO{MPS}) = inner(σ.X, ρ, σ.X)

@doc raw"""
    fidelity(ψ::MPS, ϕ::MPS; kwargs...)

Quantum state fidelity between two wavefunctions:

```math 
F(\psi,\phi) = |\langle\phi|\psi\rangle|^2.
```
"""
function fidelity(ψ::MPS, ϕ::MPS; kwargs...)
  log_F̃ = 2.0 * real(loginner(ψ, ϕ))
  log_K = 2.0 * (lognorm(ψ) + lognorm(ϕ))
  fidelity = exp(log_F̃ - log_K)
  return fidelity
end

@doc raw"""
    fidelity(ψ::MPS, ρ::MPO; kwargs...)
    fidelity(ρ::MPO, ψ::MPS; kwargs...)

Quantum state fidelity between an MPS wavefunction and an MPO 
density operator:

```math 
F(\psi,\rho) = \langle\psi|\rho|\psi\rangle.
```

"""
function fidelity(ψ::MPS, ρ::MPO; kwargs...)
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

fidelity(ρ::MPO, ψ::MPS; kwargs...) = fidelity(ψ, ρ)

@doc raw"""
    fidelity(Ψ::MPS, ϱ::LPDO{MPO}; cutoff::Float64 = 1e-15)
    fidelity(ϱ::LPDO{MPO}, ψ::MPS; kwargs...)

Quantum state fidelity between an MPS wavefunction and a 
LPDO density operator ``\varrho=XX^\dagger``

```math 
F(\psi,\rho) = \langle\psi|\varrho|\psi\rangle = |X^\dagger|\psi\rangle|^2.
```

"""
function fidelity(Ψ::MPS, ϱ::LPDO{MPO}; cutoff::Float64=1e-15)
  # TODO: fix exponential scaling
  #proj = bra(ϱ) * Ψ
  proj = *(bra(ϱ), Ψ; cutoff=cutoff)
  K = abs2(norm(Ψ)) * tr(ϱ)
  return inner(proj', proj) / K
end

fidelity(ϱ::LPDO{MPO}, ψ::MPS; kwargs...) = fidelity(ψ, ϱ; kwargs...)

struct Choi{T}
  X::T
end

@doc raw"""
    fidelity(A::MPO, B::MPO; process::Bool = false, cutoff::Float64 = 1e-15)

Fidelity ``F`` between two MPOs ``A`` and ``B``. Implements the following:
1. If ``A`` and ``B`` are density operators, ``F`` is the full quantum state fidelity

```math 
F(\rho,\sigma) = \Big(\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\Big)^2.
```

Note: this scales exponentially with the number ``n`` of qubits, as it involves a full
diagonalization.

2. If ``A`` and ``B`` are unitary operators (i.e. rank-1 channels) and `process = true`,
``F`` is the process fidelity

```math 
F = 2^{-2n} \text{Tr}(A^\dagger B) = 2^{-2n} |\langle\Phi_A|\Phi_B\rangle|^2,
```
where ``|\Phi_j\rangle = |j\rangle\rangle`` is the MPS corresponding to the vectorization
of the unitary operator.

3. If ``A`` is a Choi matrix and `B` is a unitary operator (or viceversa), return the process fidelity

```math 
F = 2^{-2n} \text{Tr}(A |\Phi_B\rangle\langle\Phi_B|) = \langle\Phi_B|A|\Phi_B\rangle.
```

4. If ``A`` and ``B`` are both Choi matrices, return the full process fidelity

```math 
F(A,B) = 2^{-2n}\Big(\text{Tr}\sqrt{\sqrt{A}B\sqrt{A}}\Big)^2.
```

which, as above, scale exponentially with ``n``.
"""
function fidelity(A::MPO, B::MPO; process::Bool=false, cutoff::Float64=1e-15)
  a = ischoi(A) ? Choi(A) : A
  b = ischoi(B) ? Choi(B) : B
  return _fidelity(a, b; process=process, cutoff=cutoff)
end

"""
    _fidelity(A::Choi, B::MPO; cutoff::Float64 = 1e-15, kwargs...)
    _fidelity(A::MPO, B::Choi; cutoff::Float64 = 1e-15, kwargs...)

[INTERNAL]: process fidelity between a Choi matrix and a unitary MPO.
"""
function _fidelity(A::Choi, B::MPO; cutoff::Float64=1e-15, kwargs...)
  return fidelity(A.X, unitary_mpo_to_choi_mps(B); cutoff=cutoff)
end

function _fidelity(A::MPO, B::Choi; cutoff::Float64=1e-15, kwargs...)
  return _fidelity(B, A; cutoff=cutoff)
end

"""
    _fidelity(A::Choi, B::Choi; kwargs...)

[INTERNAL]: process fidelity between two Choi matrices
"""
_fidelity(A::Choi, B::Choi; kwargs...) = fidelity(prod(A.X), prod(B.X); kwargs...)

"""
    _fidelity(A::MPO, B::MPO; process::Bool = false, kwargs...)

[INTERNAL]: fidelity between two MPOs, which could be either two unitary MPOs
or two density matrices.
"""
function _fidelity(A::MPO, B::MPO; process::Bool=false, kwargs...)
  # TODO: sub after implementing MPDO
  #A = !process ? MPDO(A) : A
  #B = !process ? MPDO(B) : B
  A = !process ? A : unitary_mpo_to_choi_mps(A)
  B = !process ? B : unitary_mpo_to_choi_mps(B)
  return __fidelity(A, B; kwargs...)
end

# TODO: sub after implementing MPDO
#function __fidelity(A::MPDO, B::MPDO)
#  return fidelity(prod(A), prod(B))
#end
#function __fidelity(A::MPO, B::MPO)
#  return fidelity(unitary_mpo_to_choi_mps(A), unitary_mpo_to_choi_mps(B))
#end

"""
    __fidelity(A::MPO, B::MPO; kwargs...)

[INTERNAL]: state fidelity between two density matrices
"""
__fidelity(A::MPO, B::MPO; kwargs...) = fidelity(prod(A), prod(B); kwargs...)

"""
    __fidelity(A::MPS, B::MPS; kwargs...)
[INTERNAL]: process fidelity between two unitary MPOs
"""
__fidelity(A::MPS, B::MPS; kwargs...) = fidelity(A, B)

"""
    fidelity(A::MPO, B::LPDO{MPO}; process::Bool=false, cutoff::Float64 = 1e-15)

Fidelity between a MPO and a LPDO. Wrap the MPO in the Choi if so.
"""

function fidelity(A::MPO, B::LPDO{MPO}; process::Bool=false, cutoff::Float64=1e-15)
  A = ischoi(A) ? Choi(A) : A
  return _fidelity(A, B; process=process, cutoff=cutoff)
end

fidelity(A::LPDO{MPO}, B::MPO; kwargs...) = fidelity(B, A; kwargs...)

"""
    _fidelity(A::Choi, B::LPDO{MPO}; process::Bool = false, kwargs...)

[INTERNAL]: process fidelity between a Choi MPO and a Choi LPDO
"""
function _fidelity(A::Choi, B::LPDO{MPO}; process::Bool=false, kwargs...)
  return fidelity(A.X, MPO(B); kwargs...)
end

_fidelity(A::LPDO{MPO}, B::Choi; kwargs...) = _fidelity(B, A; kwargs...)

"""
    _fidelity(A::MPO, B::LPDO{MPO}; process::Bool = false, kwargs...)

If process: fidelity between a unitary MPO and a Choi LPDO
if not: fidelity between a density  matrix and a LPDO density matrix.
"""
function _fidelity(A::MPO, B::LPDO{MPO}; process::Bool=false, kwargs...)
  A = !process ? A : unitary_mpo_to_choi_mps(A)
  B = !process ? MPO(B) : B
  return fidelity(A, B; kwargs...)
end

"""
    fidelity(A::LPDO{MPO}, B::LPDO{MPO}; kwargs...)

Quantum fidelity between two LPDO density matrices.
"""

fidelity(A::LPDO{MPO}, B::LPDO{MPO}; kwargs...) = fidelity(MPO(A), MPO(B); kwargs...)

struct ITensorState
  T::ITensor
end

struct ITensorOperator
  T::ITensor
end

operator_or_state(A::ITensor) = is_operator(A) ? ITensorOperator(A) : ITensorState(A)

@doc raw"""
    fidelity(A::ITensor, B::ITensor)

Compute the quantum fidelity between two ITensors. Wrap each one in the ITensorState
and the ITensorOperator according to the index structure to allow dispatch.
"""
function fidelity(A::ITensor, B::ITensor; process::Bool=false, cutoff::Float64=1e-15)
  return fidelity(
    operator_or_state(A), operator_or_state(B); process=process, cutoff=cutoff
  )
end

"""
    fidelity(A::ITensorState, B::ITensorState; kwargs...)

State fidelity between two wavefunctions
"""
function fidelity(A::ITensorState, B::ITensorState; kwargs...)
  K = (norm(A.T) * norm(B.T))^2
  return abs2((dag(A.T) * B.T)[]) / K
end

"""
    fidelity(A::ITensorState, B::ITensorOperator; kwargs...)
    fidelity(A::ITensorOperator, B::ITensorState; kwargs...)

State fidelity between a wavefunction and a density operator
"""
function fidelity(A::ITensorState, B::ITensorOperator; kwargs...)
  K = norm(A.T)^2 * tr(B.T)
  return real((dag(A.T') * B.T * A.T)[] / K)
end

fidelity(A::ITensorOperator, B::ITensorState; kwargs...) = fidelity(B, A)

"""
    fidelity(A::ITensorOperator, B::ITensorOperator; kwargs...)

Fidelity between two operators. Wrap the Choi type as for the TN (above)
"""
function fidelity(A::ITensorOperator, B::ITensorOperator; kwargs...)
  A = ischoi(A.T) ? Choi(A) : A
  B = ischoi(B.T) ? Choi(B) : B
  return _fidelity(A, B; kwargs...)
end

"""
    _fidelity(A::Choi, B::ITensorOperator; cutoff::Float64 = 1e-15, kwargs...)
    _fidelity(A::ITensorOperator, B::Choi; kwargs...)

Fidelity between a Choi and a unitary
"""
function _fidelity(A::Choi, B::ITensorOperator; cutoff::Float64=1e-15, kwargs...)
  return fidelity(A.X, choitags(B); cutoff=cutoff, kwargs...)
end

_fidelity(A::ITensorOperator, B::Choi; kwargs...) = _fidelity(B, A; kwargs...)

"""
    _fidelity(A::Choi{ITensorOperator}, B::Choi{ITensorOperator}; cutoff::Float64 = 1e-15, kwargs...)

fidelity between two Choi matrices
"""
function _fidelity(
  A::Choi{ITensorOperator}, B::Choi{ITensorOperator}; cutoff::Float64=1e-15, kwargs...
)
  return _fidelity(A.X, B.X; cutoff=cutoff)
end

"""
    _fidelity(A::ITensorOperator, B::ITensorOperator; process::Bool = false, kwargs...)

Fidelity betweeb two operators. If process, change the tags and make the unitary into a state (i.e. vectorization).
"""
function _fidelity(A::ITensorOperator, B::ITensorOperator; process::Bool=false, kwargs...)
  a = !process ? A : ITensorState(choitags(A).T)
  b = !process ? B : ITensorState(choitags(B).T)
  return __fidelity(a, b; kwargs...)
end

"""
    __fidelity(A::ITensorOperator, B::ITensorOperator; cutoff::Float64 = 1e-15)

Fidelity between two density matrices
"""
function __fidelity(A::ITensorOperator, B::ITensorOperator; cutoff::Float64=1e-15)
  a = copy(A.T)
  b = copy(B.T)
  @assert order(a) == order(b)
  a ./= tr(a)
  b ./= tr(b)
  sqrt_a = sqrt_hermitian(a; cutoff=cutoff)
  F = product(product(sqrt_a, b), sqrt_a)
  return real(tr(sqrt_hermitian(F; cutoff=cutoff)))^2
end

"""
Dummy function
"""
__fidelity(A::ITensorState, B::ITensorState; kwargs...) = fidelity(A, B)

"""
Finally, the fidelity between TN and ITensors, which simply convert
the TN into ITensors and call the ITensors fidelities.
"""

fidelity(M::Union{MPS,MPO,LPDO}, T::ITensor; kwargs...) = fidelity(prod(M), T; kwargs...)

fidelity(T::ITensor, M::Union{MPS,MPO,LPDO}; kwargs...) = fidelity(M, T; kwargs...)

@doc raw"""
    frobenius_distance(ρ::Union{MPO, LPDO}, σ::Union{MPO, LPDO})

Compute the trace norm of the difference between two LPDOs or MPOs:

```math 
T(\rho,\sigma) = \sqrt{\text{Tr}\big[(\rho-\sigma)^\dagger(\rho-\sigma)\big]}
```
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
