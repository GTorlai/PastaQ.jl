
# TODO: make a faster custom version that doesn't convert to MPO first
ITensors.inner(ρ::LPDO{MPO}, σ::LPDO{MPO}) =
  inner(MPO(ρ), MPO(σ))

ITensors.inner(ρ::LPDO{MPS}, σ::LPDO{MPS}) =
  abs(inner(ρ.X, σ.X))^2

ITensors.inner(ρ::LPDO{MPO}, σ::MPO) =
  inner(MPO(ρ), σ)

ITensors.inner(ρ::LPDO{MPS}, σ::MPO) =
  inner(ρ.X, σ, ρ.X)

ITensors.inner(ρ::MPO, σ::LPDO{MPO}) =
  inner(ρ, MPO(σ))

ITensors.inner(ρ::MPO, σ::LPDO{MPS}) =
  inner(σ.X, ρ, σ.X)

"""
    fidelity(ψ::MPS, ϕ::MPS)

Compute the fidelity between two MPS:

`F = |⟨ψ̃|ϕ̃⟩|²`

where `ψ̃` and `ϕ̃` are the normalized MPS.
"""
function fidelity(ψ::MPS, ϕ::MPS)
  log_F̃ = 2.0 * real(loginner(ψ, ϕ))
  log_K = 2.0 * (lognorm(ψ) + lognorm(ϕ))
  fidelity = exp(log_F̃ - log_K)
  return fidelity
end

"""
    fidelity(ψ::MPS, ρ::Union{MPO, LPDO})
    fidelity(ρ::Union{MPO, LPDO}, ψ::MPS)

Compute the fidelity between an MPS and MPO/LPDO:

`F = ⟨ψ̃|ρ̃|ψ̃⟩`

where `ψ̃` and `ρ̃` are the normalized MPS and MPO/LDPO.
"""
function fidelity(ψ::MPS, ρ::MPO)
  # TODO: replace with:
  # log_F̃ = loginner(ψ, ρ, ψ)
  # log_K = 2 * lognorm(ψ) + logtr(ρ) 
  log_F̃ = log(abs(inner(ψ, ρ, ψ)))
  
  # TODO Check if trace is real
  #@assert imag(tr(ρ)) ≈ 0.0 atol=1e-5
  log_K = 2 * lognorm(ψ) + log(real(tr(ρ))) 
  
  fidelity = exp(log_F̃ - log_K)
  return fidelity
end

# TODO: replace with:
# X = L.X
# log_F̃ = lognorm(X, ψ) # = loginner(X, ψ, X, ψ)
# log_K = 2.0 * (lognorm(ψ) + lognorm(L))
# return exp(log_F̃ - log_K)
fidelity(ψ::MPS, L::LPDO{MPO}) = fidelity(ψ, MPO(L))

fidelity(ψ::MPS, L::LPDO{MPS}) = fidelity(ψ, L.X)

fidelity(ρ::LPDO, ψ::MPS) = fidelity(ψ, ρ)

fidelity(ρ::MPO, ψ::MPS) = fidelity(ψ, ρ)

"""
    frobenius_distance(ρ::Union{MPO, LPDO}, σ::Union{MPO, LPDO})

Compute the trace norm of the difference between two LPDOs and MPOs:

`T(ρ,σ) = sqrt(trace[(ρ̃-σ̃)†(ρ̃-σ̃)])`

where `ρ̃` and `σ̃` are the normalized density matrices.
"""
function frobenius_distance(ρ::Union{MPO, LPDO},
                            σ::Union{MPO, LPDO})
  ρ̃ = copy(ρ)
  σ̃ = copy(σ)
  normalize!(ρ̃)
  normalize!(σ̃)
  distance  = real(inner(ρ̃, ρ̃))
  distance += real(inner(σ̃, σ̃))
  distance -= 2 * real(inner(ρ̃, σ̃))
  return sqrt(distance)
end

"""
    fidelity_bound(ρ::Union{MPO, LPDO}, σ::Union{MPO, LPDO})

Compute the the following lower bound of the fidelity:

`F̃(ρ,σ) = trace[ρ̃† σ̃]`

where `ρ̃` and `σ̃` are the normalized density matrices.

The bound becomes tight when the target state is nearly pure.
"""
function fidelity_bound(ρ::Union{MPO, LPDO},
                        σ::Union{MPO, LPDO})
  ρ̃ = copy(ρ)
  σ̃ = copy(σ)
  normalize!(ρ̃)
  normalize!(σ̃)
  return real(inner(ρ̃, σ̃))
end

"""
    fidelity(ρ::ITensor, σ::ITensor)

Compute the quantum fidelity between two ITensors, which are treated as density operators
from the unprimed to the primed indices (if they are matrix-like).

Matrix-like ITensors should be Hermitian and non-negative.
"""
function fidelity(ρ::ITensor{N}, σ::ITensor{N}) where {N}
  ρ ./= tr(ρ)
  σ ./= tr(σ)
  F = product(product(sqrt(ρ), σ), sqrt(ρ))
  F = real(tr(sqrt(F)))^2
  return F
end

