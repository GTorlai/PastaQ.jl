"""
    fidelity(ψ::MPS,ϕ::MPS)

Compute the fidelity between two MPS:

`F = |⟨ψ|ϕ⟩|²`
"""
function fidelity(ψ::MPS,ϕ::MPS)
  log_F̃ = 2.0*real(loginner(ψ,ϕ))
  log_K = 2.0 * (lognorm(ψ) + lognorm(ϕ))
  fidelity = exp(log_F̃ - log_K)
  return fidelity
end

"""
    fidelity(ψ::MPS, ρ::LPDO)

    fidelity(ρ::LPDO, ψ::MPS)

Compute the fidelity between an MPS and LPDO.

`F = ⟨ψ|ρ|ψ⟩`
"""
function fidelity(ψ::MPS, L::LPDO)
  if L.X isa MPS
    return fidelity(ψ,L.X)
  else
    ρ = L.X
    A = *(ρ,ψ,method="densitymatrix",cutoff=1e-10)
    log_F̃ = log(abs(inner(A,A)))
    #log_F̃ = log(abs(inner(ρ,ψ,ρ,ψ)))
    log_K = 2.0*(lognorm(ψ) + lognorm(ρ))
    return exp(log_F̃ - log_K)#fidelity
  end
end

fidelity(ρ::LPDO, ψ::MPS) = fidelity(ψ, ρ)

"""
    fidelity(ψ::MPS,ρ::MPO)

    fidelity(ρ::MPO,ψ::MPS)

Compute the fidelity between an MPS and MPO.

`F = ⟨ψ|ρ|ψ⟩`
"""
function fidelity(ψ::MPS, ρ::MPO)
  log_F̃ = log(abs(inner(ψ,ρ,ψ)))
  log_K = 2.0*lognorm(ψ) + log(tr(ρ)) 
  fidelity = exp(log_F̃ - log_K)
  return fidelity
end

fidelity(ρ::MPO, ψ::MPS) = fidelity(ψ, ρ)

"""
    frobenius_distance(ρ0::LPDO, σ0::LPDO)
    frobenius_distance(ρ0::LPDO, σ0::MPO)
    frobenius_distance(ρ0::MPO,  σ0::LPDO)
    frobenius_distance(ρ0::MPO,  σ0::MPO)

Compute the trace norm of the difference between two LPDOs and MPOs.

`T(ρ,σ) = sqrt(trace[(ρ-σ)†(ρ-σ)])`
"""
function frobenius_distance(ρ0::LPDO, σ0::LPDO)
  ρ = copy(ρ0)
  σ = copy(σ0)
  # Normalize both LPDO to 1
  normalize!(ρ)
  normalize!(σ)
  # Extract density operators MPO
  ρ′ = MPO(ρ)
  σ′ = MPO(σ)
  Kρ  = 1.0
  Kσ  = 1.0

  distance  = inner(ρ′,ρ′)/Kρ^2
  distance += inner(σ′,σ′)/Kσ^2
  distance -= 2.0*inner(ρ′,σ′)/(Kρ*Kσ)
  
  return real(sqrt(distance))
end

function frobenius_distance(ρ0::LPDO, σ0::MPO)
  # Normalize the LPDO to 1
  ρ = copy(ρ0)
  normalize!(ρ)
  ρ′ = MPO(ρ)
  σ′ = σ0
  # Get the MPO normalization
  Kρ = 1.0
  Kσ = tr(σ′)

  distance  = inner(ρ′,ρ′)/Kρ^2
  distance += inner(σ′,σ′)/Kσ^2
  distance -= 2.0*inner(ρ′,σ′)/(Kρ*Kσ)
  
  return real(sqrt(distance))
end

function frobenius_distance(ρ0::MPO, σ0::LPDO)
  # Normalize the LPDO to 1
  σ = copy(σ0)
  normalize!(σ)
  σ′ = MPO(σ)
  ρ′ = ρ0
  # Get the MPO normalization
  Kρ = tr(ρ′)
  Kσ = 1.0

  distance  = inner(ρ′,ρ′)/Kρ^2
  distance += inner(σ′,σ′)/Kσ^2
  distance -= 2.0*inner(ρ′,σ′)/(Kρ*Kσ)
  
  return real(sqrt(distance))
end

function frobenius_distance(ρ0::MPO, σ0::MPO)
  ρ′ = ρ0
  σ′ = σ0
  Kρ = tr(ρ′)
  Kσ = tr(σ′)
  
  distance  = inner(ρ′,ρ′)/Kρ^2
  distance += inner(σ′,σ′)/Kσ^2
  distance -= 2.0*inner(ρ′,σ′)/(Kρ*Kσ)
  
  return real(sqrt(distance))
end

"""
    fidelity_bound(ρ0::LPDO, σ0::LPDO)
    fidelity_bound(ρ0::LPDO, σ0::MPO)
    fidelity_bound(ρ0::MPO,  σ0::LPDO)
    fidelity_bound(ρ0::MPO,  σ0::MPO)

Compute the the following fidelity bound

`F̃(ρ,σ) = trace[ρ† σ)]`

The bound becomes tight when the target state is nearly pure.
"""
function fidelity_bound(ρ0::LPDO, σ0::LPDO)
  ρ = copy(ρ0)
  σ = copy(σ0)
  # Normalize both LPDO to 1
  normalize!(ρ)
  normalize!(σ)
  # Extract density operators MPO
  ρ′ = MPO(ρ)
  σ′ = MPO(σ)
  Kρ  = 1.0
  Kσ  = 1.0
  return real(inner(ρ′,σ′)/(Kρ*Kσ))
end

function fidelity_bound(ρ0::LPDO, σ0::MPO)
  # Normalize the LPDO to 1
  ρ = copy(ρ0)
  normalize!(ρ)
  ρ′ = MPO(ρ)
  σ′ = σ0
  # Get the MPO normalization
  Kρ = 1.0
  Kσ = tr(σ′)

  return real(inner(ρ′,σ′)/(Kρ*Kσ))
end

function fidelity_bound(ρ0::MPO, σ0::LPDO)
  # Normalize the LPDO to 1
  σ = copy(σ0)
  normalize!(σ)
  σ′ = MPO(σ)
  ρ′ = ρ0
  # Get the MPO normalization
  Kρ = tr(ρ′)
  Kσ = 1.0

  return real(inner(ρ′,σ′)/(Kρ*Kσ))
end

function fidelity_bound(ρ0::MPO, σ0::MPO)
  ρ′ = ρ0
  σ′ = σ0
  Kρ = tr(ρ′)
  Kσ = tr(σ′)
  bound = inner(ρ′,σ′)/(Kρ*Kσ)
  return real(bound)
end

"""
    fullfidelity(ρ::MPO,σ::MPO;choi::Bool=false)

Compute the full quantum fidelity between two density operators
by full enumeration.

The MPOs should be Hermitian and non-negative.
"""
function fullfidelity(L::Union{MPO, LPDO}, σ::Union{LPDO, MPO})
  @assert length(L) < 12
  ρ_mat = fullmatrix(L)
  σ_mat = fullmatrix(σ)
  
  ρ_mat ./= tr(ρ_mat)
  σ_mat ./= tr(σ_mat)
  
  F = sqrt(ρ_mat) * σ_mat * sqrt(ρ_mat)
  F = real(tr(sqrt(F)))
  return F
end

