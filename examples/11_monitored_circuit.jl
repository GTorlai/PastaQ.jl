using PastaQ
using ITensors
using Random
using LinearAlgebra
using StatsBase: mean, sem

# define the two measurement projectors
import PastaQ: gate
gate(::GateName"Π0") = 
  [1 0
   0 0]
gate(::GateName"Π1") = 
  [0 0
   0 1]

# Von Neumann entropy at center bond
function entanglemententropy(ψ₀::MPS)
  ψ = normalize!(copy(ψ₀))
  N = length(ψ)
  bond = N ÷ 2
  orthogonalize!(ψ, bond)

  row_inds = (linkind(ψ, bond - 1), siteind(ψ, bond))
  u, s, v = svd(ψ[bond], row_inds)

  S = 0.0
  for n in 1:dim(s, 1)
    λ = s[n, n]^2
    S -= λ * log(λ + 1e-20)
  end
  return S
end

# build a brick-layer of random unitaries covering all 
# nearest-neighbors bonds
function entangling_layer(N::Int)
  layer_odd  = randomlayer("randU",[(j,j+1) for j in 1:2:N-1])
  layer_even = randomlayer("randU",[(j,j+1) for j in 2:2:N-1])
  return [layer_odd..., layer_even...]
end

# perform a projective measurement at a given site
function projective_measurement!(ψ₀::MPS, site::Int)
  ψ = orthogonalize!(ψ₀, site)
  ϕ = ψ[site]
  # 1-qubit reduced density matrix
  ρ = prime(ϕ, tags="Site") * dag(ϕ)
  # Outcome probabilities
  prob = real.(diag(array(ρ)))
  # Sample
  σ = (rand() < prob[1] ? 0 : 1)
  # Projection
  ψ = runcircuit(ψ, ("Π"*"$(σ)", site))
  normalize!(ψ)
  ψ₀[:] = ψ
  return ψ₀
end

let 
  Random.seed!(1234)

  N = 10        # number of qubits
  depth = 100   # circuit's depth
  ntrials = 50  # number of random trials

  # projective measurement probability per site
  plist = [0.0, collect(0.01:0.01:0.2)...]
  
  # define the Hilbert space
  hilbert = qubits(N)
  
  for p in plist
    svn = []
    t = @elapsed begin
      for n in 1:ntrials
        # initialize state ψ = |000…⟩
        ψ = productstate(hilbert)
        for d in 1:depth
          # apply entangling unitary
          layer = entangling_layer(N)
          ψ = runcircuit(ψ, layer; cutoff = 1e-8)
          # perform measurements
          for j in 1:N
            p > rand() && (ψ = projective_measurement!(ψ, j))
          end
        end
        # record Von Neumann entropy
        push!(svn, entanglemententropy(ψ))
      end 
    end 
    @printf("p = %.2f  S(ρ) = %.5f ± %.1E\t(elapsed = %.2fs)\n", p, mean(svn), sem(svn), t)
  end
end
