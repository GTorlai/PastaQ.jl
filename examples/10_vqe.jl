using PastaQ
using ITensors
using Random
using Printf
using OptimKit
using Zygote

N = 10   # number of qubits
J = 1.0  # Ising exchange interaction
h = 0.5  # transverse magnetic field

# Hilbert space
hilbert = qubits(N)

function ising_hamiltonian(N; J, h)
  os = OpSum()
  for j in 1:(N - 1)
    os += -J, "Z", j, "Z", j + 1
  end
  for j in 1:N
    os += -h, "X", j
  end
  return os
end

# define the Hamiltonian
os = ising_hamiltonian(N; J, h)

# build MPO "cost function"
H = MPO(os, hilbert)
# find ground state with DMRG

nsweeps = 10
maxdim = [10, 20, 30, 50, 100]
cutoff = 1e-10
Edmrg, Φ = dmrg(H, randomMPS(hilbert); outputlevel=0, nsweeps, maxdim, cutoff);
@printf("\nGround state energy from DMRG: %.10f\n\n", Edmrg)

# layer of single-qubit Ry gates
Rylayer(N, θ) = [("Ry", j, (θ=θ[j],)) for j in 1:N]

# brick-layer of CX gates
function CXlayer(N, Π)
  start = isodd(Π) ? 1 : 2
  return [("CX", (j, j + 1)) for j in start:2:(N - 1)]
end

# variational ansatz
function variationalcircuit(N, depth, θ⃗)
  circuit = Tuple[]
  for d in 1:depth
    circuit = vcat(circuit, CXlayer(N, d))
    circuit = vcat(circuit, Rylayer(N, θ⃗[d]))
  end
  return circuit
end

depth = 20
ψ = productstate(hilbert)

cutoff = 1e-8
maxdim = 50

# cost function
function loss(θ⃗)
  circuit = variationalcircuit(N, depth, θ⃗)
  Uψ = runcircuit(ψ, circuit; cutoff, maxdim)
  return inner(Uψ', H, Uψ; cutoff, maxdim)
end

Random.seed!(1234)

# initialize parameters
θ⃗₀ = [2π .* rand(N) for _ in 1:depth]

# run VQE using BFGS optimization
optimizer = LBFGS(; maxiter=50, verbosity=2)
function loss_and_grad(x)
  y, (∇,) = withgradient(loss, x)
  return y, ∇
end
θ⃗, fs, gs, niter, normgradhistory = optimize(loss_and_grad, θ⃗₀, optimizer)
@printf("Relative error: %.3E", abs(Edmrg - fs[end]) / abs(Edmrg))
