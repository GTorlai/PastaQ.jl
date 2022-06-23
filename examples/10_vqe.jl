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

# define the Hamiltonian
os = OpSum()
for j in 1:(N - 1)
  os .+= (-J, "Z", j, "Z", j + 1)
  os .+= (-h, "X", j)
end
os .+= (-h, "X", N)

# build MPO "cost function"
H = MPO(os, hilbert)
# find ground state with DMRG

sweeps = Sweeps(10)
maxdim!(sweeps, 10, 20, 30, 50, 100)
cutoff!(sweeps, 1E-10)
Edmrg, Φ = dmrg(H, randomMPS(hilbert), sweeps; outputlevel=0);
@printf("\nGround state energy: %.10f\n\n", Edmrg)

#Edmrg = -9.7655034665
#@printf("Exact energy from DMRG: %.8f\n", Edmrg)

# layer of single-qubit Ry gates
Rylayer(N, θ) = [("Ry", j, (θ=θ[j],)) for j in 1:N]

# brick-layer of CX gates
function CXlayer(N, Π)
  return if isodd(Π)
    [("CX", (j, j + 1)) for j in 1:2:(N - 1)]
  else
    [("CX", (j, j + 1)) for j in 2:2:(N - 1)]
  end
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

# cost function
function loss(θ⃗)
  circuit = variationalcircuit(N, depth, θ⃗)
  Uψ = runcircuit(ψ, circuit; cutoff=1e-8)
  return inner(Uψ', H, Uψ)
end

# initialize parameters
θ⃗₀ = [2π .* rand(N) for _ in 1:depth]

# run VQE using BFGS optimization
optimizer = LBFGS(; maxiter=200, verbosity=2)
loss_n_grad(x) = (loss(x), convert(Vector, loss'(x)))
θ⃗, fs, gs, niter, normgradhistory = optimize(loss_n_grad, θ⃗₀, optimizer)
@printf("Relative error: %.3E", abs(Edmrg - fs[end]) / abs(Edmrg))
