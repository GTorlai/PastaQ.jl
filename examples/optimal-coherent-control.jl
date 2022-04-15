using Random
using ITensors
using PastaQ
using Observers
using DataFrames
using Plots

GHz = 2π
MHz = 1e-3 * GHz
plot_args = (
  dpi=1000, size=(600, 300), margin=5Plots.mm, marker=:circle, markersize=2, linewidth=1
)

n = 2            # number of qubits
g = 12 * MHz     # exchange interaction
ω₁ = 5.0 * GHz    # qubit-1 frequency
ω₂ = 5.0 * GHz    # qubit-2 frequency
ω⃗ = [ω₁, ω₂]

q₁, q₂ = 1, 2         # modes ordering
modes = ["q₁", "q₂"]  # modes labels

#generate the Hilbert space
hilbert = qubits(n)

function hamiltonian(ω⃗::Vector, g::Number)
  H = Tuple[]
  ω₁, ω₂ = ω⃗
  H = vcat(H, [(ω₁, "a† * a", q₁)])
  H = vcat(H, [(ω₂, "a† * a", q₂)])
  H = vcat(H, [(g, "a†a + aa†", (q₁, q₂))])
  return H
end

H = hamiltonian(ω⃗, g)

function population(ψ::MPS, site::Int)
  s = siteinds(ψ)[site]
  orthogonalize!(ψ, site)
  norm2_ψ = norm(ψ)^2
  val = scalar(ψ[site] * op("a† * a", s) * dag(prime(ψ[site], s))) / norm2_ψ
  return real(val)
end;

#define a vector of observables and create the `Observer`.
observables = ["n($α)" => x -> population(x, k)  # actually x -> expect(x, "a† * a"; sites = k)
               for (k, α) in enumerate(modes)]
obs = Observer(observables)

tg = 30                  # final time (in ns)
trottersteps = 100       # number of Trotter steps
δt = tg / trottersteps   # step size
ts = 0.0:δt:tg           # time list

#build the Trotter circuit
circuit = trottercircuit(H; ts=ts, layered=true)

#set initial state |ψ⟩ = |1,0⟩
ψ₀ = productstate(hilbert, [1, 0])

#perform TEBD simulation and generate output `MPS`
ψ = runcircuit(
  ψ₀, circuit; (observer!)=obs, move_sites_back_before_measurements=true, outputlevel=0
)

res = DataFrame(results(obs));
p = plot(; xlabel="time (ns)", ylabel="n̂(t)", legend=(0.40, 0.9), plot_args...)
p = plot!(p, ts, res[!, "n(q₁)"]; label="n(q₁)", plot_args...)
p = plot!(p, ts, res[!, "n(q₂)"]; label="n(q₂)", plot_args...)
p

ω₁ = 5.0 * GHz
ω₂ = 5.3 * GHz
ω⃗ = [ω₁, ω₂]

H = hamiltonian(ω⃗, g)

obs = Observer(observables)

circuit = trottercircuit(H; ts=ts, layered=true)

ψ₀ = productstate(hilbert, [1, 0])

ψ = runcircuit(
  ψ₀, circuit; (observer!)=obs, move_sites_back_before_measurements=true, outputlevel=0
)

res = DataFrame(results(obs));
p = plot(; xlabel="time (ns)", ylabel="n̂(t)", legend=(0.50, 0.9), plot_args...)
p = plot!(p, ts, res[!, "n(q₁)"]; label="n(q₁)", plot_args...)
p = plot!(p, ts, res[!, "n(q₂)"]; label="n(q₂)", plot_args...)
p

using Zygote
using OptimKit
using StatsBase: mean

tg = 25
trottersteps = 100
δt = tg / trottersteps
ts = 0.0:δt:tg

Λ = 20.0 * MHz
fourier_control(ϑ, t) = Λ * tanh(sum([ϑ[i] * sin(π * i * t / tg) for i in 1:length(ϑ)]))

function pulse_control(ϑ, t)
  y₀, ypulse, ton, toff, γ = ϑ
  f = tanh((t - ton) / γ) - tanh((t - toff) / γ)
  return y₀ + 0.5 * (ypulse - y₀) * f
end

function hamiltonian(θ⃗::Vector, ω⃗::Vector, g::Number, t::Float64)
  ω₁, ω₂ = ω⃗
  ϑ₁, ϑ₂ = θ⃗
  H = Tuple[]
  H = vcat(H, [(ω₁ + fourier_control(ϑ₁, t), "a† * a", q₁)])
  H = vcat(H, [(ω₂ + pulse_control(ϑ₂, t), "a† * a", q₂)])
  H = vcat(H, [(g, "a†a + aa†", (q₁, q₂))])
  return H
end

hamiltonian(θ::Vector, t::Float64) = hamiltonian(θ, ω⃗, g, t);

Random.seed!(12345)
Ntones = 8
ϑ₁ = rand(Ntones)
ϑ₂ = [0.0, ω₁ - ω₂, 0.1 * tg, 0.9 * tg, 1]
θ⃗₀ = [ϑ₁, ϑ₂]
p = plot(; xlabel="time (ns)", ylabel="ωⱼ(t)", title="", legend=(0.50, 0.9), plot_args...)
p = plot!(
  p, ts, [ω₁ + fourier_control(ϑ₁, t) for t in ts] ./ GHz; label="ω₁(t)", plot_args...
)
p = plot!(
  p, ts, [ω₂ + pulse_control(ϑ₂, t) for t in ts] ./ GHz; label="ω₂(t)", plot_args...
)
p

function loss(Ψ⃗, Φ⃗, θ⃗)
  #build sequence Tuple (OpSum) Hamiltonians at different times
  Ht = [hamiltonian(θ⃗, t) for t in ts]
  #Trotter-Suzuki decomposition
  circuit = trottercircuit(Ht; ts=ts)
  #run the circuit
  UΨ⃗ = [runcircuit(ψ, circuit; cutoff=1e-7) for ψ in Ψ⃗]
  #compute inner product ⟨ϕ|ψ(t)⟩
  ip = [inner(ϕ, ψ) for (ϕ, ψ) in zip(Φ⃗, UΨ⃗)]
  return 1 - abs2(mean(ip))
end;

ideal_gate = [
  [0, 0] => (1, [0, 0]),
  [1, 0] => (-im, [0, 1]),
  [0, 1] => (-im, [1, 0]),
  [1, 1] => (1, [1, 1]),
]

Ψ⃗ = [productstate(hilbert, σ) for σ in first.(ideal_gate)]
Φ⃗ = [ϕ * productstate(hilbert, σ) for (ϕ, σ) in last.(ideal_gate)];
loss(θ) = loss(Ψ⃗, Φ⃗, θ)

optimizer = LBFGS(; verbosity=2, maxiter=20)
loss_n_grad(x) = (loss(x), convert(Vector, loss'(x)))
θ⃗, fs, gs, niter, normgradhistory = optimize(loss_n_grad, θ⃗₀, optimizer)
normgradhistory[:, 1]

Ht = [hamiltonian(θ⃗, t) for t in ts]
circuit = trottercircuit(Ht; ts=ts, layered=true)
ψ₀ = productstate(hilbert, [1, 0])
observables = ["n($α)" => x -> population(x, k) for (k, α) in enumerate(modes)]
obs = Observer(observables)
ψ = runcircuit(
  ψ₀, circuit; (observer!)=obs, move_sites_back_before_measurements=true, outputlevel=0
)
res = DataFrame(results(obs));
p = plot(; xlabel="time (ns)", ylabel="n̂(t)", legend=(0.50, 0.9), plot_args...)
p = plot!(p, ts, res[!, "n(q₁)"]; label="n(q₁)", plot_args...)
p = plot!(p, ts, res[!, "n(q₂)"]; label="n(q₂)", plot_args...)
p

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
