#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
# # Optimal Coherent Control
# 
# Quantum computers perform computations by executing circuits consisting of a 
# set of quantum gates, and recording measurement at its output. At this level
# of abstraction, quantum gates are well-defined mathematical operations on a 
# ``n``-qubit Hilbert space. In practice, different hardware realizations engineer a 
# universal set of gates through ia variety of controllable sets of physical interactions between
# the qubits. Optimal coherent control (OCC) provides is a framework whereby qubit control
# functions can be optimized to produce a desired target quantum gate. 
#
#
#nb # %% a slide [markdown] {"slideshow": {"slide_type": "slide"}}
# In this tutorial example, we consider a very simple system made out of only two qubits, 
# described by the Hamiltonian 
# ```math
# H_0 = \sum_{j=1,2}\omega_j a^\dagger_ja_j + g (a^\dagger_1a_2 + a_1 a^\dagger_2)
# ```
# where ``\{\omega_j\}`` are the frequencies of the transmons and ``g`` is the qubit exchange  
# coupling.
#
# ### Trotter simulation of quantum dynamics 
#
# Before considering the gate optimization, we first need to simulate the dynamics of the system
# generate by the Hamiltonian ``H``. We begin by importing the relevant packages and
# setting up the simulation parameters:
#
#nb %% A slide [code] {"slideshow": {"slide_type": "subslide"}}
using Random
using ITensors
using PastaQ
using Observers
using DataFrames
using Plots

#jl # units  
GHz = 2π
MHz = 1e-3 * GHz
plot_args = (dpi=1000,size=(600,300), margin=5Plots.mm, marker = :circle, markersize = 2,linewidth = 1)

n  = 2            # number of qubits
g  = 12 * MHz     # exchange interaction
ω₁ = 5.0 * GHz    # qubit-1 frequency
ω₂ = 5.0 * GHz    # qubit-2 frequency
ω⃗ = [ω₁, ω₂]

q₁, q₂ = 1, 2         # modes ordering
modes = ["q₁", "q₂"]  # modes labels

#generate the Hilbert space
hilbert = qubits(n)

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "subslide"}}
# Now that we have created the Hilbert space and set the parameters, we
# can define the Hamiltonian. For now, we define it as a `Vector` of `Tuple`, 
# where each `Tuple` represent a term in the Hamiltonian. This would be normally
# defined as an ITensor object (`OpSum`), but that is not yet fully differentiable.
#

#nb %% A slide [code] {"slideshow": {"slide_type": "subslide"}}
function hamiltonian(ω⃗::Vector, g::Number)
  H = Tuple[]
  ω₁, ω₂ = ω⃗
  H = vcat(H, [(ω₁, "a† * a", q₁)])
  H = vcat(H, [(ω₂, "a† * a", q₂)])
  H = vcat(H, [(g,  "a†b + ab†", (q₁, q₂))])
  return H
end

H = hamiltonian(ω⃗, g)

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "subslide"}}
# We would like to simulate the system dynamics and record measurements, such
# as the average mode occupation. We use the `Observers.jl` package to keep
# track of observables. The `Observer` object is a container of a set of `Function`,
# which are called iteratively inside whatever iterative loop we consider.
# We also need to add a `Function` that measure the average occupation here. 
#

#nb %% A slide [code] {"slideshow": {"slide_type": "subslide"}}
function population(ψ::MPS, site::Int)
  s = siteinds(ψ)[site]
  orthogonalize!(ψ, site)
  norm2_ψ = norm(ψ)^2
  val = scalar(ψ[site] * op("a† * a", s) * dag(prime(ψ[site], s))) / norm2_ψ
  return real(val)
end;

#define a vector of observables and create the `Observer`.
observables = ["n($α)" => x -> population(x, k)  # actually x -> expect(x, "a† * a"; sites = k)
               for (k,α) in enumerate(modes)]
obs = Observer(observables)

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "subslide"}}
# We are not ready to simulate the system dynamics using a Trotter expansion.
# The time-evolution propagator up to time ``t`` is decomposed as
# ```math
# U(t) = U(\delta t)^M
# ```
# with ``t_g = M\delta t`` being the finla time. Each elementary propagator is 
# then approximated with its Trotter expansion (to order 2 by default):
# ```math
# U(\delta t) \approx U_K(\delta t)\dots U_2(\delta t) U_1(\delta t)
# ```

#nb %% A slide [code] {"slideshow": {"slide_type": "subslide"}}
tg = 30                  # final time (in ns)
trottersteps = 100       # number of Trotter steps
δt = tg / trottersteps   # step size
ts = 0.0:δt:tg           # time list

#build the Trotter circuit
circuit = trottercircuit(H; ts = ts, layered = true)

#set initial state |ψ⟩ = |1,0⟩
ψ₀ = productstate(hilbert, [1,0])

#perform TEBD simulation and generate output `MPS`
ψ = runcircuit(ψ₀, circuit; (observer!) = obs,
               move_sites_back_before_measurements = true, outputlevel = 0)

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "subslide"}}
# The measurements taken during the dynamics (one at each Trotter layer) are
# store in the Observer and can be retrieved into a `DataFrame` format (for example).
# We plot here the average occupation of the two modes as a function of time:

#nb %% A slide [code] {"slideshow": {"slide_type": "subslide"}}
res = DataFrame(results(obs));
p = plot(xlabel = "time (ns)", ylabel = "n̂(t)", legend = (0.40,0.9); plot_args...)
p = plot!(p, ts, res[!,"n(q₁)"], label = "n(q₁)";  plot_args...)
p = plot!(p, ts, res[!,"n(q₂)"], label = "n(q₂)";  plot_args...)
p


#nb # %% A slide [markdown] {"slideshow": {"slide_type": "subslide"}}
# In this simulation we placed the two qubits on resonance (``\omega_1=\omega_2``).
# By populating one of the qubit with an excitation (qubit 1 above), we observe that 
# the dynamics swaps the excitation between the two qubits at time. In fact, this system
# implements a perfect iSwap gate.
#
# In practice, in idle mode the two qubits are placed at some detuning, and placed on resonance
# only when realizing the gate. Here we consider this setting, and we will optimize the modulation
# of the two qubit frequencies to realize the gate when starting further apart. But first let's just
# re-run the previous dynamical simulation in this setup:
#
#nb %% A slide [code] {"slideshow": {"slide_type": "subslide"}}
ω₁ = 5.0 * GHz
ω₂ = 5.3 * GHz 
ω⃗ = [ω₁, ω₂]

H = hamiltonian(ω⃗, g)

obs = Observer(observables)

circuit = trottercircuit(H; ts = ts, layered = true)

ψ₀ = productstate(hilbert, [1,0])

ψ = runcircuit(ψ₀, circuit; (observer!) = obs,
               move_sites_back_before_measurements = true, outputlevel = 0)

res = DataFrame(results(obs));
p = plot(xlabel = "time (ns)", ylabel = "n̂(t)", legend = (0.50,0.9); plot_args...)
p = plot!(p, ts, res[!,"n(q₁)"], label = "n(q₁)";  plot_args...)
p = plot!(p, ts, res[!,"n(q₂)"], label = "n(q₂)";  plot_args...)
p

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "subslide"}}
# ### Control optimization
# After importing relevant packages for the optimization, we now fix the desired
# gate time to `t_g=25``ns. We also define two control functions:
# ```math
# f_{fourier}(\theta, t) = \Lambda \tanh(\sum_i\theta_i \sin(\pi i t /t_g))
# ```
# and 
# ```math
# f_{pulse}(\theta, t) = \tanh((t - t_{on})/\gamma) - \tanh((t - t_{off})/\gamma)
# ```
#nb %% A slide [code] {"slideshow": {"slide_type": "subslide"}}
using Zygote
using OptimKit
using StatsBase: mean

tg = 25
trottersteps = 100
δt = tg / trottersteps
ts = 0.0:δt:tg

Λ = 20.0 * MHz
fourier_control(ϑ, t) =
  Λ * tanh(sum([ϑ[i] * sin(π * i * t / tg) for i in 1:length(ϑ)]))

function pulse_control(ϑ, t)
  y₀, ypulse, ton, toff, γ = ϑ
  f = tanh((t - ton)/γ) - tanh((t - toff)/γ)
  return y₀ + 0.5 * (ypulse - y₀) * f 
end

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "subslide"}}
# We define the new Hamiltonian as follows. We take qubit 2 and send a pulse
# to bring its frequency near ``\omega_1``. At the same time, we also introduce
# a frequency modulation ``\omega_1(t)`` with small amplitude.
#
#nb %% A slide [code] {"slideshow": {"slide_type": "subslide"}}
function hamiltonian(θ⃗::Vector, ω⃗::Vector, g::Number, t::Float64)
  ω₁, ω₂ = ω⃗
  ϑ₁, ϑ₂ = θ⃗
  H = Tuple[]
  H = vcat(H, [(ω₁ + fourier_control(ϑ₁, t), "a† * a", q₁)])
  H = vcat(H, [(ω₂ + pulse_control(ϑ₂, t), "a† * a", q₂)])
  H = vcat(H, [(g,   "a†b + ab†", (q₁, q₂))])
  return H
end

hamiltonian(θ::Vector, t::Float64) = hamiltonian(θ, ω⃗, g, t);

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "subslide"}}
# Let's see how this look like:
#nb %% A slide [code] {"slideshow": {"slide_type": "subslide"}}
Random.seed!(12345)
Ntones = 8
ϑ₁ = rand(Ntones)
ϑ₂ = [0.0, ω₁ - ω₂, 0.1 * tg, 0.9 * tg, 1]
θ⃗₀ = [ϑ₁, ϑ₂]
p = plot(xlabel = "time (ns)", ylabel = "ωⱼ(t)", title = "", legend = (0.50,0.9); plot_args...)
p = plot!(p, ts, [ω₁ + fourier_control(ϑ₁, t) for t in ts] ./ GHz; label = "ω₁(t)", plot_args...)
p = plot!(p, ts, [ω₂ + pulse_control(ϑ₂, t) for t in ts] ./ GHz; label = "ω₂(t)", plot_args...)
p

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "subslide"}}
# The cost function to our optimization is computed from the inner products between
# the time-evolved and the desired wavefunctions:
# ```math
# C(\theta) = 1 - \frac{1}{D^2}\bigg| \sum_i\langle\phi_i|U|\psi_i\rangle\bigg|^2
# ```
#nb %% A slide [code] {"slideshow": {"slide_type": "subslide"}}
function loss(Ψ⃗, Φ⃗, θ⃗)
  #build sequence Tuple (OpSum) Hamiltonians at different times
  Ht = [hamiltonian(θ⃗, t) for t in ts]
  #Trotter-Suzuki decomposition
  circuit = trottercircuit(Ht; ts = ts)
  #run the circuit
  UΨ⃗ = [runcircuit(ψ, circuit; cutoff = 1e-7) for ψ in Ψ⃗]
  #compute inner product ⟨ϕ|ψ(t)⟩
  ip = [inner(ϕ,ψ) for (ϕ,ψ) in zip(Φ⃗, UΨ⃗)]
  return 1 - abs2(mean(ip)) 
end;

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "subslide"}}
# We not set the ideal gate (Lazy format) and define loss closure.
#nb %% A slide [code] {"slideshow": {"slide_type": "subslide"}}
ideal_gate = [
  [0,0] => (1,   [0,0]),
  [1,0] => (-im, [0,1]),
  [0,1] => (-im, [1,0]),
  [1,1] => (1,   [1,1])
]

Ψ⃗ = [productstate(hilbert, σ) for σ in first.(ideal_gate)]
Φ⃗  = [ϕ * productstate(hilbert, σ) for (ϕ, σ) in last.(ideal_gate)];
loss(θ) = loss(Ψ⃗, Φ⃗, θ)

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "subslide"}}
# We initialize the optimizer the run the optimization for ``20`` steps:
#nb %% A slide [code] {"slideshow": {"slide_type": "subslide"}}
optimizer = LBFGS(verbosity = 2, maxiter = 20)
loss_n_grad(x) = (loss(x), convert(Vector, loss'(x)))
θ⃗, fs, gs, niter, normgradhistory = optimize(loss_n_grad, θ⃗₀, optimizer)
normgradhistory[:,1]

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "subslide"}}
# In markdown sections we can use markdown syntax. For example, we can
Ht = [hamiltonian(θ⃗, t) for t in ts] 
circuit = trottercircuit(Ht; ts = ts, layered = true)
ψ₀ = productstate(hilbert, [1,0])
observables = ["n($α)" => x -> population(x, k) 
               for (k,α) in enumerate(modes)]
obs = Observer(observables)
ψ = runcircuit(ψ₀, circuit; (observer!) = obs,
               move_sites_back_before_measurements = true, outputlevel = 0)
res = DataFrame(results(obs));
p = plot(xlabel = "time (ns)", ylabel = "n̂(t)", legend = (0.50,0.9); plot_args...)
p = plot!(p, ts, res[!,"n(q₁)"], label = "n(q₁)";  plot_args...)
p = plot!(p, ts, res[!,"n(q₂)"], label = "n(q₂)";  plot_args...)
p

