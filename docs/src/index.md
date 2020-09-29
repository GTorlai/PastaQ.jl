![alt text](assets/logo.png)
[![Tests](https://github.com/GTorlai/PastaQ.jl/workflows/Tests/badge.svg)](https://github.com/GTorlai/PastaQ.jl/actions?query=workflow%3ATests)
[![codecov](https://codecov.io/gh/GTorlai/PastaQ.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/GTorlai/PastaQ.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://gtorlai.github.io/PastaQ.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://gtorlai.github.io/PastaQ.jl/dev/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![arXiv](https://img.shields.io/badge/arXiv--b31b1b.svg)](https://arxiv.org/abs/)

PLEASE NOTE THIS IS PRE-RELEASE SOFTWARE      

EXPECT ROUGH EDGES AND BACKWARD INCOMPATIBLE UPDATES

# A Package for Simulation, Tomography and Analysis of Quantum Computers

PastaQ is a julia package for simulation and benchmarking of quantum computers using a combination
of machine learning and tensor-network algorithms.

The main features of PastaQ are:
+ **Simulation of quantum circuits**. The package provides a simulator based on Matrix Product States (MPS) to simulate quantum circuits compiled into a set of quantum gates. Noisy circuits are simulated by specifying a noise model of interest, which is applied to each quantum gate.
+ **Quantum state tomography**. Data-driven reconstruction of an unknown quantum wavefunction or density operators, learned respectively with an MPS and a Locally-Purified Density Operator (LPDO). The reconstruction can be certified by fidelity measurements with the target quantum state (if known, and if it admits an efficient tensor-network representation).
+ **Quantum process tomography**. Data-driven reconstruction of an unknown quantum channel, characterized in terms of its Choi matrix (using a similar approach to quantum state tomography). The channel can be unitary (i.e. rank-1 Choi matrix) or noisy.

PastaQ is developed at the Center for Computational Quantum Physics of the Flatiron Institute,
and it is supported by the Simons Foundation.

## Installation
The PastatQ package can be installed with the Julia package manager. From the Julia REPL,
type ] to enter the Pkg REPL mode and run:

```
~ julia
```

```julia
julia> ]

pkg> add github.com/GTorlai/PastaQ.jl
```

Please note that right now, PastaQ.jl requires that you use Julia v1.4 or later.

## Documentation

- [**STABLE**](https://gtorlai.github.io/PastaQ.jl/stable/) --  **documentation of the most recently tagged version.**
- [**DEVEL**](https://gtorlai.github.io/PastaQ.jl/dev/) -- *documentation of the in-development version.*

## Code Overview
The algorithms implemented in PastaQ rely on a tensor-network representation of
quantum states, quantum circuits and quantum channels, which is provided by the
ITensor package.

### Simulation of quantum circuits
A quantum circuit is built out of a collection of elementary quantum gates. In
PastaQ, a quantum gate is described by a data structure `g = ("gatename",sites,params)`
consisting of a `gatename` string identifying a particular gate, a set of `sites`
identifying which qubits the gate acts on, and a set of gate parameters `params`
(e.g. angles of qubit rotations). A comprehensive set of gates is provided,
including Pauli matrices, phase and T gates, single-qubit rotations, controlled
gates, Toffoli gate and others. Additional user-specific gates can be added. Once
a set of gates is specified, the output quantum state (represented as an MPS) is
obtained with the `runcircuit` function.

```julia
using PastaQ

N = 4   # Number of qubits

# Building a circuit data-structure
gates = [("X" , 1),                        # Pauli X on qubit 1
         ("CX", (1, 3)),                   # Controlled-X on qubits [1,3]
         ("Rx", 2, (θ=0.5,)),              # Rotation of θ around X
         ("Rn", 3, (θ=0.5, ϕ=0.2, λ=1.2)), # Arbitrary rotation with angles (θ,ϕ,λ)
         ("√SWAP", (3, 4)),                # Sqrt Swap on qubits [2,3]
         ("T" , 4)]                        # T gate on qubit 4

# Returns the MPS at the output of the quantum circuit: `|ψ⟩ = Û|0,0,…,0⟩`
# First the gate ("X" , 1) is applied, then ("CX", (1, 3)), etc.
ψ = runcircuit(N, gates)
# This is equivalent to:
# julia> ψ0 = qubits(N) # Initialize |ψ⟩ to |0,0,…⟩
# julia> ψ = runcircuit(ψ0,gates) # Run the circuit
```

The unitary circuit can be approximated by a MPO, running the `runcircuit`
function with the flag `process=true`. Below is an example for a random
quantum circuit.

![alt text](assets/runcircuit_unitary.jpg)

```julia
using PastaQ

# Example 1a: random quantum circuit

N = 4     # Number of qubits
depth = 4 # Depth of the circuit

# Generate a random quantum circuit built out of layers of single-qubit random
# rotations + CX gates, alternating between even and of odd layers.
gates = randomcircuit(N, depth)

@show gates

# Returns the MPS at the output of the quantum circuit: `|ψ⟩ = Û|0,0,…,0⟩`
ψ = runcircuit(N, gates)

# Generate the MPO for the unitary circuit:
U = runcircuit(N, gates; process=true)
```

#### Noisy gates

If a noise model is provided, a local noise channel is applied after each quantum
gate. A noise model is described by a string identifying a set of
Kraus operators, which can depend on a set of additional parameters. The `runcircuit`
function in this setting returns the MPO for the output mixed density operator. 
The full quantum channel has several (and equivalent) mathematical representations.
Here we focus on the Choi matrix, which is obtained by applying a given channel `ε`
to half of N pairs of maximally entangled states.

![alt text](assets/runcircuit_noisy.jpg)

```julia
using PastaQ

# Example 1b: noisy quantum circuit

N = 4     # Number of qubits
depth = 4 # Depth of the quantum circuit
gates = randomcircuit(N, depth) # random circuit

# Run the circuit using an amplitude damping channel with decay rate `γ=0.01`.
# Returns the MPO for the mixed density operator `ρ = ε(|0,0,…⟩⟨0,0,̇…|), where
# `ε` is the quantum channel.
ρ = runcircuit(N, gates; noise = ("amplitude_damping", (γ = 0.01,))

# Compute the Choi matrix of the channel
Λ = runcircuit(N, gates; process = true, noise = ("amplitude_damping", (γ = 0.01,))
```


### Generation of projective measurements
For a given quantum circuit, with or without noise, different flavors of measurement
data can be obtained with the function `getsamples(...)` If one is interested in
the quantum state at the output of the circuit, the function carries out a set of
projective measurements in arbitrary local bases. By default, each qubit is measured
randomly in the bases corresponding to the Pauli matrices. The output quantum state,
given as an MPS wavefunction or MPO density operators for unitary and noisy circuits
respectively, is also returned with the data.

```julia
using PastaQ

# Example 2: generation of measurement data

# Set parameters
N = 4                           # Number of qubits
depth = 4                       # Depth of random circuit
nshots = 1000                   # Number of measurements
gates = randomcircuit(N, depth) # Build gates


# 2a) Output state of a noiseless circuit. By default, each projective measurement
# is taken in basis randomly drawn from the the Pauli group. Also returns the output MPS.
data, ψ = getsamples(N, gates, nshots)

#  Note: the above is equivalent to:
# > bases = randombases(N, nshots; localbasis = ["X","Y","Z"])
# > ψ = runcircuit(N, gates)
# > data = getsamples(ψ, bases)

# 2b) Output state of a noisy circuit. Also returns the output MPO
data, ρ = generatedata(N, gates, nshots; noise = ("amplitude_damping", (γ = 0.01,)))

For quantum process tomography of a unitary or noisy circuit, the measurement data
consists of pairs of input and output states to the channel. Each input state is a
product state of random single-qubit states. Be default, these are set to the six
eigenstates of the Pauli matrices (an overcomplete basis). The output states are
projective measurements for a set of different local bases. It returns the MPO
unitary circuit (noiseless) or the Choi matrix (noisy).

```julia
# 2c) Generate data for quantum process tomography, consisting of input states
# (data_in) to a quantum channel, and the corresponding projective measurements
# at the output. By defaul, the states prepared at the inputs are selected from
# product states of eigenstates of Pauli operators, while measurements bases are
# sampled from the Pauli group.

# Unitary channel, returns the MPO unitary circuit
data_in, data_out, U = getsamples(N, gates, nshots; process=true)

# Noisy channel, returns the Choi matrix
data_in, data_out, Λ = generatedata(N, gates, nshots; process = true, noise = ("amplitude_damping", (γ = 0.01,)))
```


### Quantum tomography


#### State tomography
![alt text](assets/quantumtomography.jpg)

#### Process tomography

## Full code examples

#### Quantum state tomography
Quantum tomography of the wavefunction at the output of a 10-qubit random quantum
circuit of depth 10, using 20000 training measurements.

```julia
using PastaQ
using Random

Random.seed!(1234)
N = 10
depth = 10
nshots = 20000
gates = randomcircuit(N, depth)

data,Ψ = getsamples(N, gates, nshots)

ψ0 = randomstate(Ψ; χ=maxlinkdim(Ψ))
opt = SGD(η = 0.01)
ψ = tomography(ψ0,data,opt;target=Ψ,epochs=10)

# Output

  Ep = 1   Loss = 7.56136E+00  Fidelity = 1.048E-01  Time = 3.640 sec
  Ep = 2   Loss = 7.15703E+00  Fidelity = 4.390E-01  Time = 3.583 sec
  Ep = 3   Loss = 6.87236E+00  Fidelity = 6.315E-01  Time = 3.734 sec
  Ep = 4   Loss = 6.70552E+00  Fidelity = 7.743E-01  Time = 3.712 sec
  Ep = 5   Loss = 6.54243E+00  Fidelity = 8.746E-01  Time = 3.645 sec
  Ep = 6   Loss = 6.48293E+00  Fidelity = 9.112E-01  Time = 3.691 sec
  Ep = 7   Loss = 6.44810E+00  Fidelity = 9.284E-01  Time = 3.647 sec
  Ep = 8   Loss = 6.43632E+00  Fidelity = 9.373E-01  Time = 3.603 sec
  Ep = 9   Loss = 6.42528E+00  Fidelity = 9.448E-01  Time = 3.559 sec
  Ep = 10  Loss = 6.41982E+00  Fidelity = 9.450E-01  Time = 3.601 sec
  Ep = 11  Loss = 6.41674E+00  Fidelity = 9.511E-01  Time = 3.557 sec
  Ep = 12  Loss = 6.41443E+00  Fidelity = 9.554E-01  Time = 3.700 sec
  Ep = 13  Loss = 6.41014E+00  Fidelity = 9.553E-01  Time = 3.717 sec
  Ep = 14  Loss = 6.40734E+00  Fidelity = 9.538E-01  Time = 3.643 sec
  Ep = 15  Loss = 6.40643E+00  Fidelity = 9.529E-01  Time = 3.782 sec
  Ep = 16  Loss = 6.40799E+00  Fidelity = 9.565E-01  Time = 3.623 sec
  Ep = 17  Loss = 6.40472E+00  Fidelity = 9.542E-01  Time = 3.623 sec
  Ep = 18  Loss = 6.40551E+00  Fidelity = 9.565E-01  Time = 3.660 sec
  Ep = 19  Loss = 6.40192E+00  Fidelity = 9.564E-01  Time = 3.578 sec
  Ep = 20  Loss = 6.40119E+00  Fidelity = 9.603E-01  Time = 3.597 sec

```
#### Quantum process tomography
