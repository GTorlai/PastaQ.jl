![alt text](https://github.com/GTorlai/PastaQ.jl/blob/master/docs/src/assets/logo.png?raw=true)
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
gates, Toffoli gate and others. Additional user-specific gates can be added.

```julia
# Building a circuit data-structure
gates = [("X" , 1),                        # Pauli X on qubit 1
         ("CX", (1, 2)),                   # Controlled-X on qubits [1,2]
         ("Rx", 2, (θ=0.5,)),              # Rotation of θ around X
         ("Rn", 3, (θ=0.5, ϕ=0.2, λ=1.2)), # Arbitrady rotation with angles (θ,ϕ,λ)
         ("√SWAP", (3, 4)],                # Sqrt Swap on qubits [2,3]
         ("T" , 4)]                        # T gate on qubit 4
```

For the case of a noiseless circuit, the output quantum state (MPS) and the
unitary circuit (MPO) can be obtained with the `runcircuit` function.

```julia
using PastaQ

# Example 1a: random quantum circuit

N = 4     # Number of qubits
depth = 4 # Depth of the circuit

# Generate a random quantum circuit built out of layers of single-qubit random
# rotations + CX gates, alternating between even and of odd layers.
gates = randomcircuit(N,depth)

# Returns the MPS at the output of the quantum circuit: `|ψ⟩ = Û|0,0,…,0⟩`
ψ = runcircuit(N,gates)

# Generate the MPO for the unitary circuit:
U = runcircuit(N,gates; process=true)
```

If a noise model is provided, a local noise channel is applied after each quantum
gate. A noise model is described by a string `noisename` identifying a set of
Kraus operators, which can depend on a set of additional parameters `params`.

```julia
using PastaQ

# Example 1b: noisy quantum circuit

N = 4   # Number of qubits
depth=4 # Depth of the quantum circuit
gates = randomcircuit(N,depth) # random circuit

# Run the circuit using an amplitude damping channel with decay rate `γ=0.01`.
# Returns the MPO for the mixed density operator `ρ = ε(|0,0,…⟩⟨0,0,̇…|), where
# `ε` is the quantum channel.
ρ = runcircuit(N,gates; noise="AD", γ=0.01)
```

#### Choi matrix

The Choi matrix provides a complete description of an arbitrary quantum channel.
It is obtained by applying a given channel `ε` to half of N pairs of entangled states.
If the channel `ε` is unitary, the Choi matrix has rank 1 `Λ = |U⟩⟩⟨⟨U|`, where
`U` is the unitary circuit and `|U⟩⟩` is an MPS obtained by bending the inpupt wires
of the circuit MPO. If the channel is noisy, the Choi matrix is described by a MPO.

```julia
using PastaQ

# Example 1c: choi matrix

N = 4   # Number of qubits
depth=4 # Depth of the quantum circuit
gates = randomcircuit(N,depth) # random circuit

# Compute MPS for rank-1 Choi matrix of a unitary channel
|U⟩⟩ = choimatrix(N,gates)

# Compute the MPO for Choi matrix of a noisy channel
Λ = choimatrix(N,gates; noise="AD", γ=0.01)

```

#### Data generation


### Quantum tomography
