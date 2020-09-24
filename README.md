![alt text](https://github.com/GTorlai/PastaQ.jl/blob/master/docs/src/assets/logo.jpg?raw=true)

PLEASE NOTE THIS IS PRE-RELEASE SOFTWARE 

EXPECT ROUGH EDGES AND BACKWARD INCOMPATIBLE UPDATES

# A Package for Simulation, Tomography and Analysis of Quantum Computers

PastaQ is a julia package for simulation and benchmarking of quantum computers using a combination of machine learning and tensor-network algorithms.

The main features of PastaQ are:
+ **Simulation of quantum circuits**. The package provides a simulator based on Matrix Product States (MPS) to simulate quantum circuits compiled into a set of quantum gates. Noisy circuits are simulated by specifying a noise model, which is applied to each quantum gate. We show as examples two quantum algorithms: the quantum Fourier transforrm and the variational quantum eigensolver.
+ **Quantum state tomography**. This module implements the reconstruction of an unknown quantum state from measurement data. Depending whether the quantum state is pure or mixed, the variational ansatz implemented is an MPS or a locally-purified density operators (LPDO). The measurement data consists of a set of measurement bases and the corresponding bit-strings of the measurement outcome. The reconstruction is realized by minimizing a statistical divergence between the data and the tensor-network probability distributions.
+ **Quantum process tomography**. This module implements the reconstruction of an unknown quantum channel from measurements.

PastaQ is developed at the Center for Computational Quantum Physics of the Flatiron Institute, and it is supported by the Simons Foundation.

## Code overview

### Quantum circuit simulator
A quantum gate is described by a data structure `g = ("gatename",sites,params)` consisting of a `gatename` string identifying a particular gate, a set of `sites` identifying which qubits the gate acts on, and a set of gate parameters `params`. A comprehensive set of gates is provided, including Pauli matrices, phase and T gates, single-qubit rotations, controlled gates, and others. Additional user-specific gates can be added, if needed.

```julia
using PastaQ
# Quantum gates
gates = [("X" , 1),                        # Pauli X on qubit 1
         ("CX", (1, 2)),                   # Controlled-X on qubits [1,2]
         ("Rx", 2, (θ=0.5,)),              # Rotation of θ around X
         ("Rn", 3, (θ=0.5, ϕ=0.2, λ=1.2)), # Arbitrady rotation
         ("√SWAP", (3, 4)],                # Sqrt Swap on qubits [2,3]
         ("T" , 4),                        # T gate on qubit 4
```

For the case of a noiseless quantum circuit, the output quantum state is obtained by contraing each quantum gate in `gates` with an initial state, which can be chosen to be either a wavefunction or a density matrix (using the `mixed=true` flag), parametrized respectively with a matrix product state (MPS) and a matrix product operators (MPO). By default, the initial state is set to the |000...> state. The output state is obtained with the `runcircuit` function. This contains an intermediate compilation step, where each gate in `gates` is transformed into an ITensor.

```julia
N = 20                     # Number of qubits
ψ0 = qubits(N)             # Initialize qubits
ψ = runcircuit(ψ0, gates)  # Run

ρ0 = qubits(N; mixed=true)
ρ = runcircuit(ρ0, gates)
```

It is also possible to add a noise model, defined in terms of its kraus operators (e.g. depolarizing channel, amplitude damping channel, etc). Similar to regular gates, each noise model is characterized by a string identified `noisename` and a set of parameters. In the following example, we first generate a list of quantum gates for a 1D random quantum circuit, and then add a depolarizing channel with some probability. When running the circuit, the channel is applied after each gate (to the qubits involved in the gate). In this case, if the initial state is an MPS, it is automatically converted to an MPO.

```julia
N = 20                                       # Number of qubits
depth = 10                                   # Circuit's depth
gates = randomcircuit(N, depth)              # Generate gates
ψ0 = qubits(N)                               # Initialize qubits
ρ = runcircuit(ψ0, gates;noise="DEP",p=0.1)  # Run
```

