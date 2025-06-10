# PyTorch-PastaQ

PyTorch-PastaQ is a Python library designed for quantum information processing tasks, leveraging PyTorch for tensor operations and potential GPU acceleration. It provides tools for quantum circuit simulation, modeling noisy quantum systems, performing quantum state tomography, and working with basic tensor network structures like Matrix Product States (MPS).

This library is a PyTorch-based adaptation inspired by features found in quantum simulation frameworks, aiming to offer a flexible and intuitive platform for researchers and developers in quantum computing.

## Core Features

*   **Quantum State Representation**:
    *   `QuantumStateVector`: Represents pure quantum states using 1D PyTorch tensors.
    *   `DensityMatrix`: Represents mixed or pure quantum states using 2D PyTorch tensors.
    *   Support for CPU and CUDA devices.

*   **Flexible Quantum Circuit Construction**:
    *   `QuantumCircuit`: Allows sequential addition of quantum gates and noise channels.
    *   Supports operations on specified qubits within a larger register.

*   **Circuit Simulation**:
    *   `CircuitSimulator`: Simulates quantum circuits.
    *   Handles both state vector evolution (for unitary operations) and density matrix evolution.
    *   Automatically converts state vectors to density matrices when non-unitary (noise) operations are applied.

*   **Noise Modeling**:
    *   Provides common single-qubit noise channels:
        *   Pauli Channel (`pauli_channel`)
        *   Depolarizing Channel (`depolarizing_channel`)
        *   Amplitude Damping Channel (`amplitude_damping_channel`)
        *   Phase Damping Channel (`phase_damping_channel`)
    *   Noise is represented by Kraus operators.

*   **Quantum State Tomography (QST)**:
    *   `qst_linear_inversion_single_qubit`: Reconstructs a single-qubit density matrix from measurement outcome probabilities in X, Y, and Z bases.
    *   `qst_linear_inversion_multi_qubit`: Reconstructs a multi-qubit density matrix from Pauli observable expectation values.
    *   Helper functions for generating measurement projectors and simulating measurement outcomes.

*   **Basic Matrix Product State (MPS) Operations**:
    *   `MPS` class for representing one-dimensional tensor networks.
    *   `MPS.product_state()`: Factory method to create an MPS representing a product state.
    *   `mps.norm_squared()`: Calculates the squared norm of an MPS.
    *   Supports basic MPS property inspection (physical dimensions, bond dimensions).

## Installation / Requirements

This library is built using PyTorch. Key dependencies are listed in the `requirements.txt` file located in the `pytorch_pastaq` directory.

To install dependencies:
```bash
pip install -r requirements.txt
```
Ensure you have a compatible version of PyTorch installed for your system (CPU or GPU). Refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) if needed.

## Running Examples

Example scripts demonstrating various functionalities of the library can be found in the `pytorch_pastaq/examples/` directory.

To run an example, navigate to the root directory of this project (which contains the `pytorch_pastaq` folder) and execute the scripts as standard Python programs. For instance:

```bash
python pytorch_pastaq/examples/01_basic_simulation_qsv.py
python pytorch_pastaq/examples/02_noisy_simulation_dm.py
# and so on...
```
The scripts include a path adjustment to correctly import the library components from the `src` directory.

## Project Structure (Brief Overview)

*   `pytorch_pastaq/src/`: Contains the core library code.
    *   `states.py`: Defines `QuantumStateVector` and `DensityMatrix`.
    *   `gates.py`: Provides common quantum gate matrices.
    *   `circuits.py`: Defines the `QuantumCircuit` class.
    *   `simulation.py`: Contains the `CircuitSimulator`.
    *   `noise.py`: Provides functions for generating Kraus operators for noise channels.
    *   `tomography.py`: Implements QST functions.
    *   `tensor_network.py`: Contains the `MPS` class.
*   `pytorch_pastaq/tests/`: Contains unit tests for the library components.
*   `pytorch_pastaq/examples/`: Contains example scripts showcasing library usage.
*   `pytorch_pastaq/requirements.txt`: Lists project dependencies.
*   `pytorch_pastaq/README.md`: This file.

---

Happy quantum simulating!
