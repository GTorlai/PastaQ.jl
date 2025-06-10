import torch
import sys
import os

# Adjust path to import from src
# This assumes the script is run from the 'examples' directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.states import QuantumStateVector, DensityMatrix
from src.circuits import QuantumCircuit
from src.gates import H, X
from src.noise import depolarizing_channel
from src.simulation import CircuitSimulator

def main():
    """Demonstrates quantum circuit simulation with noise, leading to density matrix representation."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Noisy Quantum Circuit Simulation (Density Matrix) ---")
    print(f"Using device: {device}\n")

    # 1. Initialize a 1-qubit state vector in the |0> state
    num_qubits = 1
    initial_psi = QuantumStateVector(num_qubits=num_qubits, device=device)
    print(f"Initial state |0> (QuantumStateVector) on device '{initial_psi.device}':")
    print(f"{initial_psi.state_vector}\n") # Using .state_vector

    # 2. Create a quantum circuit
    circuit = QuantumCircuit(num_qubits=num_qubits)
    print("Quantum Circuit:")
    print(f"- Total number of qubits: {circuit.num_qubits}")

    # 3. Add gates and a noise channel
    # Apply H on qubit 0 to create |+> state
    circuit.add_gate(H(), 0)
    print(f"- Added H gate on qubit 0. State becomes |+>.")

    # Apply Depolarizing channel on qubit 0 with probability p=0.3
    depol_prob = 0.3
    kraus_ops_depol = depolarizing_channel(p=depol_prob)
    circuit.add_kraus(kraus_ops_depol, 0)
    print(f"- Added depolarizing channel (p={depol_prob}) on qubit 0.")
    print("  (At this point, state vector will be converted to density matrix by the simulator).")

    # Optional: Add another gate after noise to see evolution of the mixed state
    circuit.add_gate(X(), 0)
    print(f"- Added X gate on qubit 0 after noise.\n")

    print("Circuit definition (internal representation):")
    print(repr(circuit))
    print("\n")

    # 4. Initialize the circuit simulator
    simulator = CircuitSimulator(device=device)
    print(f"Initialized CircuitSimulator on device '{simulator.device}'.\n")

    # 5. Run the simulation
    print("Running simulation...")
    # The simulator will automatically convert the QuantumStateVector to a DensityMatrix
    # when the Kraus channel (a non-unitary operation) is encountered.
    final_state = simulator.run(circuit, initial_psi)
    print("Simulation complete.\n")

    # 6. Print the final state (should be a DensityMatrix)
    print(f"Final state (type: {type(final_state)}):")
    if isinstance(final_state, DensityMatrix):
        print(final_state.density_matrix) # Using .density_matrix

        # Note: The exact expected matrix can be calculated for verification:
        # 1. Initial state: |0>
        # 2. After H: rho_plus = |+><+| = 0.5 * [[1, 1], [1, 1]]
        # 3. After depolarizing_channel(p=0.3) on rho_plus:
        #    rho_after_noise = (1 - 2*p/3) * rho_plus + (2*p/3) * |-><-|
        #                      = (1 - 0.2) * rho_plus + 0.2 * (0.5 * [[1,-1],[-1,1]])
        #                      = 0.8 * 0.5 * [[1,1],[1,1]] + 0.2 * 0.5 * [[1,-1],[-1,1]]
        #                      = 0.4 * [[1,1],[1,1]] + 0.1 * [[1,-1],[-1,1]]
        #                      = [[0.4,0.4],[0.4,0.4]] + [[0.1,-0.1],[-0.1,0.1]]
        #                      = [[0.5, 0.3],[0.3, 0.5]]
        # 4. After X gate: X @ rho_after_noise @ X_dag
        #    X = [[0,1],[1,0]]. X_dag = X.
        #    X rho X = [[rho_11, rho_10],[rho_01, rho_00]] (swaps diagonal, and off-diagonal elements)
        #    So, expected_final_dm = [[0.5, 0.3],[0.3, 0.5]] (it's symmetric to X transform in this case)
        # For p=0.3: expected_final_dm = [[0.5, 0.3],[0.3, 0.5]]
        # This manual calculation can be added as a self-check if desired.

    elif isinstance(final_state, QuantumStateVector):
        # This case should ideally not be reached if a Kraus channel was applied.
        print(final_state.state_vector)
        print("Error: Final state is a QuantumStateVector, but DensityMatrix was expected after applying noise.")
    else:
        print("Error: Unknown final state type encountered.")

if __name__ == '__main__':
    main()
