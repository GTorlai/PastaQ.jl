import torch
import sys
import os

# Adjust path to import from src
# This assumes the script is run from the 'examples' directory.
# If run from the root 'pytorch_pastaq' directory, 'src.' imports would work directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.states import QuantumStateVector
from src.circuits import QuantumCircuit
from src.gates import H, CNOT
from src.simulation import CircuitSimulator

def main():
    """Demonstrates basic quantum circuit simulation for state vectors,
    creating a Bell state |Φ+>."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Basic Quantum Circuit Simulation (State Vector) ---")
    print(f"Using device: {device}\n")

    # 1. Initialize a 2-qubit state vector in the |00> state
    num_qubits = 2
    # QuantumStateVector constructor handles device placement
    initial_psi = QuantumStateVector(num_qubits=num_qubits, device=device)
    print(f"Initial state |00> on device '{initial_psi.device}':")
    print(f"{initial_psi.state_vector}\n") # Use .state_vector

    # 2. Create a quantum circuit
    circuit = QuantumCircuit(num_qubits=num_qubits)
    print("Quantum Circuit:")
    print(f"- Total number of qubits: {circuit.num_qubits}")

    # 3. Add gates to create a Bell state |Φ+> = (|00> + |11>)/√2
    # Apply H on qubit 0
    circuit.add_gate(H(), 0)
    print(f"- Added H gate on qubit 0.")

    # Apply CNOT with control qubit 0, target qubit 1
    circuit.add_gate(CNOT(), (0, 1))
    print(f"- Added CNOT gate with control q0, target q1.")

    print("\nCircuit definition (internal representation):")
    print(repr(circuit))
    print("\n")

    # 4. Initialize the circuit simulator
    # The simulator will use the same device as specified here.
    simulator = CircuitSimulator(device=device)
    print(f"Initialized CircuitSimulator on device '{simulator.device}'.\n")

    # 5. Run the simulation
    print("Running simulation...")
    # The run method handles moving states/gates to the simulator's device if necessary.
    final_psi = simulator.run(circuit, initial_psi)
    print("Simulation complete.\n")

    # 6. Print the final state vector
    print("Final state vector (expected Bell state |Φ+>):")
    print(final_psi.state_vector) # Use .state_vector

    # 7. Verify (optional, for script's self-check)
    # Ensure the expected tensor is on the same device and dtype as the output state
    expected_bell_tensor = torch.tensor(
        [1/torch.sqrt(torch.tensor(2.0)), 0, 0, 1/torch.sqrt(torch.tensor(2.0))],
        dtype=final_psi.state_vector.dtype, # Match dtype of the result
        device=final_psi.device             # Match device of the result
    )

    if torch.allclose(final_psi.state_vector, expected_bell_tensor, atol=1e-7):
        print("\nVerification: SUCCESS - Output matches expected Bell state |Φ+>.")
    else:
        print("\nVerification: FAILED - Output does not match expected Bell state |Φ+>.")
        print(f"Expected tensor:\n{expected_bell_tensor}")

if __name__ == '__main__':
    main()
