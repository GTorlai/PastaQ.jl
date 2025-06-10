import torch
import sys
import os

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.states import DensityMatrix, QuantumStateVector
from src.tomography import (
    get_basis_projectors_single_qubit,
    simulate_measurement_probabilities,
    qst_linear_inversion_single_qubit
)
from src.gates import H, S, COMPLEX_DTYPE as SRC_COMPLEX_DTYPE # For creating interesting states

# Use a consistent COMPLEX_DTYPE, preferably from a central source like gates or states
COMPLEX_DTYPE = SRC_COMPLEX_DTYPE

def main():
    """Demonstrates single-qubit Quantum State Tomography (QST) via linear inversion."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Single-Qubit Quantum State Tomography Example ---")
    print(f"Using device: {device}\n")

    # 1. Define a target single-qubit state to be reconstructed.
    # We will create the state |+i> = S H |0>
    # |0> --H--> |+> --S--> S|+> = 1/sqrt(2) * (S|0> + S|1>) = 1/sqrt(2) * (|0> + i|1>)

    psi_0_vec = torch.tensor([1, 0], dtype=COMPLEX_DTYPE, device=device)

    # Apply H gate
    h_gate_matrix = H().to(device=device, dtype=COMPLEX_DTYPE)
    psi_plus_vec = h_gate_matrix @ psi_0_vec

    # Apply S gate
    s_gate_matrix = S().to(device=device, dtype=COMPLEX_DTYPE)
    psi_target_vec = s_gate_matrix @ psi_plus_vec

    # Create the target DensityMatrix object
    target_dm_tensor = torch.outer(psi_target_vec, psi_target_vec.conj())
    target_dm = DensityMatrix(num_qubits=1, initial_density_matrix_tensor=target_dm_tensor, device=device)

    print(f"Original Density Matrix (target state S H |0> = |+i>):")
    print(target_dm.density_matrix)
    print("\n")

    # 2. Simulate measurements to get ideal outcome probabilities for X, Y, Z bases.
    measurement_data = {}
    print("Simulating ideal measurement outcomes:")
    for basis_name in ["X", "Y", "Z"]:
        # get_basis_projectors_single_qubit already returns a Tuple[torch.Tensor, torch.Tensor]
        projectors_tuple = get_basis_projectors_single_qubit(basis_name, device=device)

        # Ensure target_dm.density_matrix is on the same device as projectors
        # (It should be if target_dm was created with `device`)
        probs = simulate_measurement_probabilities(target_dm.density_matrix, projectors_tuple)
        measurement_data[basis_name] = probs
        print(f"  Probabilities for {basis_name}-basis measurements: {probs}")

    print(f"\nSimulated measurement data dict for QST:\n{measurement_data}\n")

    # 3. Reconstruct the density matrix using QST linear inversion.
    print("Reconstructing density matrix using qst_linear_inversion_single_qubit...")
    reconstructed_dm = qst_linear_inversion_single_qubit(measurement_data, device=device)

    print("\nReconstructed Density Matrix:")
    print(reconstructed_dm.density_matrix)
    print("\n")

    # 4. Verify the reconstructed DM is close to the original.
    # Note: Using a slightly higher atol for QST due to potential accumulation of float errors
    # from probability calculations and then reconstruction formula.
    if torch.allclose(target_dm.density_matrix, reconstructed_dm.density_matrix, atol=1e-6, rtol=1e-5):
        print("Verification: SUCCESS - Reconstructed DM closely matches the original DM.")
    else:
        print("Verification: FAILED - Reconstructed DM does not closely match the original DM.")
        difference = torch.abs(target_dm.density_matrix - reconstructed_dm.density_matrix)
        print(f"Difference matrix (absolute values):\n{difference}")
        print(f"Maximum absolute difference: {torch.max(difference).item()}")

if __name__ == '__main__':
    main()
