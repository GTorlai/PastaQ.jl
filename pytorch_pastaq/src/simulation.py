import torch
from typing import Union, List, Tuple
import math

# Attempt to import from local project structure
try:
    from .circuits import QuantumCircuit
    from .states import QuantumStateVector, DensityMatrix
except ImportError:
    # Fallback for environments where the . notation doesn't work as expected
    from circuits import QuantumCircuit
    from states import QuantumStateVector, DensityMatrix

def _create_permutation_operator(num_qubits: int, qubit_map: List[int], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """
    Internal helper to construct a permutation matrix P.

    This matrix transforms a state vector such that the qubit originally at index
    `qubit_map[j]` is moved to the `j`-th logical position.
    psi_permuted_basis = P @ psi_standard_basis.

    Args:
        num_qubits: Total number of qubits.
        qubit_map: A list where `qubit_map[j]` is the original physical index
                   of the qubit that moves to the `j`-th logical position.
        dtype: PyTorch dtype for the operator.
        device: PyTorch device for the operator.

    Returns:
        A (2**num_qubits, 2**num_qubits) permutation matrix.
    """
    dim = 2**num_qubits
    P = torch.zeros((dim, dim), dtype=dtype, device=device)
    for i in range(dim): # i is the column index (standard basis state index)
        # binary_states_standard_basis[k] is state of physical qubit q_k (q_0 is LSB)
        binary_states_standard_basis = [int(c) for c in reversed(format(i, f'0{num_qubits}b'))]

        permuted_binary_states_logical_basis = [0] * num_qubits
        for k_logical in range(num_qubits):
            physical_qubit_index = qubit_map[k_logical]
            permuted_binary_states_logical_basis[k_logical] = binary_states_standard_basis[physical_qubit_index]

        # Convert permuted_binary_states_logical_basis (which is [logical_s0, logical_s1, ...])
        # back to an integer j_perm. This j_perm is the row index in P.
        j_perm_str = "".join(map(str, reversed(permuted_binary_states_logical_basis)))
        j_perm = int(j_perm_str, 2)

        P[j_perm, i] = 1.0
    return P

def _get_full_operator(op_matrix: torch.Tensor,
                       qubits: Tuple[int, ...],
                       total_qubits: int) -> torch.Tensor:
    """
    Internal helper to construct the full (2**total_qubits, 2**total_qubits) operator
    for a given smaller operator `op_matrix` acting on a subset of `qubits`.

    Args:
        op_matrix: The k-qubit operator matrix (e.g., a gate or a Kraus operator).
                   Shape (2^k, 2^k).
        qubits: Tuple of k qubit indices that `op_matrix` acts upon.
        total_qubits: Total number of qubits in the system.

    Returns:
        The full operator acting on the `total_qubits` Hilbert space.

    Raises:
        ValueError: If `op_matrix` shape is inconsistent with the number of active qubits.
    """
    op_dtype = op_matrix.dtype
    op_device = op_matrix.device
    num_op_qubits = len(qubits)

    if num_op_qubits == 0:
        return torch.eye(2**total_qubits, dtype=op_dtype, device=op_device)

    expected_dim = 2**num_op_qubits
    if op_matrix.shape != (expected_dim, expected_dim):
        raise ValueError(
            f"Operator matrix shape {op_matrix.shape} inconsistent with {num_op_qubits} active qubits {qubits}."
            f" Expected shape ({expected_dim},{expected_dim})."
        )

    # Permutation to bring active qubits to the front (most significant for kron)
    perm_map = list(qubits) + [q for q in range(total_qubits) if q not in qubits]
    P = _create_permutation_operator(total_qubits, perm_map, op_dtype, op_device)

    # Identity for the remaining qubits
    identity_part_dim = 2**(total_qubits - num_op_qubits)
    identity_matrix = torch.eye(identity_part_dim, dtype=op_dtype, device=op_device)

    # Tensor product of op_matrix (for active qubits) and identity (for others)
    op_permuted_basis = torch.kron(op_matrix, identity_matrix)

    # Transform back to standard basis: P_dagger @ op_permuted_basis @ P
    full_op = P.adjoint() @ op_permuted_basis @ P
    return full_op


class CircuitSimulator:
    """
    Simulates quantum circuits acting on quantum states (state vectors or density matrices).

    The simulator can handle unitary gate operations and non-unitary Kraus channel
    operations (noise). If a noise channel is applied to a state vector, the
    representation automatically converts to a density matrix for the rest of
    the simulation.
    """
    def __init__(self, device: str = "cpu"):
        """Initializes the CircuitSimulator.

        Args:
            device (str): The PyTorch device to run the simulation on (e.g., "cpu", "cuda").
                          Defaults to "cpu".

        Raises:
            ValueError: If the provided device string is invalid.
        """
        try:
            self.device = torch.device(device)
        except RuntimeError:
            raise ValueError(f"Invalid device string: {device}. Supported devices (e.g., 'cpu', 'cuda').")

    def run(self, circuit: QuantumCircuit, initial_state_obj: Union[QuantumStateVector, DensityMatrix]) -> Union[QuantumStateVector, DensityMatrix]:
        """
        Simulates the given quantum circuit on the initial quantum state.

        The simulation proceeds operation by operation. If the initial state is a
        `QuantumStateVector` and a Kraus channel (noise) is encountered, the state
        representation is converted to a `DensityMatrix` for all subsequent operations.

        Args:
            circuit: The `QuantumCircuit` to simulate.
            initial_state_obj: The initial quantum state, either a `QuantumStateVector`
                               or a `DensityMatrix`. The state's device will be adjusted
                               to the simulator's device.

        Returns:
            The final quantum state after all circuit operations have been applied.
            The type of the returned state (`QuantumStateVector` or `DensityMatrix`)
            depends on whether the simulation involved only unitary operations (remains
            `QuantumStateVector` if started as one) or if noise channels were applied
            (becomes `DensityMatrix`).

        Raises:
            ValueError: If the number of qubits in the circuit does not match the
                        number of qubits in the initial state.
            ValueError: If an unknown operation type is encountered in the circuit.
            TypeError: If `initial_state_obj` is not a `QuantumStateVector` or `DensityMatrix`.
        """
        num_total_qubits = circuit.num_qubits

        if not isinstance(initial_state_obj, (QuantumStateVector, DensityMatrix)):
             raise TypeError("initial_state_obj must be QuantumStateVector or DensityMatrix.")

        if num_total_qubits != initial_state_obj.num_qubits:
            raise ValueError(
                f"Circuit num_qubits ({num_total_qubits}) does not match "
                f"initial_state num_qubits ({initial_state_obj.num_qubits})."
            )

        # Move initial state to simulator's device and determine representation
        current_initial_state_obj = initial_state_obj.to(self.device)
        is_density_matrix_representation = isinstance(current_initial_state_obj, DensityMatrix)

        if is_density_matrix_representation:
            current_tensor = current_initial_state_obj.density_matrix
        else: # QuantumStateVector
            current_tensor = current_initial_state_obj.state_vector

        for op_idx, (op_data, active_qubits_tuple, op_type) in enumerate(circuit.operations):

            if op_type == "gate":
                gate_matrix = op_data.to(dtype=current_tensor.dtype, device=self.device)
                num_active_qubits = len(active_qubits_tuple)

                if not is_density_matrix_representation: # State Vector Evolution
                    perm_indices_to_front = list(active_qubits_tuple) + [q for q in range(num_total_qubits) if q not in active_qubits_tuple]
                    inv_perm_indices_back = list(torch.argsort(torch.tensor(perm_indices_to_front, device='cpu')).cpu().numpy()) # For list conversion

                    psi = current_tensor.reshape([2] * num_total_qubits)
                    psi = psi.permute(*perm_indices_to_front)
                    psi_active_front = psi.reshape(2**num_active_qubits, -1)

                    psi_transformed = gate_matrix @ psi_active_front

                    psi_transformed = psi_transformed.reshape([2] * num_total_qubits)
                    psi_transformed = psi_transformed.permute(*inv_perm_indices_back)
                    current_tensor = psi_transformed.reshape(-1)
                else: # Density Matrix Evolution for a gate
                    op_full = _get_full_operator(gate_matrix, active_qubits_tuple, num_total_qubits)
                    current_tensor = op_full @ current_tensor @ op_full.adjoint()

            elif op_type == "kraus_channel":
                kraus_ops_list = [k.to(dtype=current_tensor.dtype, device=self.device) for k in op_data]

                if not is_density_matrix_representation:
                    # Convert state vector to density matrix: rho = |psi><psi|
                    current_tensor = current_tensor.unsqueeze(1) @ current_tensor.conj().unsqueeze(0)
                    is_density_matrix_representation = True

                new_rho_tensor = torch.zeros_like(current_tensor)
                for K_op in kraus_ops_list:
                    K_full = _get_full_operator(K_op, active_qubits_tuple, num_total_qubits)
                    new_rho_tensor += K_full @ current_tensor @ K_full.adjoint()
                current_tensor = new_rho_tensor

            else:
                raise ValueError(f"Unknown operation type '{op_type}' at index {op_idx} in circuit.")

        if is_density_matrix_representation:
            return DensityMatrix(num_qubits=num_total_qubits, initial_density_matrix_tensor=current_tensor, device=self.device)
        else:
            return QuantumStateVector(num_qubits=num_total_qubits, initial_state_tensor=current_tensor, device=self.device)
