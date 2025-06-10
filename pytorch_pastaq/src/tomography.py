import torch
from typing import List, Dict, Tuple, Optional, Union # Added Union for device type hint
import numpy as np
import itertools

try:
    from .states import DensityMatrix
except ImportError:
    from states import DensityMatrix


COMPLEX_DTYPE = torch.complex64
COMPLEX_DTYPE_REAL_PART = torch.float32 if COMPLEX_DTYPE.is_complex else COMPLEX_DTYPE

# --- Single-qubit Pauli and Identity matrices (internal use) ---
_I_SINGLE = torch.eye(2, dtype=COMPLEX_DTYPE)
_X_SINGLE = torch.tensor([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)
_Y_SINGLE = torch.tensor([[0, -1j], [1j, 0]], dtype=COMPLEX_DTYPE)
_Z_SINGLE = torch.tensor([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE)

# Pauli matrices for qst_linear_inversion_single_qubit (exposed for potential direct use if needed)
I_MATRIX = _I_SINGLE.clone()
X_MATRIX = _X_SINGLE.clone()
Y_MATRIX = _Y_SINGLE.clone()
Z_MATRIX = _Z_SINGLE.clone()

# Pre-calculated constant for basis state construction
SQRT_HALF = torch.tensor(1.0 / np.sqrt(2.0), dtype=torch.float32) # real_dtype of COMPLEX_DTYPE

_PAULI_MAP_SINGLE_QUBIT = {
    'I': _I_SINGLE,
    'X': _X_SINGLE,
    'Y': _Y_SINGLE,
    'Z': _Z_SINGLE,
}

def get_basis_projectors_single_qubit(basis: str, device: Union[str, torch.device] = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns projectors for single-qubit measurements in the specified basis.

    Args:
        basis (str): The measurement basis, either "X", "Y", or "Z".
        device (Union[str, torch.device]): The PyTorch device for the projectors.
            Defaults to 'cpu'.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple `(P_0, P_1)` containing the two
            projector matrices.
            - For "Z" basis: `P_0 = |0><0|, P_1 = |1><1|`.
            - For "X" basis: `P_0 = |+><+|, P_1 = |-><-|`.
            - For "Y" basis: `P_0 = |+i><+i|, P_1 = |-i><-i|`.

    Raises:
        ValueError: If an unknown `basis` string is provided.
    """
    dev = torch.device(device)
    # Ensure SQRT_HALF is on the correct device and dtype before multiplication
    sqrt_half_dev = SQRT_HALF.to(device=dev, dtype=COMPLEX_DTYPE_REAL_PART if COMPLEX_DTYPE.is_complex else COMPLEX_DTYPE)


    if basis == "Z":
        P_0 = torch.tensor([[1, 0], [0, 0]], dtype=COMPLEX_DTYPE, device=dev)
        P_1 = torch.tensor([[0, 0], [0, 1]], dtype=COMPLEX_DTYPE, device=dev)
    elif basis == "X":
        plus_state = sqrt_half_dev * torch.tensor([1, 1], dtype=COMPLEX_DTYPE).to(dev)
        minus_state = sqrt_half_dev * torch.tensor([1, -1], dtype=COMPLEX_DTYPE).to(dev)
        P_0 = torch.outer(plus_state, plus_state.conj())
        P_1 = torch.outer(minus_state, minus_state.conj())
    elif basis == "Y":
        plus_i_state = sqrt_half_dev * torch.tensor([1, 1j], dtype=COMPLEX_DTYPE).to(dev)
        minus_i_state = sqrt_half_dev * torch.tensor([1, -1j], dtype=COMPLEX_DTYPE).to(dev)
        P_0 = torch.outer(plus_i_state, plus_i_state.conj())
        P_1 = torch.outer(minus_i_state, minus_i_state.conj())
    else:
        raise ValueError(f"Unknown basis '{basis}'. Must be 'X', 'Y', or 'Z'.")

    return P_0, P_1


def simulate_measurement_probabilities(state_tensor: torch.Tensor,
                                       projectors: Tuple[torch.Tensor, torch.Tensor]
                                       ) -> List[float]:
    """
    Simulates measurement probabilities for a single-qubit state given projectors.

    Args:
        state_tensor (torch.Tensor): A 2x2 density matrix tensor for the single qubit.
                                     Must be on the same device as the projectors.
        projectors (Tuple[torch.Tensor, torch.Tensor]): A tuple `(P_0, P_1)` of two
            2x2 projector tensors for the measurement basis.

    Returns:
        List[float]: A list of two floats `[prob_0, prob_1]`, where `prob_k` is
                     `Tr(projectors[k] @ state_tensor)`.

    Raises:
        ValueError: If `state_tensor` is not 2x2 or `projectors` are not correctly formatted.
    """
    if state_tensor.shape != (2,2):
        raise ValueError(f"state_tensor must be a 2x2 matrix, got {state_tensor.shape}")
    if not (isinstance(projectors, tuple) and len(projectors) == 2 and
            projectors[0].shape == (2,2) and projectors[1].shape == (2,2) and
            projectors[0].device == state_tensor.device and projectors[1].device == state_tensor.device):
        raise ValueError("projectors must be a tuple of two 2x2 tensors on the same device as state_tensor.")

    prob_0 = torch.trace(projectors[0] @ state_tensor).real.item()
    prob_1 = torch.trace(projectors[1] @ state_tensor).real.item()

    # It's good practice for probabilities to sum to 1. Caller should handle normalization
    # or ensure input state_tensor is trace 1 and projectors sum to I.
    return [prob_0, prob_1]


def qst_linear_inversion_single_qubit(measurement_data: Dict[str, List[float]],
                                      device: Union[str, torch.device] = 'cpu'
                                      ) -> DensityMatrix:
    """
    Performs single-qubit Quantum State Tomography (QST) using Linear Inversion.

    This method reconstructs the density matrix `rho` from expectation values of
    Pauli operators (X, Y, Z), which are derived from measurement probabilities
    in the respective bases. The formula used is:
    `rho = 0.5 * (I + <X>X + <Y>Y + <Z>Z)`,
    where `<P> = prob(outcome_+) - prob(outcome_-)` for Pauli P.

    Args:
        measurement_data (Dict[str, List[float]]): A dictionary mapping basis strings
            ("X", "Y", "Z") to a list of two probabilities `[prob_+, prob_-]`.
            - For "X": `[P(+x), P(-x)]` (eigenstates |+>, |->)
            - For "Y": `[P(+y), P(-y)]` (eigenstates |+i>, |-i>)
            - For "Z": `[P(0), P(1)]` (eigenstates |0>, |1>)
        device (Union[str, torch.device]): PyTorch device for the reconstructed density matrix.
            Defaults to 'cpu'.

    Returns:
        DensityMatrix: A `DensityMatrix` object for the reconstructed single-qubit state.

    Raises:
        ValueError: If `measurement_data` is missing required keys, if probabilities
                    are not valid (e.g., not summing to ~1, out of [0,1] range),
                    or if `device` string is invalid.
    """
    dev = torch.device(device)
    expected_keys = {"X", "Y", "Z"}
    if not expected_keys.issubset(measurement_data.keys()):
        raise ValueError(f"measurement_data must contain keys 'X', 'Y', and 'Z'. Found: {list(measurement_data.keys())}")

    for basis, probs in measurement_data.items():
        if not (isinstance(probs, list) and len(probs) == 2):
            raise ValueError(f"Probabilities for basis {basis} must be a list of 2 floats. Got: {probs}")
        if not all(isinstance(p, (float, int)) and -1e-6 <= p <= 1.0 + 1e-6 for p in probs): # Allow small tolerance
            raise ValueError(f"Probabilities for basis {basis} must be floats approximately between 0 and 1. Got: {probs}")
        if not np.isclose(sum(probs), 1.0, atol=1e-6):
            raise ValueError(f"Probabilities for basis {basis} do not sum to 1 (approx). Sum: {sum(probs)}, Probs: {probs}")

    p_plus_x, p_minus_x = measurement_data["X"]
    p_plus_y, p_minus_y = measurement_data["Y"]
    p_zero_z, p_one_z = measurement_data["Z"]

    r_x = p_plus_x - p_minus_x
    r_y = p_plus_y - p_minus_y
    r_z = p_zero_z - p_one_z

    rho_tensor = 0.5 * (
        I_MATRIX.to(dev) +
        r_x * X_MATRIX.to(dev) +
        r_y * Y_MATRIX.to(dev) +
        r_z * Z_MATRIX.to(dev)
    )
    return DensityMatrix(num_qubits=1, initial_density_matrix_tensor=rho_tensor, device=dev)


def get_pauli_operator(pauli_string: str, device: Union[str, torch.device] ='cpu', dtype: torch.dtype = COMPLEX_DTYPE) -> torch.Tensor:
    """
    Constructs a multi-qubit Pauli operator as a tensor product of single-qubit Pauli matrices.

    Args:
        pauli_string (str): A string representing the Pauli operator, e.g., "IX", "XYZ".
                            Characters must be 'I', 'X', 'Y', or 'Z'.
        device (Union[str, torch.device]): PyTorch device for the resulting tensor.
                                           Defaults to 'cpu'.
        dtype (torch.dtype): PyTorch dtype for the resulting tensor.
                             Defaults to `COMPLEX_DTYPE`.

    Returns:
        torch.Tensor: The N-qubit Pauli operator tensor of shape (2^N, 2^N).

    Raises:
        ValueError: If `pauli_string` is empty or contains invalid characters.
    """
    if not pauli_string:
        raise ValueError("Pauli string cannot be empty.")

    dev = torch.device(device)

    full_operator_tensor: Optional[torch.Tensor] = None

    for pauli_char in pauli_string:
        single_qubit_matrix = _PAULI_MAP_SINGLE_QUBIT.get(pauli_char)
        if single_qubit_matrix is None:
            raise ValueError(f"Invalid character '{pauli_char}' in Pauli string. Must be 'I', 'X', 'Y', or 'Z'.")

        current_single_qubit_matrix = single_qubit_matrix.to(device=dev, dtype=dtype)

        if full_operator_tensor is None:
            full_operator_tensor = current_single_qubit_matrix
        else:
            full_operator_tensor = torch.kron(full_operator_tensor, current_single_qubit_matrix)

    if full_operator_tensor is None: # Should not be reached if pauli_string is not empty
        raise ValueError("Failed to construct Pauli operator, pauli_string might have been effectively empty or logic error.")

    return full_operator_tensor


def qst_linear_inversion_multi_qubit(measurement_data: Dict[str, float],
                                      num_qubits: int,
                                      device: Union[str, torch.device] = 'cpu'
                                      ) -> DensityMatrix:
    """
    Performs multi-qubit Quantum State Tomography (QST) using Linear Inversion.

    Reconstructs the density matrix `rho` using the formula:
    `rho = (1 / 2^N) * sum_{P_s in PauliBasis} Tr(P_s * rho) * P_s`
    where `Tr(P_s * rho)` is the expectation value of the Pauli operator `P_s`,
    and N is `num_qubits`.

    Args:
        measurement_data (Dict[str, float]): Dictionary mapping Pauli strings
            (e.g., "IX", "ZY", "XIX") to their measured expectation values `Tr(P_s * rho)`.
            It should contain expectation values for non-identity Pauli strings.
            If a non-identity Pauli string is missing, its expectation is assumed to be 0.
            The expectation value for the identity string ("I...I") is assumed to be 1.
        num_qubits (int): The number of qubits. Must be a positive integer.
        device (Union[str, torch.device]): PyTorch device for the reconstructed density matrix.
            Defaults to 'cpu'.

    Returns:
        DensityMatrix: A `DensityMatrix` object for the reconstructed multi-qubit state.

    Raises:
        ValueError: If `num_qubits` is not positive.
        TypeError: If an expectation value in `measurement_data` is not a float or int.
    """
    if not isinstance(num_qubits, int) or num_qubits <= 0:
        raise ValueError("num_qubits must be a positive integer.")

    dev = torch.device(device)
    final_dtype = COMPLEX_DTYPE

    dim = 2**num_qubits
    rho_tensor = torch.zeros((dim, dim), dtype=final_dtype, device=dev)

    identity_pauli_string = "I" * num_qubits
    identity_operator = get_pauli_operator(identity_pauli_string, device=dev, dtype=final_dtype)
    rho_tensor += 1.0 * identity_operator # Tr(I...I * rho) = 1.0

    pauli_chars = ['I', 'X', 'Y', 'Z']

    for pauli_tuple in itertools.product(pauli_chars, repeat=num_qubits):
        current_pauli_string = "".join(pauli_tuple)

        if current_pauli_string == identity_pauli_string:
            continue

        expectation_value = measurement_data.get(current_pauli_string, 0.0) # Default to 0.0 if missing

        if not isinstance(expectation_value, (float, int)):
             raise TypeError(f"Expectation value for {current_pauli_string} must be a float/int. Got {type(expectation_value).__name__}")

        if abs(expectation_value) > 1e-9: # Only add if non-zero (tolerance for float)
            pauli_operator = get_pauli_operator(current_pauli_string, device=dev, dtype=final_dtype)
            rho_tensor += float(expectation_value) * pauli_operator # Cast to float just in case int was passed

    rho_tensor /= (2**num_qubits)

    return DensityMatrix(num_qubits=num_qubits, initial_density_matrix_tensor=rho_tensor, device=dev)
