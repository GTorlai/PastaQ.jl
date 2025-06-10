import torch
from typing import List
import numpy as np # For np.sqrt, though torch.sqrt(torch.tensor(x)) is used

COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.tensor(0., dtype=COMPLEX_DTYPE).real.dtype # Derived real dtype

# Helper Pauli matrices (internal use, simple docstrings or comments suffice)
def _I() -> torch.Tensor:
    """Single-qubit Identity matrix."""
    return torch.tensor([[1, 0], [0, 1]], dtype=COMPLEX_DTYPE)

def _X() -> torch.Tensor:
    """Single-qubit Pauli-X matrix."""
    return torch.tensor([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)

def _Y() -> torch.Tensor:
    """Single-qubit Pauli-Y matrix."""
    return torch.tensor([[0, -1j], [1j, 0]], dtype=COMPLEX_DTYPE)

def _Z() -> torch.Tensor:
    """Single-qubit Pauli-Z matrix."""
    return torch.tensor([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE)


def pauli_channel(px: float, py: float, pz: float) -> List[torch.Tensor]:
    """
    Generates Kraus operators for a Pauli channel.

    The Pauli channel applies a Pauli-X, Pauli-Y, or Pauli-Z error with
    probabilities px, py, pz respectively. With probability 1-px-py-pz,
    no error (Identity) occurs.

    Args:
        px (float): Probability of a Pauli-X error.
        py (float): Probability of a Pauli-Y error.
        pz (float): Probability of a Pauli-Z error.
            Constraint: 0 <= px, py, pz and px + py + pz <= 1.

    Returns:
        List[torch.Tensor]: A list of four Kraus operators [K_I, K_X, K_Y, K_Z],
                            where:
                            K_I = sqrt(1-px-py-pz) * I
                            K_X = sqrt(px) * X
                            K_Y = sqrt(py) * Y
                            K_Z = sqrt(pz) * Z
                            All operators are 2x2 torch.Tensors with COMPLEX_DTYPE.

    Raises:
        ValueError: If probabilities are not in the range [0,1] or if their sum exceeds 1.
    """
    if not (0 <= px <= 1 and 0 <= py <= 1 and 0 <= pz <= 1):
        raise ValueError("Probabilities px, py, pz must be between 0 and 1.")

    sum_p = px + py + pz
    if not (0 <= sum_p <= 1.000001): # Allow for small floating point inaccuracies if sum_p is very close to 1
        raise ValueError(f"Sum of probabilities px + py + pz ({sum_p}) must be between 0 and 1.")

    p_i = 1.0 - sum_p
    # Handle potential negative p_i due to floating point arithmetic if sum_p slightly > 1
    p_i = max(0.0, p_i)

    K_I = torch.sqrt(torch.tensor(p_i, dtype=REAL_DTYPE)) * _I()
    K_X = torch.sqrt(torch.tensor(px, dtype=REAL_DTYPE)) * _X()
    K_Y = torch.sqrt(torch.tensor(py, dtype=REAL_DTYPE)) * _Y()
    K_Z = torch.sqrt(torch.tensor(pz, dtype=REAL_DTYPE)) * _Z() # Corrected: was missing * _Z()

    return [K_I, K_X, K_Y, K_Z]


def depolarizing_channel(p: float) -> List[torch.Tensor]:
    """
    Generates Kraus operators for a single-qubit depolarizing channel.

    This channel describes a process where, with probability `p`, the qubit's
    state is randomized to the maximally mixed state (I/2). Equivalently,
    with probability `p`, one of X, Y, or Z errors occurs with equal likelihood (p/3).
    The Kraus operators are:
    K_0 = sqrt(1-p) * I
    K_1 = sqrt(p/3) * X
    K_2 = sqrt(p/3) * Y
    K_3 = sqrt(p/3) * Z

    Args:
        p (float): The probability of depolarization (0 <= p <= 1).

    Returns:
        List[torch.Tensor]: A list of four Kraus operators [K_0, K_1, K_2, K_3],
                            all 2x2 torch.Tensors with COMPLEX_DTYPE.

    Raises:
        ValueError: If the depolarizing probability `p` is not in the range [0,1].
    """
    if not (0 <= p <= 1.000001): # Allow for small floating point inaccuracies
        raise ValueError(f"Depolarizing probability p ({p}) must be between 0 and 1.")
    p = min(max(0.0, p), 1.0) # Clamp p to [0,1] after check to handle precision issues for sqrt

    K_0 = torch.sqrt(torch.tensor(1.0 - p, dtype=REAL_DTYPE)) * _I()
    K_1 = torch.sqrt(torch.tensor(p / 3.0, dtype=REAL_DTYPE)) * _X()
    K_2 = torch.sqrt(torch.tensor(p / 3.0, dtype=REAL_DTYPE)) * _Y()
    K_3 = torch.sqrt(torch.tensor(p / 3.0, dtype=REAL_DTYPE)) * _Z()

    return [K_0, K_1, K_2, K_3]


def amplitude_damping_channel(gamma: float) -> List[torch.Tensor]:
    """
    Generates Kraus operators for an amplitude damping channel (e.g., T1 decay).

    This channel models the loss of energy from a qubit state, e.g., the decay
    of |1> to |0> with probability `gamma`.
    The Kraus operators are:
    K_0 = [[1, 0], [0, sqrt(1-gamma)]]
    K_1 = [[0, sqrt(gamma)], [0, 0]]

    Args:
        gamma (float): The probability of a qubit losing excitation (0 <= gamma <= 1).

    Returns:
        List[torch.Tensor]: A list of two Kraus operators [K_0, K_1],
                            all 2x2 torch.Tensors with COMPLEX_DTYPE.

    Raises:
        ValueError: If the damping probability `gamma` is not in the range [0,1].
    """
    if not (0 <= gamma <= 1.000001): # Allow for small floating point inaccuracies
        raise ValueError(f"Damping probability gamma ({gamma}) must be between 0 and 1.")
    gamma = min(max(0.0, gamma), 1.0) # Clamp gamma to [0,1] for sqrt

    K_0 = torch.tensor([[1, 0], [0, torch.sqrt(torch.tensor(1.0 - gamma, dtype=REAL_DTYPE))]], dtype=COMPLEX_DTYPE)
    K_1 = torch.tensor([[0, torch.sqrt(torch.tensor(gamma, dtype=REAL_DTYPE))], [0, 0]], dtype=COMPLEX_DTYPE)

    return [K_0, K_1]


def phase_damping_channel(gamma: float) -> List[torch.Tensor]:
    """
    Generates Kraus operators for a phase damping channel (pure dephasing).

    This channel models the loss of phase coherence without loss of energy.
    It's equivalent to applying a Pauli-Z error with probability `gamma`.
    The Kraus operators are:
    K_0 = sqrt(1-gamma) * I
    K_1 = sqrt(gamma) * Z

    Args:
        gamma (float): The probability of a phase error (Z error) occurring (0 <= gamma <= 1).

    Returns:
        List[torch.Tensor]: A list of two Kraus operators [K_0, K_1],
                            all 2x2 torch.Tensors with COMPLEX_DTYPE.

    Raises:
        ValueError: If the dephasing probability `gamma` is not in the range [0,1].
    """
    if not (0 <= gamma <= 1.000001): # Allow for small floating point inaccuracies
        raise ValueError(f"Dephasing probability gamma ({gamma}) must be between 0 and 1.")
    gamma = min(max(0.0, gamma), 1.0) # Clamp gamma to [0,1] for sqrt

    K_0 = torch.sqrt(torch.tensor(1.0 - gamma, dtype=REAL_DTYPE)) * _I()
    K_1 = torch.sqrt(torch.tensor(gamma, dtype=REAL_DTYPE)) * _Z()

    return [K_0, K_1]
