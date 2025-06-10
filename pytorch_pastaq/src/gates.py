import torch
import numpy as np # For np.pi, as torch.pi might not be available in all torch versions

# Common dtype for all gate tensors
COMPLEX_DTYPE = torch.complex64
# Real dtype corresponding to the complex one (e.g., float32 for complex64)
# This is determined by creating a zero tensor with COMPLEX_DTYPE and then accessing its real part's dtype.
REAL_DTYPE = torch.tensor(0., dtype=COMPLEX_DTYPE).real.dtype


# Single-Qubit Gates

def H() -> torch.Tensor:
    """Hadamard gate.

    Returns:
        torch.Tensor: The 2x2 Hadamard gate matrix.
    """
    factor = 1 / torch.sqrt(torch.tensor(2.0, dtype=REAL_DTYPE))
    return factor * torch.tensor(
        [[1, 1], [1, -1]], dtype=COMPLEX_DTYPE
    )

def X() -> torch.Tensor:
    """Pauli-X gate (NOT gate).

    Returns:
        torch.Tensor: The 2x2 Pauli-X gate matrix.
    """
    return torch.tensor([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)

def Y() -> torch.Tensor:
    """Pauli-Y gate.

    Returns:
        torch.Tensor: The 2x2 Pauli-Y gate matrix.
    """
    return torch.tensor([[0, -1j], [1j, 0]], dtype=COMPLEX_DTYPE)

def Z() -> torch.Tensor:
    """Pauli-Z gate.

    Returns:
        torch.Tensor: The 2x2 Pauli-Z gate matrix.
    """
    return torch.tensor([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE)

def S() -> torch.Tensor:
    """S gate (Phase gate, sqrt(Z) up to global phase).

    Returns:
        torch.Tensor: The 2x2 S gate matrix.
    """
    return torch.tensor([[1, 0], [0, 1j]], dtype=COMPLEX_DTYPE)

def Sdg() -> torch.Tensor:
    """Dagger of S gate (S†).

    Returns:
        torch.Tensor: The 2x2 S† gate matrix.
    """
    return torch.tensor([[1, 0], [0, -1j]], dtype=COMPLEX_DTYPE)

def T() -> torch.Tensor:
    """T gate (π/8 gate).

    Returns:
        torch.Tensor: The 2x2 T gate matrix.
    """
    pi_val = getattr(torch, 'pi', np.pi)
    angle = torch.tensor(pi_val / 4, dtype=REAL_DTYPE)
    return torch.tensor([[1, 0], [0, torch.exp(1j * angle)]], dtype=COMPLEX_DTYPE)

def Tdg() -> torch.Tensor:
    """Dagger of T gate (T†).

    Returns:
        torch.Tensor: The 2x2 T† gate matrix.
    """
    pi_val = getattr(torch, 'pi', np.pi)
    angle = torch.tensor(pi_val / 4, dtype=REAL_DTYPE)
    return torch.tensor([[1, 0], [0, torch.exp(-1j * angle)]], dtype=COMPLEX_DTYPE)

def P(theta: float) -> torch.Tensor:
    """Phase shift gate. Matrix form: [[1, 0], [0, exp(i*theta)]].

    Args:
        theta (float): The phase shift angle in radians.

    Returns:
        torch.Tensor: The 2x2 Phase shift gate matrix.
    """
    theta_t = torch.tensor(theta, dtype=REAL_DTYPE)
    return torch.tensor([[1, 0], [0, torch.exp(1j * theta_t)]], dtype=COMPLEX_DTYPE)

# Rotation Gates

def Rx(theta: float) -> torch.Tensor:
    """Rotation around X-axis. Rx(theta) = exp(-i*theta*X/2).
    Matrix form: [[cos(theta/2), -i*sin(theta/2)],
                  [-i*sin(theta/2), cos(theta/2)]].

    Args:
        theta (float): The rotation angle in radians.

    Returns:
        torch.Tensor: The 2x2 Rx gate matrix.
    """
    theta_half_t = torch.tensor(theta / 2, dtype=REAL_DTYPE)
    cos_t_2 = torch.cos(theta_half_t)
    sin_t_2 = torch.sin(theta_half_t)
    return torch.tensor(
        [[cos_t_2, -1j * sin_t_2], [-1j * sin_t_2, cos_t_2]], dtype=COMPLEX_DTYPE
    )

def Ry(theta: float) -> torch.Tensor:
    """Rotation around Y-axis. Ry(theta) = exp(-i*theta*Y/2).
    Matrix form: [[cos(theta/2), -sin(theta/2)],
                  [sin(theta/2), cos(theta/2)]].

    Args:
        theta (float): The rotation angle in radians.

    Returns:
        torch.Tensor: The 2x2 Ry gate matrix.
    """
    theta_half_t = torch.tensor(theta / 2, dtype=REAL_DTYPE)
    cos_t_2 = torch.cos(theta_half_t)
    sin_t_2 = torch.sin(theta_half_t)
    return torch.tensor(
        [[cos_t_2, -sin_t_2], [sin_t_2, cos_t_2]], dtype=COMPLEX_DTYPE
    )

def Rz(phi: float) -> torch.Tensor:
    """Rotation around Z-axis. Rz(phi) = exp(-i*phi*Z/2).
    Matrix form: [[exp(-i*phi/2), 0], [0, exp(i*phi/2)]].

    Args:
        phi (float): The rotation angle in radians.

    Returns:
        torch.Tensor: The 2x2 Rz gate matrix.
    """
    phi_half_t = torch.tensor(phi / 2, dtype=REAL_DTYPE)
    exp_minus_iphi_2 = torch.exp(-1j * phi_half_t)
    exp_iphi_2 = torch.exp(1j * phi_half_t)
    return torch.tensor(
        [[exp_minus_iphi_2, 0], [0, exp_iphi_2]], dtype=COMPLEX_DTYPE
    )

# Two-Qubit Gates

def CNOT() -> torch.Tensor:
    """Controlled-NOT gate (CX).
    Control qubit is the first qubit (index 0), target is the second (index 1).
    Matrix form: [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]].

    Returns:
        torch.Tensor: The 4x4 CNOT gate matrix.
    """
    return torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=COMPLEX_DTYPE,
    )

def CZ() -> torch.Tensor:
    """Controlled-Z gate.
    Applies Z to target if control is |1>.
    Matrix form: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]].

    Returns:
        torch.Tensor: The 4x4 CZ gate matrix.
    """
    return torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
        dtype=COMPLEX_DTYPE,
    )

def SWAP() -> torch.Tensor:
    """SWAP gate. Swaps the states of two qubits.
    Matrix form: [[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]].

    Returns:
        torch.Tensor: The 4x4 SWAP gate matrix.
    """
    return torch.tensor(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=COMPLEX_DTYPE,
    )

# Controlled Rotation Gate

def CRz(phi: float) -> torch.Tensor:
    """Controlled-Rz gate.
    Applies Rz(phi) to the target qubit (second qubit, index 1) if the control
    qubit (first qubit, index 0) is |1>.
    Rz(phi) is defined as [[exp(-i*phi/2), 0], [0, exp(i*phi/2)]].

    Args:
        phi (float): The rotation angle in radians for the Rz gate.

    Returns:
        torch.Tensor: The 4x4 Controlled-Rz gate matrix.
    """
    phi_half_t = torch.tensor(phi / 2, dtype=REAL_DTYPE)
    exp_minus_iphi_2 = torch.exp(-1j * phi_half_t)
    exp_iphi_2 = torch.exp(1j * phi_half_t)

    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, exp_minus_iphi_2, 0],
            [0, 0, 0, exp_iphi_2],
        ],
        dtype=COMPLEX_DTYPE,
    )
