import torch
import pytest
import numpy as np
import sys
import os

# Adjust sys.path to allow importing from the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from gates import (
    H, X, Y, Z, S, Sdg, T, Tdg, P,
    Rx, Ry, Rz,
    CNOT, CZ, SWAP, CRz, COMPLEX_DTYPE as GATE_COMPLEX_DTYPE # Import COMPLEX_DTYPE from gates
)

# Use the COMPLEX_DTYPE from the gates module for consistency
COMPLEX_DTYPE = GATE_COMPLEX_DTYPE

# Helper function for checking gates
def check_gate(gate_tensor: torch.Tensor,
               expected_shape: tuple,
               expected_elements: torch.Tensor): # Removed expected_dtype, it's fixed by gates.py
    assert gate_tensor.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {gate_tensor.shape}"
    assert gate_tensor.dtype == COMPLEX_DTYPE, f"Dtype mismatch: expected {COMPLEX_DTYPE}, got {gate_tensor.dtype}"
    # Ensure expected_elements is also on the same dtype for comparison
    torch.testing.assert_close(gate_tensor, expected_elements.to(COMPLEX_DTYPE), rtol=1e-6, atol=1e-7)

# --- Tests for Single-Qubit Gates ---

def test_H_gate():
    expected = (1/np.sqrt(2.0)) * torch.tensor([[1, 1], [1, -1]], dtype=COMPLEX_DTYPE)
    check_gate(H(), (2,2), expected)

def test_X_gate():
    expected = torch.tensor([[0, 1], [1, 0]], dtype=COMPLEX_DTYPE)
    check_gate(X(), (2,2), expected)

def test_Y_gate():
    expected = torch.tensor([[0, -1j], [1j, 0]], dtype=COMPLEX_DTYPE)
    check_gate(Y(), (2,2), expected)

def test_Z_gate():
    expected = torch.tensor([[1, 0], [0, -1]], dtype=COMPLEX_DTYPE)
    check_gate(Z(), (2,2), expected)

def test_S_gate():
    expected = torch.tensor([[1, 0], [0, 1j]], dtype=COMPLEX_DTYPE)
    check_gate(S(), (2,2), expected)

def test_Sdg_gate():
    expected = torch.tensor([[1, 0], [0, -1j]], dtype=COMPLEX_DTYPE)
    check_gate(Sdg(), (2,2), expected)
    torch.testing.assert_close(S() @ Sdg(), torch.eye(2, dtype=COMPLEX_DTYPE), rtol=1e-6, atol=1e-7)

def test_T_gate():
    val = np.exp(1j * np.pi / 4.0)
    expected = torch.tensor([[1, 0], [0, val]], dtype=COMPLEX_DTYPE)
    check_gate(T(), (2,2), expected)

def test_Tdg_gate():
    val = np.exp(-1j * np.pi / 4.0)
    expected = torch.tensor([[1, 0], [0, val]], dtype=COMPLEX_DTYPE)
    check_gate(Tdg(), (2,2), expected)
    torch.testing.assert_close(T() @ Tdg(), torch.eye(2, dtype=COMPLEX_DTYPE), rtol=1e-6, atol=1e-7)

def test_P_gate():
    check_gate(P(0), (2,2), torch.eye(2, dtype=COMPLEX_DTYPE))
    check_gate(P(np.pi/2.0), (2,2), S()) # P(pi/2) is S
    check_gate(P(np.pi), (2,2), Z())     # P(pi) is Z

    theta = np.pi / 3.0
    expected_P_general = torch.tensor([[1, 0], [0, np.exp(1j * theta)]], dtype=COMPLEX_DTYPE)
    check_gate(P(theta), (2,2), expected_P_general)

# --- Tests for Rotation Gates ---

def test_Rx_gate():
    check_gate(Rx(0), (2,2), torch.eye(2, dtype=COMPLEX_DTYPE))

    expected_Rx_pi = torch.tensor([[0, -1j], [-1j, 0]], dtype=COMPLEX_DTYPE) # -iX
    check_gate(Rx(np.pi), (2,2), expected_Rx_pi)

    theta = np.pi / 3.0
    c = np.cos(theta/2.0)
    s = np.sin(theta/2.0)
    expected_Rx_general = torch.tensor([[c, -1j*s], [-1j*s, c]], dtype=COMPLEX_DTYPE)
    check_gate(Rx(theta), (2,2), expected_Rx_general)

def test_Ry_gate():
    check_gate(Ry(0), (2,2), torch.eye(2, dtype=COMPLEX_DTYPE))

    expected_Ry_pi = torch.tensor([[0, -1], [1, 0]], dtype=COMPLEX_DTYPE) # -iY
    check_gate(Ry(np.pi), (2,2), expected_Ry_pi)

    theta = np.pi / 3.0
    c = np.cos(theta/2.0)
    s = np.sin(theta/2.0)
    expected_Ry_general = torch.tensor([[c, -s], [s, c]], dtype=COMPLEX_DTYPE)
    check_gate(Ry(theta), (2,2), expected_Ry_general)

def test_Rz_gate():
    check_gate(Rz(0), (2,2), torch.eye(2, dtype=COMPLEX_DTYPE))

    phi_half = np.pi / 2.0 # Rz(pi/2)
    val_minus = np.exp(-1j * phi_half / 2.0) # exp(-i*pi/4)
    val_plus = np.exp(1j * phi_half / 2.0)   # exp(i*pi/4)
    expected_Rz_pi_2 = torch.tensor([[val_minus, 0], [0, val_plus]], dtype=COMPLEX_DTYPE)
    check_gate(Rz(phi_half), (2,2), expected_Rz_pi_2)

    expected_Rz_pi = torch.tensor([[-1j, 0], [0, 1j]], dtype=COMPLEX_DTYPE) # Rz(pi) = diag(-i,i)
    check_gate(Rz(np.pi), (2,2), expected_Rz_pi)

    phi_gen = np.pi / 3.0
    vm = np.exp(-1j * phi_gen/2.0)
    vp = np.exp(1j * phi_gen/2.0)
    expected_Rz_general = torch.tensor([[vm, 0], [0, vp]], dtype=COMPLEX_DTYPE)
    check_gate(Rz(phi_gen), (2,2), expected_Rz_general)

# --- Tests for Two-Qubit Gates ---

def test_CNOT_gate():
    expected = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=COMPLEX_DTYPE)
    check_gate(CNOT(), (4,4), expected)

def test_CZ_gate():
    expected = torch.tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=COMPLEX_DTYPE)
    check_gate(CZ(), (4,4), expected)

def test_SWAP_gate():
    expected = torch.tensor([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=COMPLEX_DTYPE)
    check_gate(SWAP(), (4,4), expected)

def test_CRz_gate():
    check_gate(CRz(0), (4,4), torch.eye(4, dtype=COMPLEX_DTYPE))

    expected_CRz_pi = torch.diag(torch.tensor([1, 1, -1j, 1j], dtype=COMPLEX_DTYPE))
    check_gate(CRz(np.pi), (4,4), expected_CRz_pi)

    phi = np.pi / 3.0
    vm = np.exp(-1j * phi/2.0)
    vp = np.exp(1j * phi/2.0)
    expected_CRz_general = torch.diag(torch.tensor([1, 1, vm, vp], dtype=COMPLEX_DTYPE))
    check_gate(CRz(phi), (4,4), expected_CRz_general)
