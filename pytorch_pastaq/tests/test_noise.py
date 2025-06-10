import torch
import pytest
import sys
import os
import numpy as np # For np.sqrt, used in constructing expected values

# Adjust sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from noise import (
    pauli_channel,
    depolarizing_channel,
    amplitude_damping_channel,
    phase_damping_channel,
    COMPLEX_DTYPE, # Import COMPLEX_DTYPE from noise.py
    _I, _X, _Y, _Z # Import helper Pauli matrices from noise.py
)

# --- Helper Function ---

def check_kraus_properties(kraus_ops: list[torch.Tensor],
                           expected_num_ops: int,
                           expected_shape: tuple,
                           filter_zero_ops: bool = False): # Option to filter zero ops before check

    if filter_zero_ops:
        # Filter out Kraus operators that are effectively zero (all elements close to zero)
        # This is for cases where a probability is 0, leading to a zero matrix.
        # The sum K_dag K should still be I even with these zero matrices included,
        # but if the channel implementation *optimizes* them away, the test needs to adapt.
        # The current noise.py implementations return all Kraus ops, even if some are zero matrices.
        # So, filter_zero_ops might not be needed if testing current noise.py.
        # For now, assume all ops are returned.
        pass

    assert len(kraus_ops) == expected_num_ops, f"Expected {expected_num_ops} Kraus operators, got {len(kraus_ops)}"

    sum_KdagK = torch.zeros(expected_shape, dtype=COMPLEX_DTYPE, device=kraus_ops[0].device if kraus_ops else 'cpu')

    for i, K in enumerate(kraus_ops):
        assert K.shape == expected_shape, f"Kraus op {i} shape mismatch: expected {expected_shape}, got {K.shape}"
        assert K.dtype == COMPLEX_DTYPE, f"Kraus op {i} dtype mismatch: expected {COMPLEX_DTYPE}, got {K.dtype}"
        sum_KdagK += K.adjoint() @ K

    identity_matrix = torch.eye(expected_shape[0], dtype=COMPLEX_DTYPE, device=sum_KdagK.device)
    torch.testing.assert_close(sum_KdagK, identity_matrix, rtol=1e-6, atol=1e-7, msg="Sum K_dag @ K is not Identity")

# --- Tests for pauli_channel() ---

def test_pauli_channel_no_error():
    kraus_ops = pauli_channel(px=0, py=0, pz=0)
    check_kraus_properties(kraus_ops, expected_num_ops=4, expected_shape=(2,2))
    # K_I = I, K_X=0, K_Y=0, K_Z=0
    torch.testing.assert_close(kraus_ops[0], _I())
    torch.testing.assert_close(kraus_ops[1], torch.zeros_like(_X()))
    torch.testing.assert_close(kraus_ops[2], torch.zeros_like(_Y()))
    torch.testing.assert_close(kraus_ops[3], torch.zeros_like(_Z()))


def test_pauli_channel_only_X():
    px = 0.1
    kraus_ops = pauli_channel(px=px, py=0, pz=0)
    check_kraus_properties(kraus_ops, expected_num_ops=4, expected_shape=(2,2))
    torch.testing.assert_close(kraus_ops[0], np.sqrt(1-px) * _I())
    torch.testing.assert_close(kraus_ops[1], np.sqrt(px) * _X())
    torch.testing.assert_close(kraus_ops[2], torch.zeros_like(_Y()))
    torch.testing.assert_close(kraus_ops[3], torch.zeros_like(_Z()))

def test_pauli_channel_full():
    px, py, pz = 0.1, 0.05, 0.02
    p_sum = px + py + pz
    kraus_ops = pauli_channel(px=px, py=py, pz=pz)
    check_kraus_properties(kraus_ops, expected_num_ops=4, expected_shape=(2,2))
    torch.testing.assert_close(kraus_ops[0], np.sqrt(1-p_sum) * _I())
    torch.testing.assert_close(kraus_ops[1], np.sqrt(px) * _X())
    torch.testing.assert_close(kraus_ops[2], np.sqrt(py) * _Y())
    torch.testing.assert_close(kraus_ops[3], np.sqrt(pz) * _Z())

def test_pauli_channel_sum_prob_one():
    px, py, pz = 0.5, 0.5, 0.0
    kraus_ops = pauli_channel(px=px, py=py, pz=pz) # p_i = 0
    check_kraus_properties(kraus_ops, expected_num_ops=4, expected_shape=(2,2))
    torch.testing.assert_close(kraus_ops[0], torch.zeros_like(_I())) # sqrt(0) * I
    torch.testing.assert_close(kraus_ops[1], np.sqrt(px) * _X())

def test_pauli_channel_invalid_probs():
    with pytest.raises(ValueError, match="Probabilities px, py, pz must be between 0 and 1."):
        pauli_channel(px=-0.1, py=0, pz=0)
    with pytest.raises(ValueError, match="Sum of probabilities px \+ py \+ pz must be between 0 and 1."):
        pauli_channel(px=0.5, py=0.6, pz=0)

# --- Tests for depolarizing_channel() ---

def test_depolarizing_no_error():
    p = 0.0
    kraus_ops = depolarizing_channel(p=p)
    check_kraus_properties(kraus_ops, expected_num_ops=4, expected_shape=(2,2))
    torch.testing.assert_close(kraus_ops[0], _I()) # sqrt(1-0)I
    for i in range(1,4): # X, Y, Z terms should be zero
        torch.testing.assert_close(kraus_ops[i], torch.zeros((2,2), dtype=COMPLEX_DTYPE))

def test_depolarizing_full_error():
    p = 1.0
    kraus_ops = depolarizing_channel(p=p)
    check_kraus_properties(kraus_ops, expected_num_ops=4, expected_shape=(2,2))
    torch.testing.assert_close(kraus_ops[0], torch.zeros_like(_I())) # sqrt(1-1)I
    val = np.sqrt(p/3.0)
    torch.testing.assert_close(kraus_ops[1], val * _X())
    torch.testing.assert_close(kraus_ops[2], val * _Y())
    torch.testing.assert_close(kraus_ops[3], val * _Z())

def test_depolarizing_general_p():
    p = 0.3
    kraus_ops = depolarizing_channel(p=p)
    check_kraus_properties(kraus_ops, expected_num_ops=4, expected_shape=(2,2))
    torch.testing.assert_close(kraus_ops[0], np.sqrt(1-p) * _I())
    val = np.sqrt(p/3.0)
    torch.testing.assert_close(kraus_ops[1], val * _X())
    torch.testing.assert_close(kraus_ops[2], val * _Y())
    torch.testing.assert_close(kraus_ops[3], val * _Z())

def test_depolarizing_invalid_p():
    with pytest.raises(ValueError, match="Depolarizing probability p must be between 0 and 1."):
        depolarizing_channel(p=-0.1)
    with pytest.raises(ValueError, match="Depolarizing probability p must be between 0 and 1."):
        depolarizing_channel(p=1.1)

# --- Tests for amplitude_damping_channel() ---

def test_amplitude_damping_no_damping():
    gamma = 0.0
    kraus_ops = amplitude_damping_channel(gamma=gamma)
    check_kraus_properties(kraus_ops, expected_num_ops=2, expected_shape=(2,2))
    # K0 = [[1,0],[0,sqrt(1-0)]] = I
    # K1 = [[0,sqrt(0)],[0,0]] = 0
    torch.testing.assert_close(kraus_ops[0], _I())
    torch.testing.assert_close(kraus_ops[1], torch.zeros((2,2), dtype=COMPLEX_DTYPE))

def test_amplitude_damping_full_damping():
    gamma = 1.0
    kraus_ops = amplitude_damping_channel(gamma=gamma)
    check_kraus_properties(kraus_ops, expected_num_ops=2, expected_shape=(2,2))
    # K0 = [[1,0],[0,sqrt(0)]] = [[1,0],[0,0]]
    # K1 = [[0,sqrt(1)],[0,0]] = [[0,1],[0,0]]
    expected_K0 = torch.tensor([[1,0],[0,0]], dtype=COMPLEX_DTYPE)
    expected_K1 = torch.tensor([[0,1],[0,0]], dtype=COMPLEX_DTYPE)
    torch.testing.assert_close(kraus_ops[0], expected_K0)
    torch.testing.assert_close(kraus_ops[1], expected_K1)

def test_amplitude_damping_general_gamma():
    gamma = 0.2
    kraus_ops = amplitude_damping_channel(gamma=gamma)
    check_kraus_properties(kraus_ops, expected_num_ops=2, expected_shape=(2,2))
    expected_K0 = torch.tensor([[1,0],[0, np.sqrt(1-gamma)]], dtype=COMPLEX_DTYPE)
    expected_K1 = torch.tensor([[0, np.sqrt(gamma)],[0,0]], dtype=COMPLEX_DTYPE)
    torch.testing.assert_close(kraus_ops[0], expected_K0)
    torch.testing.assert_close(kraus_ops[1], expected_K1)

def test_amplitude_damping_invalid_gamma():
    with pytest.raises(ValueError, match="Damping probability gamma must be between 0 and 1."):
        amplitude_damping_channel(gamma=-0.1)
    with pytest.raises(ValueError, match="Damping probability gamma must be between 0 and 1."):
        amplitude_damping_channel(gamma=1.1)

# --- Tests for phase_damping_channel() ---

def test_phase_damping_no_damping():
    gamma = 0.0
    kraus_ops = phase_damping_channel(gamma=gamma)
    check_kraus_properties(kraus_ops, expected_num_ops=2, expected_shape=(2,2))
    # K0 = sqrt(1)I = I
    # K1 = sqrt(0)Z = 0
    torch.testing.assert_close(kraus_ops[0], _I())
    torch.testing.assert_close(kraus_ops[1], torch.zeros_like(_Z()))

def test_phase_damping_full_damping():
    gamma = 1.0
    kraus_ops = phase_damping_channel(gamma=gamma)
    check_kraus_properties(kraus_ops, expected_num_ops=2, expected_shape=(2,2))
    # K0 = sqrt(0)I = 0
    # K1 = sqrt(1)Z = Z
    torch.testing.assert_close(kraus_ops[0], torch.zeros_like(_I()))
    torch.testing.assert_close(kraus_ops[1], _Z())

def test_phase_damping_general_gamma():
    gamma = 0.25
    kraus_ops = phase_damping_channel(gamma=gamma)
    check_kraus_properties(kraus_ops, expected_num_ops=2, expected_shape=(2,2))
    expected_K0 = np.sqrt(1-gamma) * _I()
    expected_K1 = np.sqrt(gamma) * _Z()
    torch.testing.assert_close(kraus_ops[0], expected_K0)
    torch.testing.assert_close(kraus_ops[1], expected_K1)

def test_phase_damping_invalid_gamma():
    with pytest.raises(ValueError, match="Dephasing probability gamma must be between 0 and 1."):
        phase_damping_channel(gamma=-0.1)
    with pytest.raises(ValueError, match="Dephasing probability gamma must be between 0 and 1."):
        phase_damping_channel(gamma=1.1)
