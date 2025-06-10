import torch
import pytest
import sys
import os
import math

# Adjust sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from circuits import QuantumCircuit
from states import QuantumStateVector, DensityMatrix
from gates import X, H, CNOT, Z, Y
from noise import depolarizing_channel, amplitude_damping_channel, phase_damping_channel
from simulation import CircuitSimulator, _get_full_operator, _create_permutation_operator

COMPLEX_DTYPE = torch.complex64

CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_CUDA_TESTS = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")

def get_zero_state_vector(num_qubits, device='cpu') -> QuantumStateVector:
    return QuantumStateVector(num_qubits=num_qubits, device=device)

def get_zero_density_matrix(num_qubits, device='cpu') -> DensityMatrix:
    return DensityMatrix(num_qubits=num_qubits, device=device)

I_2x2 = torch.eye(2, dtype=COMPLEX_DTYPE)

def test_simulator_init():
    sim_cpu = CircuitSimulator(device='cpu')
    assert sim_cpu.device == torch.device('cpu')
    if CUDA_AVAILABLE:
        sim_cuda = CircuitSimulator(device='cuda')
        assert sim_cuda.device.type == 'cuda'
    with pytest.raises(ValueError, match="Invalid device string"):
        CircuitSimulator(device='invalid_device_str')

def test_permutation_operator_identity():
    P_identity_2q = _create_permutation_operator(num_qubits=2, qubit_map=[0,1], dtype=COMPLEX_DTYPE, device='cpu')
    torch.testing.assert_close(P_identity_2q, torch.eye(4, dtype=COMPLEX_DTYPE))
    P_identity_3q = _create_permutation_operator(num_qubits=3, qubit_map=[0,1,2], dtype=COMPLEX_DTYPE, device='cpu')
    torch.testing.assert_close(P_identity_3q, torch.eye(8, dtype=COMPLEX_DTYPE))

def test_permutation_operator_swap_2q():
    P_swap_2q = _create_permutation_operator(num_qubits=2, qubit_map=[1,0], dtype=COMPLEX_DTYPE, device='cpu')
    expected_swap = torch.tensor([
        [1,0,0,0], [0,0,1,0], [0,1,0,0], [0,0,0,1]
    ], dtype=COMPLEX_DTYPE)
    torch.testing.assert_close(P_swap_2q, expected_swap)
    psi_01 = torch.tensor([0,1,0,0], dtype=COMPLEX_DTYPE).reshape(-1,1)
    psi_10_expected = torch.tensor([0,0,1,0], dtype=COMPLEX_DTYPE).reshape(-1,1)
    torch.testing.assert_close(P_swap_2q @ psi_01, psi_10_expected)

# Corrected expected matrix for CNOT(0,2)
_CNOT_02_expected_matrix_corrected = torch.zeros((8,8), dtype=COMPLEX_DTYPE)
_CNOT_02_expected_matrix_corrected[0,0] = 1; _CNOT_02_expected_matrix_corrected[2,2] = 1
_CNOT_02_expected_matrix_corrected[4,4] = 1; _CNOT_02_expected_matrix_corrected[6,6] = 1
_CNOT_02_expected_matrix_corrected[5,1] = 1; _CNOT_02_expected_matrix_corrected[7,3] = 1
_CNOT_02_expected_matrix_corrected[1,5] = 1; _CNOT_02_expected_matrix_corrected[3,7] = 1

def test_get_full_operator_single_qubit():
    X_q0_expected = torch.kron(X(), I_2x2)
    X_q0_actual = _get_full_operator(X(), qubits=(0,), total_qubits=2)
    torch.testing.assert_close(X_q0_actual, X_q0_expected)
    Y_q1_expected = torch.kron(I_2x2, Y())
    Y_q1_actual = _get_full_operator(Y(), qubits=(1,), total_qubits=2)
    torch.testing.assert_close(Y_q1_actual, Y_q1_expected)

def test_get_full_operator_two_qubit():
    CNOT_actual = _get_full_operator(CNOT(), qubits=(0,1), total_qubits=2)
    torch.testing.assert_close(CNOT_actual, CNOT())
    CNOT_02_actual = _get_full_operator(CNOT(), qubits=(0,2), total_qubits=3)
    torch.testing.assert_close(CNOT_02_actual, _CNOT_02_expected_matrix_corrected)

sim_cpu = CircuitSimulator(device='cpu')

def test_run_qsv_single_qubit_gates():
    psi_0 = get_zero_state_vector(1)
    qc_x = QuantumCircuit(1); qc_x.add_gate(X(), 0)
    psi_1_actual = sim_cpu.run(qc_x, psi_0)
    psi_1_expected = QuantumStateVector(1, initial_state_tensor=torch.tensor([0,1], dtype=COMPLEX_DTYPE))
    torch.testing.assert_close(psi_1_actual.state_vector, psi_1_expected.state_vector)

    qc_h = QuantumCircuit(1); qc_h.add_gate(H(), 0)
    psi_plus_actual = sim_cpu.run(qc_h, psi_0)
    psi_plus_expected_tensor = (1/math.sqrt(2)) * torch.tensor([1,1], dtype=COMPLEX_DTYPE)
    torch.testing.assert_close(psi_plus_actual.state_vector, psi_plus_expected_tensor)

    psi_1_qsv = QuantumStateVector(1, initial_state_tensor=torch.tensor([0,1], dtype=COMPLEX_DTYPE))
    psi_minus_actual = sim_cpu.run(qc_h, psi_1_qsv)
    psi_minus_expected_tensor = (1/math.sqrt(2)) * torch.tensor([1,-1], dtype=COMPLEX_DTYPE)
    torch.testing.assert_close(psi_minus_actual.state_vector, psi_minus_expected_tensor)

    qc_zh = QuantumCircuit(1); qc_zh.add_gate(H(),0); qc_zh.add_gate(Z(),0);
    psi_zh_actual = sim_cpu.run(qc_zh, get_zero_state_vector(1))
    torch.testing.assert_close(psi_zh_actual.state_vector, psi_minus_expected_tensor)

def test_run_qsv_two_qubit_gates():
    psi_00 = get_zero_state_vector(2)
    qc_cnot_on_10 = QuantumCircuit(2); qc_cnot_on_10.add_gate(X(),0); qc_cnot_on_10.add_gate(CNOT(), (0,1))
    psi_11_actual = sim_cpu.run(qc_cnot_on_10, psi_00)
    psi_11_expected_tensor = torch.tensor([0,0,0,1],dtype=COMPLEX_DTYPE)
    torch.testing.assert_close(psi_11_actual.state_vector, psi_11_expected_tensor)

    qc_bell = QuantumCircuit(2); qc_bell.add_gate(H(),0); qc_bell.add_gate(CNOT(), (0,1))
    bell_actual = sim_cpu.run(qc_bell, psi_00)
    bell_expected_tensor = (1/math.sqrt(2)) * torch.tensor([1,0,0,1], dtype=COMPLEX_DTYPE)
    torch.testing.assert_close(bell_actual.state_vector, bell_expected_tensor)

def test_run_qsv_multi_qubit_non_adjacent():
    psi_101_tensor = torch.zeros(8, dtype=COMPLEX_DTYPE); psi_101_tensor[0b101] = 1.0
    psi_101_init = QuantumStateVector(3, initial_state_tensor=psi_101_tensor)
    qc = QuantumCircuit(3); qc.add_gate(CNOT(), (0,2))
    psi_final_actual = sim_cpu.run(qc, psi_101_init)
    psi_100_expected_tensor = torch.zeros(8, dtype=COMPLEX_DTYPE); psi_100_expected_tensor[0b100] = 1.0
    torch.testing.assert_close(psi_final_actual.state_vector, psi_100_expected_tensor)

def test_run_qsv_empty_circuit():
    psi_0 = get_zero_state_vector(1)
    qc_empty = QuantumCircuit(1)
    psi_final = sim_cpu.run(qc_empty, psi_0)
    assert psi_final is not psi_0
    torch.testing.assert_close(psi_final.state_vector, psi_0.state_vector)
    assert psi_final.device == psi_0.device

@SKIP_CUDA_TESTS
def test_run_qsv_device_consistency():
    sim_cuda = CircuitSimulator(device='cuda')
    psi_cpu = get_zero_state_vector(1, device='cpu')
    qc = QuantumCircuit(1); qc.add_gate(X(),0)
    psi_final_cuda = sim_cuda.run(qc, psi_cpu)
    assert psi_final_cuda.device.type == 'cuda'
    expected_tensor_on_cuda = torch.tensor([0,1], dtype=COMPLEX_DTYPE, device=sim_cuda.device)
    torch.testing.assert_close(psi_final_cuda.state_vector, expected_tensor_on_cuda)

def test_run_dm_single_qubit_gates():
    rho_0 = get_zero_density_matrix(1)
    qc_x = QuantumCircuit(1); qc_x.add_gate(X(),0)
    rho_1_actual = sim_cpu.run(qc_x, rho_0)
    rho_1_expected_tensor = torch.zeros((2,2), dtype=COMPLEX_DTYPE); rho_1_expected_tensor[1,1] = 1.0
    torch.testing.assert_close(rho_1_actual.density_matrix, rho_1_expected_tensor)

    qc_h = QuantumCircuit(1); qc_h.add_gate(H(),0)
    rho_plus_actual = sim_cpu.run(qc_h, rho_0)
    rho_plus_expected_tensor = 0.5 * torch.tensor([[1,1],[1,1]], dtype=COMPLEX_DTYPE)
    torch.testing.assert_close(rho_plus_actual.density_matrix, rho_plus_expected_tensor)

def test_run_dm_consistency_with_qsv():
    psi_00_qsv = get_zero_state_vector(2)
    qc_bell = QuantumCircuit(2); qc_bell.add_gate(H(),0); qc_bell.add_gate(CNOT(),(0,1))
    bell_qsv_final = sim_cpu.run(qc_bell, psi_00_qsv)
    expected_dm_tensor = torch.outer(bell_qsv_final.state_vector, bell_qsv_final.state_vector.conj())
    rho_00_dm = get_zero_density_matrix(2)
    bell_dm_final = sim_cpu.run(qc_bell, rho_00_dm)
    torch.testing.assert_close(bell_dm_final.density_matrix, expected_dm_tensor)

def test_run_noise_on_qsv_converts_to_dm():
    psi_0_qsv = get_zero_state_vector(1)
    p = 0.3
    kraus_ops = depolarizing_channel(p)
    qc = QuantumCircuit(1); qc.add_kraus(kraus_ops, 0)
    final_state = sim_cpu.run(qc, psi_0_qsv)
    assert isinstance(final_state, DensityMatrix)
    expected_dm_tensor = torch.zeros((2,2), dtype=COMPLEX_DTYPE)
    expected_dm_tensor[0,0] = 1 - (2*p/3); expected_dm_tensor[1,1] = (2*p/3)
    torch.testing.assert_close(final_state.density_matrix, expected_dm_tensor, atol=1e-6)

def test_run_noise_amplitude_damping_on_dm():
    rho_1_tensor = torch.zeros((2,2), dtype=COMPLEX_DTYPE); rho_1_tensor[1,1]=1.0
    rho_1_dm = DensityMatrix(1, initial_density_matrix_tensor=rho_1_tensor)
    gamma = 0.4
    kraus_ops = amplitude_damping_channel(gamma)
    qc = QuantumCircuit(1); qc.add_kraus(kraus_ops, 0)
    final_dm = sim_cpu.run(qc, rho_1_dm)
    expected_final_tensor = torch.zeros((2,2), dtype=COMPLEX_DTYPE)
    expected_final_tensor[1,1] = 1 - gamma; expected_final_tensor[0,0] = gamma
    torch.testing.assert_close(final_dm.density_matrix, expected_final_tensor)

def test_run_noise_phase_damping_on_dm():
    plus_tensor = 0.5 * torch.tensor([[1,1],[1,1]], dtype=COMPLEX_DTYPE)
    plus_dm = DensityMatrix(1, initial_density_matrix_tensor=plus_tensor)
    gamma = 0.25
    kraus_ops = phase_damping_channel(gamma)
    qc = QuantumCircuit(1); qc.add_kraus(kraus_ops, 0)
    final_dm = sim_cpu.run(qc, plus_dm)
    expected_final_tensor = torch.tensor([[0.5, 0.5 * (1 - 2*gamma)], [0.5 * (1 - 2*gamma), 0.5]], dtype=COMPLEX_DTYPE)
    torch.testing.assert_close(final_dm.density_matrix, expected_final_tensor, atol=1e-6)

def test_run_gate_after_noise():
    psi_0 = get_zero_state_vector(1)
    p = 0.1
    qc = QuantumCircuit(1)
    qc.add_gate(H(), 0)
    qc.add_kraus(depolarizing_channel(p), 0)
    qc.add_gate(X(), 0)
    final_state = sim_cpu.run(qc, psi_0)
    assert isinstance(final_state, DensityMatrix)
    expected_final_tensor = 0.5 * torch.tensor([[1, 1-p],[1-p, 1]], dtype=COMPLEX_DTYPE)
    torch.testing.assert_close(final_state.density_matrix, expected_final_tensor, atol=1e-6)

def test_run_qubit_mismatch():
    qc = QuantumCircuit(1)
    initial_state_2q = get_zero_state_vector(2)
    with pytest.raises(ValueError, match="Circuit num_qubits .* does not match initial_state num_qubits"):
        sim_cpu.run(qc, initial_state_2q)
