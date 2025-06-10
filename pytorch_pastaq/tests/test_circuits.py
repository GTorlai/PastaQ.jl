import torch
import pytest
import sys
import os

# Adjust sys.path to allow importing from the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from circuits import QuantumCircuit
from gates import X, CNOT, COMPLEX_DTYPE as GATE_COMPLEX_DTYPE
from noise import depolarizing_channel

# Use the COMPLEX_DTYPE from the gates module for consistency
COMPLEX_DTYPE = GATE_COMPLEX_DTYPE

# --- Tests for QuantumCircuit Initialization ---

def test_qc_initialization():
    for n in [1, 2, 3]:
        qc = QuantumCircuit(num_qubits=n)
        assert qc.num_qubits == n
        assert qc.operations == []
        assert len(qc) == 0

def test_qc_init_invalid_num_qubits():
    with pytest.raises(ValueError, match="num_qubits must be a positive integer"):
        QuantumCircuit(num_qubits=0)
    with pytest.raises(ValueError, match="num_qubits must be a positive integer"):
        QuantumCircuit(num_qubits=-1)
    with pytest.raises(ValueError, match="num_qubits must be a positive integer"):
        QuantumCircuit(num_qubits="abc") # type: ignore

# --- Tests for add_gate() ---

def test_qc_add_single_qubit_gate():
    qc = QuantumCircuit(1)
    x_gate = X()
    qc.add_gate(x_gate, 0)
    assert len(qc) == 1
    op_data, op_qubits, op_type = qc.operations[0]
    assert torch.equal(op_data, x_gate)
    assert op_qubits == (0,)
    assert op_type == "gate"

def test_qc_add_two_qubit_gate():
    qc = QuantumCircuit(2)
    cnot_gate = CNOT()
    qc.add_gate(cnot_gate, (0, 1))
    assert len(qc) == 1
    op_data, op_qubits, op_type = qc.operations[0]
    assert torch.equal(op_data, cnot_gate)
    assert op_qubits == (0, 1)
    assert op_type == "gate"

def test_qc_add_gate_validation_qubit_index():
    qc1 = QuantumCircuit(1)
    x_gate = X()
    with pytest.raises(ValueError, match=r"Qubit index 1 is out of range"):
        qc1.add_gate(x_gate, 1)

    qc2 = QuantumCircuit(2)
    cnot_gate = CNOT()
    with pytest.raises(ValueError, match=r"Qubit index 2 is out of range"):
        qc2.add_gate(cnot_gate, (0, 2))

    with pytest.raises(ValueError, match=r"Duplicate qubit indices specified: \(0, 0\)"):
        qc2.add_gate(cnot_gate, (0,0)) # Using CNOT which expects 2 distinct qubits

    # Test providing more target qubits than the gate acts on
    with pytest.raises(ValueError, match=r"Operation expected to act on 1 qubit\(s\), but 2 target qubit\(s\) provided"):
        qc2.add_gate(x_gate, (0,1))


def test_qc_add_gate_validation_matrix_dim():
    qc1 = QuantumCircuit(1)
    cnot_gate = CNOT() # 2-qubit gate
    with pytest.raises(ValueError, match=r"Operation expected to act on 2 qubit\(s\), but 1 target qubit\(s\) provided"):
        qc1.add_gate(cnot_gate, 0)

    qc2 = QuantumCircuit(2)
    x_gate = X() # 1-qubit gate
    # This is now caught by the check in _validate_qubits because num_gate_qubits_from_matrix (1)
    # doesn't match len(qubits_tuple) which is 2.
    with pytest.raises(ValueError, match=r"Operation expected to act on 1 qubit\(s\), but 2 target qubit\(s\) provided"):
        qc2.add_gate(x_gate, (0, 1))

    # Test non-power-of-2 matrix dimension
    qc_bad_dim = QuantumCircuit(1)
    bad_matrix = torch.randn(3,3, dtype=COMPLEX_DTYPE)
    with pytest.raises(ValueError, match="Gate matrix dimension 3 is not a power of 2"):
        qc_bad_dim.add_gate(bad_matrix, 0)

def test_qc_add_gate_non_tensor_matrix():
    qc = QuantumCircuit(1)
    with pytest.raises(TypeError, match="gate_matrix must be a PyTorch Tensor"):
        qc.add_gate("not_a_tensor", 0) # type: ignore

# --- Tests for add_kraus() ---

def test_qc_add_single_qubit_kraus():
    qc = QuantumCircuit(1)
    kraus_ops = depolarizing_channel(0.1) # List of 2x2 tensors
    qc.add_kraus(kraus_ops, 0)
    assert len(qc) == 1
    op_data, op_qubits, op_type = qc.operations[0]
    assert op_data == kraus_ops # Check if the list itself is stored
    for i in range(len(kraus_ops)):
        assert torch.equal(op_data[i], kraus_ops[i])
    assert op_qubits == (0,)
    assert op_type == "kraus_channel"

def test_qc_add_multi_qubit_kraus():
    qc = QuantumCircuit(2)
    # Dummy 2-qubit Kraus channel (e.g., just identity)
    # K0 = I_4x4, K1 = 0 ... (ensure sum K_dag K = I)
    # For simplicity, just use one op for testing structure
    kraus_op_4x4 = torch.eye(4, dtype=COMPLEX_DTYPE)
    dummy_2q_kraus_ops = [kraus_op_4x4 * np.sqrt(0.5), kraus_op_4x4 * np.sqrt(0.5)] # Example

    qc.add_kraus(dummy_2q_kraus_ops, (0, 1))
    assert len(qc) == 1
    op_data, op_qubits, op_type = qc.operations[0]
    assert op_data == dummy_2q_kraus_ops
    assert op_qubits == (0,1)
    assert op_type == "kraus_channel"

def test_qc_add_kraus_validation_qubit_index():
    qc1 = QuantumCircuit(1)
    kraus_ops_1q = depolarizing_channel(0.1)
    with pytest.raises(ValueError, match=r"Qubit index 1 is out of range"):
        qc1.add_kraus(kraus_ops_1q, 1)

    qc2 = QuantumCircuit(2)
    kraus_op_4x4 = torch.eye(4, dtype=COMPLEX_DTYPE)
    dummy_2q_kraus_ops = [kraus_op_4x4]
    with pytest.raises(ValueError, match=r"Qubit index 2 is out of range"):
        qc2.add_kraus(dummy_2q_kraus_ops, (0,2))

    with pytest.raises(ValueError, match=r"Duplicate qubit indices specified: \(0, 0\)"):
         qc2.add_kraus(dummy_2q_kraus_ops, (0,0))


def test_qc_add_kraus_validation_ops_format():
    qc = QuantumCircuit(1)
    # Not a list
    with pytest.raises(ValueError, match="kraus_operators must be a non-empty list"):
        qc.add_kraus("not_a_list", 0) # type: ignore
    # Empty list
    with pytest.raises(ValueError, match="kraus_operators must be a non-empty list"):
        qc.add_kraus([], 0)
    # List contains non-tensor
    with pytest.raises(TypeError, match="All elements in kraus_operators must be PyTorch Tensors"):
        qc.add_kraus([torch.eye(2, dtype=COMPLEX_DTYPE), "not_a_tensor"], 0) # type: ignore
    # Non-square tensor
    with pytest.raises(ValueError, match="Kraus operators must be square 2D tensors"):
        qc.add_kraus([torch.randn(2,3, dtype=COMPLEX_DTYPE)], 0)
    # Tensors of different shapes
    with pytest.raises(ValueError, match=r"Kraus operator at index 1 has shape .* expected .* based on the first operator"):
        qc.add_kraus([torch.eye(2, dtype=COMPLEX_DTYPE), torch.eye(4, dtype=COMPLEX_DTYPE)], 0)
    # Kraus op dimension not matching number of target qubits
    kraus_ops_1q = depolarizing_channel(0.1) # for 1 qubit
    qc2 = QuantumCircuit(2)
    with pytest.raises(ValueError, match=r"Operation expected to act on 1 qubit\(s\), but 2 target qubit\(s\) provided"):
        qc2.add_kraus(kraus_ops_1q, (0,1))

    kraus_ops_2q = [torch.eye(4, dtype=COMPLEX_DTYPE)] # for 2 qubits
    qc1 = QuantumCircuit(1)
    with pytest.raises(ValueError, match=r"Operation expected to act on 2 qubit\(s\), but 1 target qubit\(s\) provided"):
        qc1.add_kraus(kraus_ops_2q, 0)

# --- Tests for __repr__ ---

def test_qc_repr():
    qc = QuantumCircuit(num_qubits=2)
    assert "num_qubits=2" in repr(qc)
    assert "operations=[]" in repr(qc)

    qc.add_gate(X(), 0)
    rep_str = repr(qc)
    assert "'gate'" in rep_str
    assert "tensor(shape=torch.Size([2, 2])" in rep_str
    assert "(0,)" in rep_str # Qubit tuple

    kraus_ops = depolarizing_channel(0.1)
    qc.add_kraus(kraus_ops, 1)
    rep_str_2 = repr(qc)
    assert "'kraus'" in rep_str_2
    assert f"List[tensor(shape={kraus_ops[0].shape}" in rep_str_2
    assert f"x {len(kraus_ops)}]" in rep_str_2
    assert "(1,)" in rep_str_2 # Qubit tuple for Kraus

# --- Tests for __len__ ---

def test_qc_len():
    qc = QuantumCircuit(2)
    assert len(qc) == 0
    qc.add_gate(X(), 0)
    assert len(qc) == 1
    qc.add_kraus(depolarizing_channel(0.1), 1)
    assert len(qc) == 2
    qc.add_gate(CNOT(), (0,1))
    assert len(qc) == 3
