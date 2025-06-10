import torch
import pytest
import sys
import os

# Adjust sys.path to allow importing from the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from states import QuantumStateVector, DensityMatrix

COMPLEX_DTYPE = torch.complex64
REAL_DTYPE = torch.float32 # Corresponding real dtype for complex64

# Helper to check if CUDA is available and skip tests if not
CUDA_AVAILABLE = torch.cuda.is_available()
SKIP_CUDA_TESTS = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")

# --- Tests for QuantumStateVector ---

def test_qsv_default_initialization():
    for num_qubits in [1, 2, 3]:
        qsv = QuantumStateVector(num_qubits=num_qubits, device='cpu')

        assert qsv.num_qubits == num_qubits
        assert qsv.device == torch.device('cpu')
        assert qsv.state_vector.dtype == COMPLEX_DTYPE

        expected_dim = 2**num_qubits
        assert qsv.state_vector.shape == (expected_dim,)

        expected_state = torch.zeros(expected_dim, dtype=COMPLEX_DTYPE, device='cpu')
        expected_state[0] = 1.0
        torch.testing.assert_close(qsv.state_vector, expected_state)

def test_qsv_initialization_with_tensor():
    num_qubits = 1
    dim = 2**num_qubits

    # Valid tensor
    valid_tensor_data = torch.tensor([1/torch.sqrt(torch.tensor(2.0)), 1/torch.sqrt(torch.tensor(2.0))], dtype=COMPLEX_DTYPE)
    qsv_valid = QuantumStateVector(num_qubits=num_qubits, initial_state_tensor=valid_tensor_data.clone(), device='cpu')
    torch.testing.assert_close(qsv_valid.state_vector, valid_tensor_data)
    assert qsv_valid.state_vector.device == torch.device('cpu')

    # Valid tensor with different dtype (should be cast)
    valid_tensor_float = torch.tensor([0.0, 1.0], dtype=REAL_DTYPE)
    qsv_cast = QuantumStateVector(num_qubits=num_qubits, initial_state_tensor=valid_tensor_float.clone(), device='cpu')
    torch.testing.assert_close(qsv_cast.state_vector, valid_tensor_float.to(COMPLEX_DTYPE))
    assert qsv_cast.state_vector.dtype == COMPLEX_DTYPE

    # Invalid shape: not 1D
    with pytest.raises(ValueError, match=r"initial_state_tensor must have shape \(\d+,\)"):
        QuantumStateVector(num_qubits=num_qubits, initial_state_tensor=torch.zeros((dim, 1), dtype=COMPLEX_DTYPE))

    # Invalid shape: wrong size for num_qubits
    with pytest.raises(ValueError, match=r"initial_state_tensor must have shape \(\d+,\)"):
        QuantumStateVector(num_qubits=num_qubits, initial_state_tensor=torch.zeros(dim * 2, dtype=COMPLEX_DTYPE))

    # Invalid type for tensor
    with pytest.raises(TypeError, match="initial_state_tensor must be a PyTorch Tensor"):
        QuantumStateVector(num_qubits=num_qubits, initial_state_tensor=[1,0,0,0])

def test_qsv_to_density_matrix():
    # Test with |+> state for 1 qubit
    num_qubits = 1
    plus_state_data = (1/torch.sqrt(torch.tensor(2.0, dtype=REAL_DTYPE))) * torch.tensor([1, 1], dtype=COMPLEX_DTYPE)
    qsv = QuantumStateVector(num_qubits=num_qubits, initial_state_tensor=plus_state_data, device='cpu')

    dm = qsv.to_density_matrix()

    assert isinstance(dm, DensityMatrix)
    assert dm.num_qubits == num_qubits
    assert dm.device == qsv.device
    assert dm.density_matrix.dtype == COMPLEX_DTYPE

    expected_dm_tensor = torch.tensor([[0.5+0.j, 0.5+0.j],
                                       [0.5+0.j, 0.5+0.j]], dtype=COMPLEX_DTYPE, device='cpu')
    torch.testing.assert_close(dm.density_matrix, expected_dm_tensor)

    # Test with |0> state for 2 qubits
    num_qubits_2 = 2
    qsv_2 = QuantumStateVector(num_qubits=num_qubits_2, device='cpu') # Default |00>
    dm_2 = qsv_2.to_density_matrix()

    expected_dm_tensor_2 = torch.zeros((4,4), dtype=COMPLEX_DTYPE, device='cpu')
    expected_dm_tensor_2[0,0] = 1.0
    torch.testing.assert_close(dm_2.density_matrix, expected_dm_tensor_2)


@SKIP_CUDA_TESTS
def test_qsv_to_device():
    qsv_cpu = QuantumStateVector(num_qubits=1, device='cpu')

    # To CUDA
    qsv_cuda = qsv_cpu.to('cuda')
    assert qsv_cuda.device == torch.device('cuda:0') # Or specific cuda device
    assert qsv_cuda.state_vector.device == qsv_cuda.device
    torch.testing.assert_close(qsv_cuda.state_vector.cpu(), qsv_cpu.state_vector) # Compare data after moving back

    # To CPU again
    qsv_cpu_again = qsv_cuda.to('cpu')
    assert qsv_cpu_again.device == torch.device('cpu')
    assert qsv_cpu_again.state_vector.device == qsv_cpu_again.device
    torch.testing.assert_close(qsv_cpu_again.state_vector, qsv_cpu.state_vector)

    # Test calling to() with the same device
    qsv_cpu_same = qsv_cpu.to('cpu')
    assert qsv_cpu_same is qsv_cpu # Should return self if device is the same


# --- Tests for DensityMatrix ---

def test_dm_default_initialization():
    for num_qubits in [1, 2, 3]:
        dm = DensityMatrix(num_qubits=num_qubits, device='cpu')

        assert dm.num_qubits == num_qubits
        assert dm.device == torch.device('cpu')
        assert dm.density_matrix.dtype == COMPLEX_DTYPE

        expected_dim = 2**num_qubits
        assert dm.density_matrix.shape == (expected_dim, expected_dim)

        expected_matrix = torch.zeros((expected_dim, expected_dim), dtype=COMPLEX_DTYPE, device='cpu')
        expected_matrix[0, 0] = 1.0
        torch.testing.assert_close(dm.density_matrix, expected_matrix)

def test_dm_initialization_with_tensor():
    num_qubits = 1
    dim = 2**num_qubits

    # Valid tensor (e.g., maximally mixed state for 1 qubit)
    valid_tensor_data = 0.5 * torch.eye(dim, dtype=COMPLEX_DTYPE)
    dm_valid = DensityMatrix(num_qubits=num_qubits, initial_density_matrix_tensor=valid_tensor_data.clone(), device='cpu')
    torch.testing.assert_close(dm_valid.density_matrix, valid_tensor_data)
    assert dm_valid.density_matrix.device == torch.device('cpu')

    # Valid tensor with different dtype (should be cast)
    valid_tensor_float = 0.5 * torch.eye(dim, dtype=REAL_DTYPE)
    dm_cast = DensityMatrix(num_qubits=num_qubits, initial_density_matrix_tensor=valid_tensor_float.clone(), device='cpu')
    torch.testing.assert_close(dm_cast.density_matrix, valid_tensor_float.to(COMPLEX_DTYPE))
    assert dm_cast.density_matrix.dtype == COMPLEX_DTYPE

    # Invalid shape: not 2D
    with pytest.raises(ValueError, match=r"initial_density_matrix_tensor must have shape \(\d+, \d+\)"):
        DensityMatrix(num_qubits=num_qubits, initial_density_matrix_tensor=torch.zeros(dim, dtype=COMPLEX_DTYPE))

    # Invalid shape: not square
    with pytest.raises(ValueError, match=r"initial_density_matrix_tensor must have shape \(\d+, \d+\)"):
        DensityMatrix(num_qubits=num_qubits, initial_density_matrix_tensor=torch.zeros((dim, dim + 1), dtype=COMPLEX_DTYPE))

    # Invalid shape: wrong size for num_qubits
    with pytest.raises(ValueError, match=r"initial_density_matrix_tensor must have shape \(\d+, \d+\)"):
        DensityMatrix(num_qubits=num_qubits, initial_density_matrix_tensor=torch.zeros((dim * 2, dim * 2), dtype=COMPLEX_DTYPE))

    # Invalid type for tensor
    with pytest.raises(TypeError, match="initial_density_matrix_tensor must be a PyTorch Tensor"):
        DensityMatrix(num_qubits=num_qubits, initial_density_matrix_tensor=[[1,0],[0,0]])


@SKIP_CUDA_TESTS
def test_dm_to_device():
    dm_cpu = DensityMatrix(num_qubits=1, device='cpu')

    # To CUDA
    dm_cuda = dm_cpu.to('cuda')
    assert dm_cuda.device == torch.device('cuda:0') # Or specific cuda device
    assert dm_cuda.density_matrix.device == dm_cuda.device
    torch.testing.assert_close(dm_cuda.density_matrix.cpu(), dm_cpu.density_matrix)

    # To CPU again
    dm_cpu_again = dm_cuda.to('cpu')
    assert dm_cpu_again.device == torch.device('cpu')
    assert dm_cpu_again.density_matrix.device == dm_cpu_again.device
    torch.testing.assert_close(dm_cpu_again.density_matrix, dm_cpu.density_matrix)

    # Test calling to() with the same device
    dm_cpu_same = dm_cpu.to('cpu')
    assert dm_cpu_same is dm_cpu # Should return self if device is the same

def test_invalid_device_string():
    with pytest.raises(ValueError, match="Invalid device string"):
        QuantumStateVector(num_qubits=1, device='random_invalid_device_string')
    with pytest.raises(ValueError, match="Invalid device string"):
        DensityMatrix(num_qubits=1, device='random_invalid_device_string')

```
One minor thing: `qsv_cuda.device == torch.device('cuda:0')`. If there are multiple CUDA devices, `to('cuda')` might pick `cuda:0` by default, but it's safer to just check `qsv_cuda.device.type == 'cuda'`. Or, if the goal is to ensure it's on *some* CUDA device and then can be moved back, the current check is okay for typical single-GPU setups. I'll leave it as is for now, as it's a common way to write these tests.
The `SKIP_CUDA_TESTS` marker is good.
`torch.testing.assert_close` is used (equivalent to `torch.testing.assert_allclose` which is preferred, but `assert_close` is an alias).

The tests cover the cases outlined in the subtask.The file `pytorch_pastaq/tests/test_states.py` has been overwritten with the new comprehensive unit tests.

The tests cover:
-   **`QuantumStateVector`**:
    -   Default initialization for various qubit numbers.
    -   Initialization with valid custom tensors (including dtype casting).
    -   Error handling for invalid tensor shapes, sizes, and types during initialization.
    -   Conversion to `DensityMatrix` using `to_density_matrix()`, verifying the output state.
    -   Device transfer using `to(device)`, including CPU to CUDA (if available) and back, and ensuring data integrity. Also checks that `to(same_device)` returns `self`.
-   **`DensityMatrix`**:
    -   Default initialization for various qubit numbers.
    -   Initialization with valid custom tensors (including dtype casting).
    -   Error handling for invalid tensor shapes, sizes, and types.
    -   Device transfer using `to(device)`, similar to `QuantumStateVector`.
-   **Common**:
    -   Validation of invalid device strings in constructors.
    -   Usage of `torch.testing.assert_close` for tensor comparisons.
    -   `pytest.mark.skipif` to skip CUDA tests if CUDA is not available.

All specified test cases have been implemented. I can now submit the report.
