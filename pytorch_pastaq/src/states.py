import torch
from typing import Optional, Union # Union for device type hint

COMPLEX_DTYPE = torch.complex64 # Consistent complex type

class QuantumStateVector:
    """
    Represents a quantum state vector for a system of qubits.

    The state vector is stored as a 1D PyTorch tensor of complex numbers.
    The ordering of amplitudes corresponds to the computational basis states
    (e.g., for 2 qubits: |00>, |01>, |10>, |11>).

    Attributes:
        num_qubits (int): The number of qubits in the system.
        dim (int): The dimension of the Hilbert space (2**num_qubits).
        device (torch.device): The PyTorch device where the state vector is stored.
        state_vector (torch.Tensor): The 1D tensor representing the quantum state.
                                     Shape is (2**num_qubits,).
    """
    def __init__(self,
                 num_qubits: int,
                 initial_state_tensor: Optional[torch.Tensor] = None,
                 device: Union[str, torch.device] = 'cpu'):
        """Initializes a QuantumStateVector.

        Args:
            num_qubits: The number of qubits. Must be a positive integer.
            initial_state_tensor: Optional PyTorch tensor to initialize the state vector.
                If provided, it must be a 1D tensor of shape (2**num_qubits,)
                and will be converted to COMPLEX_DTYPE. If None, the state is
                initialized to |0...0>.
            device: The PyTorch device (e.g., 'cpu', 'cuda') where the tensor
                should be stored. Defaults to 'cpu'.

        Raises:
            ValueError: If num_qubits is not positive, or if initial_state_tensor
                        has an incorrect shape, or if device string is invalid.
            TypeError: If initial_state_tensor is not a PyTorch Tensor.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("num_qubits must be a positive integer.")
        self.num_qubits = num_qubits
        self.dim = 2**num_qubits

        try:
            self.device = torch.device(device)
        except RuntimeError:
            raise ValueError(f"Invalid device string: {device}")

        if initial_state_tensor is not None:
            if not isinstance(initial_state_tensor, torch.Tensor):
                raise TypeError("initial_state_tensor must be a PyTorch Tensor.")
            if initial_state_tensor.shape != (self.dim,):
                raise ValueError(
                    f"initial_state_tensor must have shape ({self.dim},), "
                    f"but got {initial_state_tensor.shape}"
                )
            # Ensure tensor is on the correct device and dtype
            self.state_vector = initial_state_tensor.to(device=self.device, dtype=COMPLEX_DTYPE)
        else:
            self.state_vector = torch.zeros(self.dim, dtype=COMPLEX_DTYPE, device=self.device)
            self.state_vector[0] = 1.0

    def to(self, device: Union[str, torch.device]) -> 'QuantumStateVector':
        """
        Moves the state vector to the specified PyTorch device.

        Args:
            device: The target PyTorch device (e.g., 'cpu', 'cuda').

        Returns:
            A new QuantumStateVector object on the specified device if the device
            is different from the current one, otherwise returns self.
        """
        new_device = torch.device(device)
        if new_device == self.device:
            return self
        new_tensor = self.state_vector.to(new_device)
        # Create a new instance, as device is tied to the tensor instance
        return QuantumStateVector(self.num_qubits, initial_state_tensor=new_tensor, device=new_device)

    def to_density_matrix(self) -> 'DensityMatrix':
        """
        Converts the state vector to its corresponding density matrix.

        The conversion is done by computing the outer product: rho = |psi><psi|.

        Returns:
            A DensityMatrix object representing the pure state rho.
        """
        # rho = |psi><psi| = psi.unsqueeze(1) @ psi.conj().unsqueeze(0)
        # Corrected: psi is (dim,); psi.unsqueeze(1) is (dim,1); psi.conj().unsqueeze(0) is (1,dim)
        density_matrix_tensor = self.state_vector.unsqueeze(1) @ self.state_vector.conj().unsqueeze(0)
        return DensityMatrix(self.num_qubits, initial_density_matrix_tensor=density_matrix_tensor, device=self.device)

    def __repr__(self) -> str:
        return (f"QuantumStateVector(num_qubits={self.num_qubits}, "
                f"device='{self.device}', state_vector=\n{self.state_vector})")

class DensityMatrix:
    """
    Represents a quantum density matrix for a system of qubits.

    The density matrix is stored as a 2D PyTorch tensor of complex numbers.
    The rows and columns correspond to the computational basis states.

    Attributes:
        num_qubits (int): The number of qubits in the system.
        dim (int): The dimension of the Hilbert space (2**num_qubits).
        device (torch.device): The PyTorch device where the density matrix is stored.
        density_matrix (torch.Tensor): The 2D tensor representing the quantum state.
                                       Shape is (2**num_qubits, 2**num_qubits).
    """
    def __init__(self,
                 num_qubits: int,
                 initial_density_matrix_tensor: Optional[torch.Tensor] = None,
                 device: Union[str, torch.device] = 'cpu'):
        """Initializes a DensityMatrix.

        Args:
            num_qubits: The number of qubits. Must be a positive integer.
            initial_density_matrix_tensor: Optional PyTorch tensor to initialize
                the density matrix. If provided, it must be a 2D tensor of shape
                (2**num_qubits, 2**num_qubits) and will be converted to COMPLEX_DTYPE.
                If None, the state is initialized to |0...0><0...0|.
            device: The PyTorch device (e.g., 'cpu', 'cuda') where the tensor
                should be stored. Defaults to 'cpu'.

        Raises:
            ValueError: If num_qubits is not positive, or if initial_density_matrix_tensor
                        has an incorrect shape, or if device string is invalid.
            TypeError: If initial_density_matrix_tensor is not a PyTorch Tensor.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("num_qubits must be a positive integer.")
        self.num_qubits = num_qubits
        self.dim = 2**num_qubits

        try:
            self.device = torch.device(device)
        except RuntimeError:
            raise ValueError(f"Invalid device string: {device}")

        if initial_density_matrix_tensor is not None:
            if not isinstance(initial_density_matrix_tensor, torch.Tensor):
                raise TypeError("initial_density_matrix_tensor must be a PyTorch Tensor.")
            if initial_density_matrix_tensor.shape != (self.dim, self.dim):
                raise ValueError(
                    f"initial_density_matrix_tensor must have shape ({self.dim}, {self.dim}), "
                    f"but got {initial_density_matrix_tensor.shape}"
                )
            # Ensure tensor is on the correct device and dtype
            self.density_matrix = initial_density_matrix_tensor.to(device=self.device, dtype=COMPLEX_DTYPE)
        else:
            self.density_matrix = torch.zeros((self.dim, self.dim), dtype=COMPLEX_DTYPE, device=self.device)
            self.density_matrix[0, 0] = 1.0

    def to(self, device: Union[str, torch.device]) -> 'DensityMatrix':
        """
        Moves the density matrix to the specified PyTorch device.

        Args:
            device: The target PyTorch device (e.g., 'cpu', 'cuda').

        Returns:
            A new DensityMatrix object on the specified device if the device
            is different from the current one, otherwise returns self.
        """
        new_device = torch.device(device)
        if new_device == self.device:
            return self
        new_tensor = self.density_matrix.to(new_device)
        # Create a new instance, as device is tied to the tensor instance
        return DensityMatrix(self.num_qubits, initial_density_matrix_tensor=new_tensor, device=new_device)

    def __repr__(self) -> str:
        return (
            f"DensityMatrix(num_qubits={self.num_qubits}, device='{self.device}', "
            f"density_matrix=\n{self.density_matrix})"
        )
