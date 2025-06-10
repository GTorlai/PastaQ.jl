import torch
from typing import List, Tuple, Union, Any, Optional

class QuantumCircuit:
    """
    Represents a quantum circuit as a sequence of operations (gates or Kraus channels)
    applied to specific qubits.

    Attributes:
        num_qubits (int): The total number of qubits in the circuit.
        operations (List[Tuple[Any, Tuple[int, ...], str]]): A list storing the
            operations. Each element is a tuple:
            - op_data (torch.Tensor or List[torch.Tensor]): The gate matrix or list of Kraus operators.
            - qubits_tuple (Tuple[int, ...]): The qubit indices this operation acts upon.
            - op_type_str (str): Type of operation, either "gate" or "kraus_channel".
    """
    def __init__(self, num_qubits: int):
        """Initializes a QuantumCircuit.

        Args:
            num_qubits: The total number of qubits in the circuit.
                        Must be a positive integer.

        Raises:
            ValueError: If num_qubits is not a positive integer.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("num_qubits must be a positive integer.")
        self.num_qubits = num_qubits
        self.operations: List[Tuple[Any, Tuple[int, ...], str]] = []

    def _validate_qubits(self, qubits: Union[int, Tuple[int, ...]],
                         expected_num_op_qubits: Optional[int] = None) -> Tuple[int, ...]:
        """
        Internal helper to validate qubit arguments and normalize to a tuple.

        Args:
            qubits: Qubit argument, can be a single int or a tuple of ints.
            expected_num_op_qubits: The number of qubits this operation is expected to act upon.
                                     If provided, validates that `len(qubits_tuple)` matches this.
        Returns:
            A tuple of validated qubit indices.
        Raises:
            TypeError: If qubit indices are not integers or `qubits` is not int/tuple.
            ValueError: If qubit indices are out of range, duplicated, or if the number
                        of provided qubits does not match `expected_num_op_qubits`.
        """
        if isinstance(qubits, int):
            qubits_tuple = (qubits,)
        elif isinstance(qubits, tuple):
            qubits_tuple = qubits
        else:
            raise TypeError("qubits must be an int or a tuple of ints.")

        num_qubits_in_tuple = len(qubits_tuple)

        if num_qubits_in_tuple == 0:
            raise ValueError("At least one qubit must be specified for an operation.")

        if expected_num_op_qubits is not None and num_qubits_in_tuple != expected_num_op_qubits:
            raise ValueError(
                f"Operation expected to act on {expected_num_op_qubits} qubit(s), "
                f"but {num_qubits_in_tuple} target qubit(s) provided: {qubits_tuple}."
            )

        for q_idx in qubits_tuple:
            if not isinstance(q_idx, int):
                raise TypeError(f"Qubit indices must be integers, got {type(q_idx).__name__} for index {q_idx}.")
            if not (0 <= q_idx < self.num_qubits):
                raise ValueError(
                    f"Qubit index {q_idx} is out of range for a circuit with "
                    f"{self.num_qubits} qubits (valid range [0, {self.num_qubits - 1}])."
                )

        if len(set(qubits_tuple)) != num_qubits_in_tuple:
            raise ValueError(f"Duplicate qubit indices specified: {qubits_tuple}.")

        return qubits_tuple


    def add_gate(self, gate_matrix: torch.Tensor, qubits: Union[int, Tuple[int, ...]]):
        """
        Adds a unitary gate operation to the circuit.

        Args:
            gate_matrix: The PyTorch tensor representing the unitary gate.
                         Its dimensions must be (2^k, 2^k) for a k-qubit gate.
            qubits: An integer for a single-qubit gate, or a tuple of k integers
                    for a k-qubit gate, specifying the target qubit(s). Qubit
                    indices must be unique and within `[0, num_qubits - 1]`.

        Raises:
            TypeError: If `gate_matrix` is not a PyTorch Tensor or if `qubits`
                       are not of the correct type.
            ValueError: If `gate_matrix` is not a square 2D tensor, its dimension
                        is not a power of 2, or if qubit indices are invalid (out of
                        range, duplicates, or number of qubits doesn't match gate matrix).
        """
        if not isinstance(gate_matrix, torch.Tensor):
            raise TypeError("gate_matrix must be a PyTorch Tensor.")

        if gate_matrix.ndim != 2 or gate_matrix.shape[0] != gate_matrix.shape[1]:
            raise ValueError(f"Gate matrix must be a square 2D tensor, got shape {gate_matrix.shape}.")

        dim = gate_matrix.shape[0]
        if not (dim > 0 and (dim & (dim - 1) == 0)): # Check if dim is a power of 2
            raise ValueError(f"Gate matrix dimension {dim} is not a power of 2.")
        num_gate_qubits_from_matrix = dim.bit_length() - 1

        qubits_tuple = self._validate_qubits(qubits, num_gate_qubits_from_matrix)

        # Redundant check for matrix dim vs qubits_tuple length, as _validate_qubits handles it.
        # Kept as a safeguard or for clarity if _validate_qubits changes.
        expected_dim_from_qubits = 2**len(qubits_tuple)
        if gate_matrix.shape != (expected_dim_from_qubits, expected_dim_from_qubits):
            raise ValueError(
                f"Gate matrix dimension {gate_matrix.shape} is inconsistent with "
                f"the number of specified qubits {len(qubits_tuple)} {qubits_tuple}. "
                f"Expected matrix shape for these qubits: ({expected_dim_from_qubits}, {expected_dim_from_qubits})."
            )

        self.operations.append((gate_matrix, qubits_tuple, "gate"))

    def add_kraus(self, kraus_operators: List[torch.Tensor], qubits: Union[int, Tuple[int, ...]]):
        """
        Adds a Kraus channel operation to the circuit.

        The number of qubits the channel acts upon is inferred from the dimension
        of the Kraus operators (e.g., 2x2 operators for a 1-qubit channel).
        All Kraus operators in the list must have the same dimensions.

        Args:
            kraus_operators: A non-empty list of PyTorch tensors, where each tensor
                             is a Kraus operator for the channel.
            qubits: An integer or a tuple of integers specifying the target qubit(s)
                    for the channel. The number of qubits must match the dimension
                    of the Kraus operators. Qubit indices must be unique and
                    within `[0, num_qubits - 1]`.

        Raises:
            TypeError: If `kraus_operators` is not a list of Tensors or if `qubits`
                       are not of the correct type.
            ValueError: If `kraus_operators` list is empty, or if Kraus operators
                        are not square 2D tensors, have inconsistent shapes, have
                        dimensions that are not a power of 2, or if qubit indices
                        are invalid.
        """
        if not isinstance(kraus_operators, list) or not kraus_operators:
            raise ValueError("kraus_operators must be a non-empty list of PyTorch Tensors.")
        if not all(isinstance(k, torch.Tensor) for k in kraus_operators):
            raise TypeError("All elements in kraus_operators must be PyTorch Tensors.")

        first_k_op = kraus_operators[0]
        if first_k_op.ndim != 2 or first_k_op.shape[0] != first_k_op.shape[1]:
            raise ValueError(f"Kraus operators must be square 2D tensors. Got shape {first_k_op.shape} for the first operator.")

        dim_k = first_k_op.shape[0]
        if not (dim_k > 0 and (dim_k & (dim_k - 1) == 0)):
            raise ValueError(f"Kraus operator dimension {dim_k} is not a power of 2.")
        num_channel_qubits = dim_k.bit_length() - 1

        for i, k_op in enumerate(kraus_operators):
            if k_op.shape != (dim_k, dim_k):
                raise ValueError(
                    f"Kraus operator at index {i} has shape {k_op.shape}, "
                    f"expected ({dim_k},{dim_k}) based on the first operator."
                )

        qubits_tuple = self._validate_qubits(qubits, num_channel_qubits)

        self.operations.append((kraus_operators, qubits_tuple, "kraus_channel"))


    def __repr__(self) -> str:
        op_list_str = []
        for op_data, q_tuple, op_type in self.operations:
            if op_type == "gate":
                tensor_repr = f"tensor(shape={op_data.shape}, dtype={op_data.dtype})"
                op_list_str.append(f"('gate', {tensor_repr}, {q_tuple})")
            elif op_type == "kraus_channel":
                if op_data:
                    k_repr = f"List[tensor(shape={op_data[0].shape}, dtype={op_data[0].dtype}) x {len(op_data)}]"
                else: # Should not happen due to validation in add_kraus
                    k_repr = "List[] (empty)"
                op_list_str.append(f"('kraus', {k_repr}, {q_tuple})")
            else: # Should not happen
                op_list_str.append(f"('unknown_op', {op_data}, {q_tuple})")
        return f"QuantumCircuit(num_qubits={self.num_qubits}, operations=[{', '.join(op_list_str)}])"

    def __len__(self) -> int:
        """Returns the number of operations (gates or channels) in the circuit."""
        return len(self.operations)
