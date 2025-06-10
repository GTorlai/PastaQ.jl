import torch
from typing import List, Optional, Union
import numpy as np # Not strictly needed if using torch.prod, but was in original

COMPLEX_DTYPE = torch.complex64

class MPS:
    """
    Represents a Matrix Product State (MPS).

    The MPS is stored as a list of tensors, one for each site.
    The class follows a common convention for tensor shapes:
    - Site 0 (leftmost): `(physical_dim, right_bond_dim)` - Rank 2
    - Site i (middle): `(left_bond_dim, physical_dim, right_bond_dim)` - Rank 3
    - Site N-1 (rightmost): `(left_bond_dim, physical_dim)` - Rank 2

    Attributes:
        site_tensors (List[torch.Tensor]): The list of tensors defining the MPS.
        num_sites (int): The number of sites (length of the MPS chain).
        physical_dims (List[int]): List of physical dimensions for each site.
        bond_dims (List[int]): List of bond dimensions. `bond_dims[i]` is the
            dimension of the bond connecting site `i` and site `i+1`. Length is `num_sites - 1`.
        center_site (Optional[int]): If the MPS is in a canonical form, this indicates
            the site of the orthogonality center. Defaults to None.
        _device (torch.device): The PyTorch device of the MPS tensors.
        _dtype (torch.dtype): The PyTorch dtype of the MPS tensors.
    """
    def __init__(self, site_tensors: List[torch.Tensor], center_site: Optional[int] = None):
        """
        Initializes a Matrix Product State (MPS).

        Args:
            site_tensors (List[torch.Tensor]): A list of PyTorch tensors representing the MPS sites.
                Tensors must adhere to the shape convention:
                - `M[0]` (leftmost): `(physical_dim, right_bond_dim)`
                - `M[i]` (middle): `(left_bond_dim, physical_dim, right_bond_dim)`
                - `M[N-1]` (rightmost): `(left_bond_dim, physical_dim)`
                All tensors must be on the same device and have the same dtype.
            center_site (Optional[int]): The index of the orthogonality center if the MPS
                is in a canonical form. Defaults to None (generic MPS form).

        Raises:
            ValueError: If `site_tensors` is empty, if tensor ranks or shapes are
                        inconsistent with the MPS convention, if bond dimensions
                        do not match between adjacent tensors, or if `center_site`
                        is out of range.
            TypeError: If any element in `site_tensors` is not a PyTorch Tensor.
        """
        if not site_tensors:
            raise ValueError("site_tensors list cannot be empty.")

        self.site_tensors: List[torch.Tensor] = site_tensors
        self.num_sites: int = len(site_tensors)

        if not all(isinstance(t, torch.Tensor) for t in self.site_tensors):
            raise TypeError("All elements in site_tensors must be PyTorch Tensors.")

        self._device: torch.device = self.site_tensors[0].device
        self._dtype: torch.dtype = self.site_tensors[0].dtype

        # Check for consistent device and dtype across all tensors
        # Note: This was previously checking all(t.device == self._device and t.dtype == self._dtype).
        # A more robust way to check if there are multiple devices/dtypes:
        devices = {t.device for t in self.site_tensors}
        dtypes = {t.dtype for t in self.site_tensors}
        if len(devices) > 1 or len(dtypes) > 1:
            # Providing more info in the error message
            device_list_str = [str(d) for d in devices]
            dtype_list_str = [str(d) for d in dtypes]
            raise ValueError(
                f"All site_tensors must have the same device and dtype. "
                f"Found devices: {device_list_str}, dtypes: {dtype_list_str}"
            )

        self.physical_dims: List[int] = []
        self.bond_dims: List[int] = []

        for i, t in enumerate(self.site_tensors):
            if i == 0:
                if t.ndim != 2:
                    raise ValueError(f"Tensor at site 0 must be rank 2 (phys, D_0), got rank {t.ndim} with shape {t.shape}.")
                self.physical_dims.append(t.shape[0])
                if self.num_sites > 1:
                    self.bond_dims.append(t.shape[1])
            elif i == self.num_sites - 1:
                if t.ndim != 2:
                    raise ValueError(f"Tensor at site {i} (rightmost) must be rank 2 (D_{i-1}, phys), got rank {t.ndim} with shape {t.shape}.")
                self.physical_dims.append(t.shape[1])
                if self.num_sites > 1 and t.shape[0] != self.bond_dims[-1]: # Check bond only if num_sites > 1
                    raise ValueError(
                        f"Left bond dimension mismatch at site {i}: tensor shape {t.shape[0]} vs previous right bond {self.bond_dims[-1]}"
                    )
            else:
                if t.ndim != 3:
                    raise ValueError(f"Tensor at site {i} must be rank 3 (D_{i-1}, phys, D_i), got rank {t.ndim} with shape {t.shape}.")
                self.physical_dims.append(t.shape[1])
                if t.shape[0] != self.bond_dims[-1]:
                     raise ValueError(
                        f"Left bond dimension mismatch at site {i}: tensor shape {t.shape[0]} vs previous right bond {self.bond_dims[-1]}"
                    )
                self.bond_dims.append(t.shape[2])

        if center_site is not None:
            if not (0 <= center_site < self.num_sites):
                raise ValueError(f"center_site index {center_site} out of range [0, {self.num_sites-1}]")
        self.center_site: Optional[int] = center_site


    @property
    def device(self) -> torch.device:
        """The PyTorch device on which the MPS tensors are stored."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """The PyTorch dtype of the MPS tensors."""
        return self._dtype

    @staticmethod
    def product_state(physical_states: List[torch.Tensor],
                      device: Optional[Union[str, torch.device]] = None,
                      dtype: Optional[torch.dtype] = None) -> 'MPS':
        """
        Creates an MPS representing a product state of local quantum states.

        Each site tensor in the resulting MPS will have the minimal bond dimension of 1.

        Args:
            physical_states (List[torch.Tensor]): A list of 1D PyTorch tensors.
                Each tensor `physical_states[i]` represents the state vector for the
                i-th physical site (e.g., `torch.tensor([1,0])` for state |0>).
            device (Optional[Union[str, torch.device]]): The target PyTorch device for the MPS tensors.
                If None, the device is inferred from `physical_states[0]`. Defaults to None.
            dtype (Optional[torch.dtype]): The target PyTorch dtype for the MPS tensors.
                If None, the dtype is inferred from `physical_states[0]`. Defaults to None.

        Returns:
            MPS: An `MPS` object representing the specified product state.

        Raises:
            ValueError: If `physical_states` list is empty or if any element is not a 1D tensor.
        """
        if not physical_states:
            raise ValueError("physical_states list cannot be empty.")

        # Determine target device and dtype
        _device = torch.device(device) if device is not None else physical_states[0].device
        _dtype = dtype if dtype is not None else physical_states[0].dtype

        site_tensors: List[torch.Tensor] = []
        num_sites = len(physical_states)

        for i, state_vec in enumerate(physical_states):
            if state_vec.ndim != 1:
                raise ValueError(f"Each physical state must be a 1D tensor; site {i} has ndim {state_vec.ndim}.")

            current_tensor = state_vec.to(device=_device, dtype=_dtype)
            phys_dim = current_tensor.shape[0]

            if num_sites == 1:
                # M[0] shape: (phys_dim, 1) following the convention for the first site.
                site_tensors.append(current_tensor.reshape(phys_dim, 1))
                break
            if i == 0: # Leftmost site
                site_tensors.append(current_tensor.reshape(phys_dim, 1)) # Shape (phys_dim, D_0=1)
            elif i == num_sites - 1: # Rightmost site
                site_tensors.append(current_tensor.reshape(1, phys_dim)) # Shape (D_{N-2}=1, phys_dim)
            else: # Middle site
                site_tensors.append(current_tensor.reshape(1, phys_dim, 1)) # Shape (D_L=1, phys_dim, D_R=1)

        return MPS(site_tensors)

    def norm_squared(self) -> torch.Tensor:
        """
        Calculates the squared norm (<psi|psi>) of the MPS.

        This is done by contracting the MPS with its conjugate counterpart.
        The result should be a scalar tensor.

        Returns:
            torch.Tensor: A scalar tensor representing the squared norm of the MPS.
                          The dtype of the returned tensor is real.
        """
        if self.num_sites == 0: # Should be caught by __init__
            return torch.tensor(0.0, device=self.device, dtype=self.dtype.real if self.dtype.is_complex else self.dtype)

        op0 = self.site_tensors[0]
        # Contract physical leg of M[0] with its conjugate: M[0]* M[0]
        # M[0] is (P0, D0). M[0].conj() is (P0, D0_c). Sum over P0.
        # Result 'env' has shape (D0_c, D0)
        env = torch.tensordot(op0.conj(), op0, dims=([0],[0]))

        if self.num_sites == 1:
            # env has shape (D0_c, D0). For the norm to be scalar, we need to trace if D0 > 1.
            # If D0 = 1 (e.g. from product_state for a single site), env is (1,1).
            # trace of (1,1) matrix is just the element.
            # The norm^2 is Tr(M_dag M). M_dag M is what 'env' represents here.
            scalar_norm_sq = torch.trace(env)
            return scalar_norm_sq.real if self.dtype.is_complex else scalar_norm_sq

        # For num_sites > 1
        for i in range(1, self.num_sites - 1): # Middle tensors
            op_i = self.site_tensors[i] # Shape (D_left, P_i, D_right)
            # env has shape (D_left_c, D_left) from previous site.
            # C = tensordot(env, op_i, dims=([1],[0])) contracts D_left of env with D_left of op_i.
            # C has shape (D_left_c, P_i, D_right)
            C = torch.tensordot(env, op_i, dims=([1],[0]))
            # env = tensordot(op_i.conj(), C, dims=([0,1],[0,1])) contracts D_left_c and P_i.
            # op_i.conj() has shape (D_left_c, P_i, D_right_c)
            # C has shape (D_left_c, P_i, D_right)
            # Result env has shape (D_right_c, D_right)
            env = torch.tensordot(op_i.conj(), C, dims=([0,1], [0,1]))

        # Rightmost tensor M[N-1]
        op_last = self.site_tensors[self.num_sites - 1] # Shape (D_left, P_last)
        # env has shape (D_left_c, D_left)
        # C = tensordot(env, op_last, dims=([1],[0])) contracts D_left of env with D_left of op_last.
        # C has shape (D_left_c, P_last)
        C = torch.tensordot(env, op_last, dims=([1],[0]))
        # final_val = tensordot(C, op_last.conj(), dims=([0,1],[0,1])) contracts D_left_c and P_last.
        # C has shape (D_left_c, P_last)
        # op_last.conj() has shape (D_left_c, P_last)
        # Result final_val is scalar
        final_val = torch.tensordot(C, op_last.conj(), dims=([0,1],[0,1])) # Sum over all remaining indices

        # Squeeze to ensure scalar output if all bond dims were 1.
        # The result of the final tensordot should already be a 0-dim tensor (scalar).
        return final_val.real if self.dtype.is_complex else final_val


    def get_tensor(self, site: int) -> torch.Tensor:
        """
        Retrieves the tensor at a specific site in the MPS.

        Args:
            site (int): The index of the site (0 to num_sites - 1).

        Returns:
            torch.Tensor: The MPS tensor at the specified site.

        Raises:
            IndexError: If `site` index is out of range.
        """
        if not (0 <= site < self.num_sites):
            raise IndexError(f"Site index {site} out of range [0, {self.num_sites-1}]")
        return self.site_tensors[site]

    def physical_dim(self, site: int) -> int:
        """
        Returns the physical dimension of a specific site.

        Args:
            site (int): The index of the site (0 to num_sites - 1).

        Returns:
            int: The physical dimension at the specified site.

        Raises:
            IndexError: If `site` index is out of range.
        """
        if not (0 <= site < self.num_sites):
            raise IndexError(f"Site index {site} out of range [0, {self.num_sites-1}]")
        return self.physical_dims[site]

    def bond_dim_left(self, site: int) -> int:
        """
        Returns the left bond dimension of site `site` (D_{site-1}).

        By convention, the left bond dimension of the first site (site 0) is 1.

        Args:
            site (int): The index of the site (0 to num_sites - 1).

        Returns:
            int: The left bond dimension.

        Raises:
            IndexError: If `site` index is out of range.
        """
        if not (0 <= site < self.num_sites):
            raise IndexError(f"Site index {site} out of range [0, {self.num_sites-1}]")
        if site == 0:
            return 1
        # For M[i] (D_L, P, D_R) or M[N-1] (D_L, P), D_L is shape[0]
        return self.site_tensors[site].shape[0]

    def bond_dim_right(self, site: int) -> int:
        """
        Returns the right bond dimension of site `site` (D_{site}).

        By convention, the right bond dimension of the last site (site N-1) is 1.

        Args:
            site (int): The index of the site (0 to num_sites - 1).

        Returns:
            int: The right bond dimension.

        Raises:
            IndexError: If `site` index is out of range.
        """
        if not (0 <= site < self.num_sites):
            raise IndexError(f"Site index {site} out of range [0, {self.num_sites-1}]")
        if site == self.num_sites - 1:
            return 1

        # For M[0] (P, D_R), D_R is shape[1]
        # For M[i] (D_L, P, D_R), D_R is shape[2]
        tensor = self.site_tensors[site]
        if tensor.ndim == 2: # This is M[0]
            return tensor.shape[1]
        else: # This is M[i] (must be ndim 3 for i < N-1)
            return tensor.shape[2]

    def __repr__(self) -> str:
        return (f"MPS(num_sites={self.num_sites}, physical_dims={self.physical_dims}, "
                f"bond_dims={self.bond_dims}, center_site={self.center_site}, "
                f"device='{self.device}', dtype={self.dtype})")

    def __len__(self) -> int:
        """Returns the number of sites in the MPS."""
        return self.num_sites
