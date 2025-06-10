import torch
import sys
import os

# Adjust path to import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tensor_network import MPS, COMPLEX_DTYPE as SRC_COMPLEX_DTYPE

# Use a consistent COMPLEX_DTYPE, preferably from a central source
COMPLEX_DTYPE = SRC_COMPLEX_DTYPE

def main():
    """Demonstrates basic MPS (Matrix Product State) operations."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Basic MPS Operations Example ---")
    print(f"Using device: {device}\n")

    # 1. Define local physical states for a product state
    # Example: |010> for a 3-qubit system (physical dimension 2 for each)
    state_q0 = torch.tensor([1, 0], dtype=COMPLEX_DTYPE, device=device)  # |0>
    state_q1 = torch.tensor([0, 1], dtype=COMPLEX_DTYPE, device=device)  # |1>
    state_q2 = torch.tensor([1, 0], dtype=COMPLEX_DTYPE, device=device)  # |0>

    physical_states_normalized = [state_q0, state_q1, state_q2]
    print(f"Target product state: [|0>, |1>, |0>] (normalized local states)\n")

    # 2. Create an MPS for the product state using the static factory method
    # The device and dtype for the MPS will be inferred from physical_states_normalized[0]
    # or can be explicitly passed to MPS.product_state(..., device=device, dtype=COMPLEX_DTYPE)
    mps_normalized = MPS.product_state(physical_states_normalized)

    # 3. Print its properties
    print(f"MPS (normalized) created for {mps_normalized.num_sites} sites.")
    print(f"  Device: {mps_normalized.device}")
    print(f"  Dtype: {mps_normalized.dtype}")
    print(f"  Physical dimensions: {mps_normalized.physical_dims}") # Expected: [2, 2, 2]
    print(f"  Bond dimensions: {mps_normalized.bond_dims}")       # Expected: [1, 1] for 3 sites

    print("\n  Site tensors shapes:")
    for i, tensor in enumerate(mps_normalized.site_tensors):
        print(f"    M[{i}]: {tensor.shape}") # M[0]:(2,1), M[1]:(1,2,1), M[2]:(1,2)
    print("\n")

    # 4. Calculate and print its norm squared
    # For normalized input states forming a product state, this should be 1.0
    norm_sq_normalized = mps_normalized.norm_squared()
    print(f"MPS (normalized) norm squared: {norm_sq_normalized.item():.6f}") # .item() for scalar Python number
    if torch.isclose(norm_sq_normalized, torch.tensor(1.0, device=device, dtype=norm_sq_normalized.dtype)):
        print("Verification successful: Normalized MPS norm is ~1.0.\n")
    else:
        print(f"Verification failed: Normalized MPS norm is {norm_sq_normalized.item()}.\n")


    # --- Example with unnormalized states to see norm_squared > 1 ---
    print("--- MPS with unnormalized local states ---")
    un_state_q0 = torch.tensor([1, 1], dtype=COMPLEX_DTYPE, device=device) # Norm^2 = 2
    un_state_q1 = torch.tensor([2, 0], dtype=COMPLEX_DTYPE, device=device) # Norm^2 = 4
    un_physical_states = [un_state_q0, un_state_q1]
    print(f"Target product state from unnormalized local states: [|0>+|1>, 2|0>]\n")

    un_mps = MPS.product_state(un_physical_states)
    print(f"MPS (unnormalized) created for {un_mps.num_sites} sites.")
    print(f"  Physical dimensions: {un_mps.physical_dims}") # Expected: [2, 2]
    print(f"  Bond dimensions: {un_mps.bond_dims}")       # Expected: [1]

    print("\n  Site tensors shapes:")
    for i, tensor in enumerate(un_mps.site_tensors):
        print(f"    M[{i}]: {tensor.shape}") # M[0]:(2,1), M[1]:(1,2)
    print("\n")

    un_norm_sq = un_mps.norm_squared()

    # Expected norm^2 = (<un_s0|un_s0>) * (<un_s1|un_s1>)
    # <un_state_q0|un_state_q0> = 1*1 + 1*1 = 2.0
    # <un_state_q1|un_state_q1> = 2*2 + 0*0 = 4.0
    # Expected norm_sq = 2.0 * 4.0 = 8.0
    expected_un_norm_sq = torch.tensor( (1**2 + 1**2) * (2**2 + 0**2), device=device, dtype=un_norm_sq.dtype)

    print(f"MPS (unnormalized) norm squared: {un_norm_sq.item():.6f}")
    if torch.isclose(un_norm_sq, expected_un_norm_sq):
        print(f"Verification successful: Unnormalized MPS norm is ~{expected_un_norm_sq.item()}.")
    else:
        print(f"Verification failed: Unnormalized MPS norm. Expected ~{expected_un_norm_sq.item()}, got {un_norm_sq.item()}.")

if __name__ == '__main__':
    main()
