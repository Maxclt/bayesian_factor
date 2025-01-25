import numpy as np


def compute_projection_norm(subspace: np.ndarray, vector: np.ndarray) -> float:
    # Compute the projection of vector onto the ortogonal space of subspace
    if subspace.size > 0:  # Ensure there are columns to project onto
        # (Omega_{-k}^T Omega_{-k})^-1 Omega_{-k}^T Omega_{k}
        projection = (
            subspace @ np.linalg.pinv(subspace.T @ subspace) @ subspace.T @ vector
        )
    else:
        projection = np.zeros_like(vector)

    # Compute the orthogonal component
    orthogonal_component = vector - projection

    # Return the norm of the orthogonal component
    return np.linalg.norm(orthogonal_component)
