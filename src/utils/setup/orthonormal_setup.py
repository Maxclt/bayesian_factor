import numpy as np

def compute_orthogonal_projection(subspace: np.ndarray, vector: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Compute the projection of `vector` onto the orthogonal complement of `subspace`.

    Parameters:
        subspace (np.ndarray): The subspace, either as a matrix (multiple vectors) or a single vector.
        vector (np.ndarray): The vector to project.
        epsilon (float): Small value to prevent division by zero.

    Returns:
        np.ndarray: The projection of `vector` onto the orthogonal complement of `subspace`.
    """
    if subspace.ndim == 1:  # Handle `subspace` as a single vector
        norm = max(epsilon, np.linalg.norm(subspace)**2)
        projection = np.outer(subspace, subspace) @ vector / norm
    elif subspace.ndim == 2 and subspace.size > 0:  # Handle `subspace` as a matrix
        projection = (
            subspace @ np.linalg.pinv(subspace.T @ subspace) @ subspace.T @ vector
        )
    else:  # Handle empty subspace
        projection = np.zeros_like(vector)

    # Compute the orthogonal component
    orthogonal_projection = vector - projection
    return orthogonal_projection


def compute_projection_norm(subspace: np.ndarray, vector: np.ndarray) -> float:
    """
    Compute the norm of the component of `vector` orthogonal to `subspace`.

    This function first projects `vector` onto the orthogonal complement of `subspace`
    using the `compute_orthogonal_projection` function, and then calculates the norm
    of the resulting orthogonal component.

    Parameters:
        subspace (np.ndarray): The subspace, either as a matrix (multiple vectors)
                               or a single vector. If it is empty, the function
                               assumes no projection and considers the full `vector`.
        vector (np.ndarray): The vector whose orthogonal component norm is to be computed.

    Returns:
        float: The norm of the orthogonal component of `vector` relative to `subspace`.
    """
    orthogonal_component = compute_orthogonal_projection(subspace, vector)
    # Return the norm of the orthogonal component
    return np.linalg.norm(orthogonal_component)
