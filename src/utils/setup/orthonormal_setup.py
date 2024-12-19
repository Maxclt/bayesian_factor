import numpy as np

def compute_projection_norm(Omega_minus_k, bar_Omega_k):
        # Compute the projection onto the span of Omega_{-k}
        if Omega_minus_k.size > 0:  # Ensure there are columns to project onto
            # (Omega_{-k}^T Omega_{-k})^-1 Omega_{-k}^T Omega_{k}
            projection = Omega_minus_k @ np.linalg.pinv(Omega_minus_k.T @ Omega_minus_k) @ Omega_minus_k.T @ bar_Omega_k
        else:
            projection = np.zeros_like(bar_Omega_k)

        # Compute the orthogonal component
        orthogonal_component = bar_Omega_k - projection

        # Return the norm of the orthogonal component
        return np.linalg.norm(orthogonal_component)
    
