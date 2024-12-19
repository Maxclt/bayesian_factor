import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import Lasso


def map_estimation(
    B: torch.tensor,
    Sigma: torch.tensor,
    Theta: torch.tensor,
    Y: torch.tensor,
    alpha: float,
    lambda0: float,
    lambda1: float,
    epsilon: float,
    num_var: int,
    num_obs: int,
    num_factors: int,
    convergence_criterion: float = 0.05,
    forced_stop: int = 500,
):
    """
    Get MAP estimates for B, Sigma and Theta.

    Args:
        B (torch.tensor): size (G*K)
        Sigma (torch.tensor): size (G)
        Theta (torch.tensor): size (K)
        Y (torch.tensor): size (G*n)
        alpha (float)
        lambda0 (float)
        lambda1 (float)
        epsilon (float)
        num_var(int): G
        num_obs (int): n
        num_factors (int): K
        convergence_criterion (float)
        forced_stop (int)

    Returns:
        B (torch.tensor): size (G*K)
        Sigma (torch.tensor): size (G)
        Theta (torch.tensor): size (K)
    """
    d_inf = np.inf
    count = 0
    B_star_old = torch.zeros(num_factors, num_var).to(B.device)
    while d_inf > convergence_criterion:

        # E-Step
        Omega, M = get_latent_features(B, Sigma, Y, num_factors)
        gamma = get_latent_indicators(B, Theta, lambda0, lambda1, epsilon)

        # M-Step
        ## Set new variables
        Y_tilde = F.pad(Y.T, (0, 0, 0, num_factors), mode='constant', value=0) # size ((K+n)*G)
        Omega_tilde = get_Omega_tilde(Omega, M, num_obs)
        B_star = torch.zeros(num_factors, num_var).to(B.device)
        Theta = torch.zeros(num_factors).to(B.device)

        ## Update
        for j in range(num_var):
            beta_j_star = get_loading(Y_tilde, Omega_tilde, B, Sigma, gamma, lambda0, lambda1, j)
            B_star[:, j] = beta_j_star
            sigma_j = get_variance(Y_tilde, Omega_tilde, beta_j_star, num_obs, j)
            Sigma[j] = sigma_j
        
        for k in range(num_factors):
            theta_k = get_weight(gamma, alpha, num_var, num_factors, k)
            Theta[k] = theta_k

        # Rotation Step
        A = get_rotation_matrix(Omega, M, num_obs)
        B = rotation(B_star.T, A)

        # Update distance
        d_inf = infinite_norm_distance(B_star, B_star_old)
        B_star_old = B_star
        count += 1
        if count > forced_stop:
            break

    return B, Sigma, Theta



######## E-Step
def get_latent_features(B, Sigma, Y, num_factors):
    """
    Extract the latent features mean.

    Args:
        B (torch.tensor): size (G*K)
        Sigma (torch.tensor): size (G)
        Y (torch.tensor): size (G*n)
        num_factors (int): K

    Returns:
        Omega (torch.tensor): size (K*n)
        M (torch.tensor): size (K*K)
    """

    precision = (
        torch.eye(num_factors, device=B.device)
        + B.T @ torch.diag(1 / Sigma) @ B
    )
    M = torch.linalg.inv(precision)
    Omega = M @ B.T @ torch.diag(1 / Sigma) @ Y

    return Omega, M
    
def get_latent_indicators(B, Theta, lambda0, lambda1, epsilon):
    """
    Extract the latent indicators means.

    Args:
        B (torch.tensor): size (G*K)
        Theta (torch.tensor): size (K)
        lambda0 (float)
        lambda1 (float)
        epsilon (float)

    Returns:
        gamma (torch.tensor): size (G*K)
    """
    gamma = (
        lambda1 * torch.exp(-lambda1 * torch.abs(B)) * Theta
    ) / (
        lambda0
        * torch.exp(-lambda0 * torch.abs(B))
        * (1 - Theta)
        + lambda1 * torch.exp(-lambda1 * torch.abs(B)) * Theta
        + epsilon
    )

    return gamma

######## M-Step

def get_cholesky_factor(matrix: torch.Tensor) -> torch.Tensor:
    """Returns the Cholesky factor (lower triangular matrix) of the input matrix."""
    # Ensure the input matrix is square and positive-definite
    if matrix.size(0) != matrix.size(1):
        raise ValueError("Input matrix must be square.")
    
    # Compute the Cholesky decomposition
    cholesky_factor = torch.cholesky(matrix)
    
    return cholesky_factor

def get_Omega_tilde(Omega, M, num_obs):
    """
    Get Omega_tilde = (Omega, sqrt(n)M_L).

    Args:
        Omega (torch.tensor): size (K*n)
        M (torch.tensor): size (K*K)
        num_obs (int)

    Returns:
        Omega_tilde (torch.tensor): size ((K+n)*K)    
    """
    M_L = get_cholesky_factor(M)
    low = np.sqrt(num_obs) * M_L
    Omega_tilde = torch.cat([Omega.T.unsqueeze(0), low.unsqueeze(0)], dim=1).squeeze()

    return Omega_tilde

def get_loading(Y_tilde, Omega_tilde, B, Sigma, gamma, lambda0, lambda1, j):
    """
    Computes the updated value of beta_j_star, as the result of a Lasso regression.

    beta_j_star = argmax_beta{
        -norm_2(Y_tilde[j].T - Omega_tile @ beta) - 2*Sigma[j]*sum_k^K(abs(B[j,k])*lambda_{jk})
        }
    where lambda_{jk} = lambda_0*(1 - gamma[j,k]) + lambda_1*gamma[j,k]

    Args:
        Y_tilde (torch.tensor): size (G*(K+n))
        Omega_tilde (torch.tensor): size (K*(K+n)) 
        B (torch.tensor): size (G*K)
        Sigma (torch.tensor): size (G)
        gamma (torch.tensor): size (G*K)
        lambda0 (float)
        lambda1 (float)
        j (int)

    Returns:
        beta_j_star (torch.tensor): size(K+n)
    """
    # Convert PyTorch tensors to NumPy arrays for compatibility with sklearn
    Y_tilde_np = Y_tilde.cpu().numpy()
    Omega_tilde_np = Omega_tilde.cpu().numpy()
    Sigma_np = Sigma.cpu().numpy()
    gamma_np = gamma.cpu().numpy()
    
    # Compute lambda_{jk} for the current j
    lambdas = lambda0 * (1 - gamma_np[j, :]) + lambda1 * gamma_np[j, :]

    # Define the Lasso regularization parameter for sklearn
    # This is an approx because sklearn implementation => ... + alpha*norm(B[j])
    # and not sum_k^K(abs(B[j,k])*lambda_{jk}
    alpha = 2 * Sigma_np[j] * np.mean(lambdas)

    # Prepare the Lasso regression
    lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)

    # Fit Lasso regression: solve the Lasso problem for this j
    # TODO: Im pretty sure the inputs are alright but alpha to high
    # so it forces beta to 0 which sucks
    lasso.fit(Omega_tilde_np, Y_tilde_np[:, j])

    # Store the result
    beta_j_star = lasso.coef_

    # Convert the result back to a PyTorch tensor
    return torch.tensor(beta_j_star)

def get_variance(Y_tilde, Omega_tilde, beta_j_star, num_obs, j):
    """
    Update the jth diagonal coefficient of Sigma.

    Args:
        Y_tilde (torch.tensor): size (G*(K+n))
        Omega_tilde (torch.tensor): size (K*(K+n))
        beta_j_star (torch.tensor): size(K+n)
        num_obs (int)
        j (int)

    Returns:
        sigma_j (float)
    """
    residuals = Y_tilde[:, j] - Omega_tilde @ beta_j_star
    sigma_j = (torch.sum(residuals**2) + 1) / (num_obs + 1)

    return sigma_j

    
def get_weight(gamma, alpha, num_var, num_factor, k):
    """
    Get theta_k.

    Args:
        gamma (torch.tensor): size (G*K)
        alpha (float)
        num_var (int)
        num_factor(int)
        k (int)

    Returns:
        theta_k (float)
    """
    indicators_sum = torch.sum(gamma[:, k])
    theta_k = (indicators_sum + alpha/num_factor - 1) / (alpha/num_factor + 1 + num_var - 2)

    return theta_k

######## Rotation Step

def get_rotation_matrix(Omega, M, num_obs):
    """
    Get A the rotation matrix.

    Args:
        Omega (torch.tensor): size (K*n)
        M (torch.tensor): size (K*K)
        num_obs (int)

    Returns:
        A (torch.tensor): size (K*K)    
    """
    A = (Omega @ Omega.T) / num_obs + M

    return A


def rotation(B_star, A):
    """
    Rotates B_star to get B.

    Args:
        B_star (torch.tensor): size (G*(K+n))
        A (torch.tensor): size (K*K)

    Returns:
        B (torch.tensor): size (G*K)
    """
    A_L = get_cholesky_factor(A) # size (K*K)

    return B_star @ A_L.float()

######## Convergence criterion

def infinite_norm_distance(A, B):
    """
    Computes the distance between two matrices in the infinite norm.

    Args:
        A (torch.Tensor): First matrix of size [m, n].
        B (torch.Tensor): Second matrix of size [m, n].

    Returns:
        float: The distance between A and B in the infinite norm.
    """
    # Ensure A and B have the same shape
    assert A.shape == B.shape, "Matrices A and B must have the same shape"

    # Compute the element-wise absolute difference
    diff = torch.abs(A - B)

    # Compute the col sums
    col_sums = torch.sum(diff, dim=0)

    # Take the maximum col sum
    infinite_norm = torch.max(col_sums).item()

    return infinite_norm
