import numpy as np


def metropolis_hastings(
    target_pdf: function,
    proposal_sampler: function,
    initial_state: float,
    burn_in: int = 100,
) -> float:
    """Perform Metropolis-Hastings sampling.

    Args:
        target_pdf (function): _description_
        proposal_sampler (function): _description_
        initial_state (float): _description_
        burn_in (int, optional): _description_. Defaults to 100.

    Returns:
        float: _description_
    """
    # Initialize variables
    current_state = initial_state
    current_pdf = target_pdf(current_state)

    for _ in range(burn_in):
        # Propose a new state
        proposed_state = proposal_sampler(current_state)
        proposed_pdf = target_pdf(proposed_state)

        # Acceptance probability
        acceptance_ratio = proposed_pdf / current_pdf

        # Decide whether to accept the new state
        if np.random.rand() < acceptance_ratio:
            current_state = proposed_state
            current_pdf = proposed_pdf

    return current_state
