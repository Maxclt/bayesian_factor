import numpy as 

def sample_d_metropolis(self, k, bar_Omega_k, bar_sigma_k):
    def log_prob(d, proj):
        return (self.num_obs - d**2) ** (
            (self.num_obs - self.num_factor - 2) / 2
        ) * torch.exp(torch.norm(proj) * d / bar_sigma_k**2)

    d = torch.empty(1, device=self.device).uniform_(
        -torch.sqrt(torch.tensor(self.num_obs, device=self.device)),
        torch.sqrt(torch.tensor(self.num_obs, device=self.device)),
    )
    proposal_scale = 0.1

    Omega_minus_k = torch.cat((self.Omega[:k], self.Omega[k + 1 :]), dim=0).T
    proj = compute_projection_norm(Omega_minus_k, bar_Omega_k)

    for _ in range(self.burn_in):
        d_proposal = d + torch.randn(1, device=self.device) * proposal_scale
        if abs(d_proposal) > torch.sqrt(torch.tensor(self.num_obs, device=self.device)):
            continue
        acceptance_ratio = log_prob(d_proposal, proj) / log_prob(d, proj)
        if torch.rand(1, device=self.device) < acceptance_ratio:
            d = d_proposal

    return d.item()
