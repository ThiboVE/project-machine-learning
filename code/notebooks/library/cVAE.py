from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.nn as nn
import torch


class cVAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        device,
        latent_dim
        ) -> None:

        super(cVAE, self).__init__()

        self.encoder = encoder.to(device)

        self.decoder = decoder.to(device)

        # self.encoder_mu = nn.Linear(..., latent_dim)
        # self.encoder_variance = nn.Linear(..., latent_dim)

    def _sample_latent(self, hidden_encoder):
        """Return the latent normal sample z ~ N(mu, sigma^2)"""

        mu = self.encoder_mu(hidden_encoder)
        log_sigma_sq = self.encdoer_variance(hidden_encoder)
        sigma_sq = torch.exp(0.5*log_sigma_sq)

        eps = torch.randn_like(sigma_sq).float().to(self.device)

        self.z_mean = mu
        self.z_sigma_sq = sigma_sq

        return mu + sigma_sq * eps  # Reparameterization trick