from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.nn as nn
import random
import torch


class cVAE(nn.Module):
    def __init__(self, encoder, decoder, device,
                 latent_dim, gru_dim, vocab_size, embedding_dim,
                 teacher_forcing_ratio=0.5):
        super(cVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.latent_dim = latent_dim
        self.gru_dim = gru_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Token embedding (for decoder)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Latent space projections
        self.encoder_mu = nn.Linear(gru_dim, latent_dim)
        self.encoder_logvar = nn.Linear(gru_dim, latent_dim)

    def _sample_latent(self, hidden_encoder):
        """
        Reparameterization trick: z ~ N(mu, sigma^2)
        """
        mu = self.encoder_mu(hidden_encoder)
        logvar = self.encoder_logvar(hidden_encoder)
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma).to(self.device)
        z = mu + sigma * eps

        # Save for loss calculation
        self.z_mean = mu
        self.z_logvar = logvar

        return z

    def forward_decoder(self, z, x, y):
        """
        Autoregressive decoding
        z: [batch, latent_dim]
        x: [batch, seq_len] token indices
        y: [batch, 1] property vector
        """
        batch_size, target_len = x.size()
        device = x.device

        outputs = torch.zeros(batch_size, target_len, self.vocab_size).to(device)

        # Initialize first token as <STR> (index 2)
        input_token = torch.ones(batch_size, dtype=torch.long).to(device) * 2
        outputs[:,0,2] = 1

        # Initialize hidden state (zeros)
        hidden = torch.zeros(self.decoder.n_layers, batch_size, self.decoder.gru_size).to(device)

        for t in range(1, target_len):
            output, hidden = self.decoder(input_token, z, y, hidden)
            outputs[:, t, :] = output

            # Get predicted token
            top1 = output.argmax(1)

            # Teacher forcing
            if random.random() < self.teacher_forcing_ratio:
                input_token = x[:, t]  # ground truth
            else:
                input_token = top1.detach()

        return outputs

    def forward(self, x, y):
        """
        Full forward pass
        x: [batch, seq_len] token indices
        y: [batch, 1] property vector
        """
        # Encode graph to hidden representation
        hidden_encoder = self.encoder(x)  # should output [batch, gru_dim]

        # Sample latent vector
        z = self._sample_latent(hidden_encoder)

        # Decode sequence
        recon_x = self.forward_decoder(z, x, y)

        return recon_x
    


class GRU_Decoder(nn.Module):
    def __init__(self, vocab_size, latent_dim, gru_size, n_layers, embedding_dim):
        super(GRU_Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.gru_size = gru_size
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim

        # Embedding layer for input tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # GRU input size: token embedding + latent vector + property
        self.gru = nn.GRU(embedding_dim + latent_dim + 1, gru_size, n_layers, batch_first=True)

        # Output layer: project GRU hidden state to vocab size
        self.fc = nn.Linear(gru_size, vocab_size)

    def forward(self, input_token, z, y, hidden):
        """
        input_token: [batch] token indices
        z: [batch, latent_dim]
        y: [batch, 1] property vector
        hidden: [n_layers, batch, gru_size]
        """
        # Embed token
        token_embed = self.embedding(input_token)  # [batch, embed_dim]

        # Concatenate token embedding + latent + property
        decoder_input = torch.cat([token_embed, z, y], dim=1).unsqueeze(1)  # [batch, 1, embed+latent+1]

        # GRU forward
        output, hidden = self.gru(decoder_input, hidden)  # output: [batch, 1, gru_size]

        # Project to vocab
        output = self.fc(output.squeeze(1))  # [batch, vocab_size]

        return output, hidden