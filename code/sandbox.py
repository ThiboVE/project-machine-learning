from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.nn as nn
import random
import torch

class CVAE(nn.Module):
    def __init__(
        self,
        vocab_size,
        latent_size,
        num_prop,
        unit_size,
        n_rnn_layer,
        embedding_dim,
    ):
        """
        PyTorch version of your TensorFlow CVAE.
        """

        super(CVAE, self).__init__()

        self.vocab_size = vocab_size
        self.latent_size = latent_size
        self.num_prop = num_prop
        self.unit_size = unit_size
        self.n_rnn_layer = n_rnn_layer
        self.embedding_dim = embedding_dim

        # ------------------------------
        #  Embeddings
        # ------------------------------
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # ------------------------------
        # Encoder (LSTM)
        # ------------------------------
        self.encoder = nn.LSTM(
            input_size=embedding_dim + num_prop,
            hidden_size=unit_size,
            num_layers=n_rnn_layer,
            batch_first=True,
            bidirectional=False,
        )

        # Projection to mean/logvar
        self.linear_mean = nn.Linear(unit_size, latent_size)
        self.linear_logvar = nn.Linear(unit_size, latent_size)

        # ------------------------------
        # Decoder (LSTM)
        # ------------------------------
        self.decoder = nn.LSTM(
            input_size=embedding_dim + num_prop + latent_size,
            hidden_size=unit_size,
            num_layers=n_rnn_layer,
            batch_first=True,
            bidirectional=False,
        )

        # Output projection
        self.output_proj = nn.Linear(unit_size, vocab_size)

    # ======================================================
    #  Reparameterization trick
    # ======================================================
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ======================================================
    #  Encoder forward
    # ======================================================
    def encode(self, x, c, lengths):
        """
        x: token sequence [B, T]
        c: properties [B, num_prop]
        lengths: actual lengths [B]
        """
        emb = self.embedding(x)   # [B, T, embedding_dim]

        # Tile condition across time
        c_rep = c.unsqueeze(1).repeat(1, emb.size(1), 1)

        # Encoder input
        enc_in = torch.cat([emb, c_rep], dim=-1)

        # Pack for variable lengths
        packed = pack_padded_sequence(enc_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.encoder(packed)

        # Take top layer hidden state
        h_final = h_n[-1]    # [B, unit_size]

        # Mean and log variance
        mu = self.linear_mean(h_final)
        logvar = self.linear_logvar(h_final)

        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    # ======================================================
    #  Decoder forward
    # ======================================================
    def decode(self, x, z, c, lengths):
        """
        x: input tokens [B, T]
        z: latent vector [B, latent_size]
        c: condition [B, num_prop]
        """
        emb = self.embedding(x)  # [B, T, embed]

        # Repeat latent vector for each time step
        z_rep = z.unsqueeze(1).repeat(1, emb.size(1), 1)

        # Repeat condition
        c_rep = c.unsqueeze(1).repeat(1, emb.size(1), 1)

        # Decoder input
        dec_in = torch.cat([emb, z_rep, c_rep], dim=-1)

        # Pack
        packed = pack_padded_sequence(dec_in, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.decoder(packed)
        dec_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Predict logits
        logits = self.output_proj(dec_out)  # [B, T, vocab]

        return logits

    # ======================================================
    #  Forward pass: encode → decode
    # ======================================================
    def forward(self, x, y, c, lengths):
        """
        x: input tokens
        y: target tokens
        c: condition
        lengths: sequence lengths
        """

        z, mu, logvar = self.encode(x, c, lengths)
        logits = self.decode(x, z, c, lengths)

        return logits, mu, logvar

    # ======================================================
    #  Sampling (autoregressive)
    # ======================================================
    def sample(self, z, c, start_token, max_len):
        """
        z: latent vector [B, latent]
        c: property vector [B, prop]
        """
        batch_size = z.size(0)

        preds = []
        x_t = torch.tensor([[start_token]] * batch_size).to(z.device)

        hidden = None

        for _ in range(max_len):
            emb = self.embedding(x_t)

            z_rep = z.unsqueeze(1)
            c_rep = c.unsqueeze(1)

            dec_in = torch.cat([emb, z_rep, c_rep], dim=-1)

            out, hidden = self.decoder(dec_in, hidden)

            logits = self.output_proj(out.squeeze(1))
            x_t = torch.argmax(logits, dim=-1, keepdim=True)

            preds.append(x_t)

        return torch.cat(preds, dim=1)  # [B, max_len]


