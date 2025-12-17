from library.GCN import ConvolutionLayer, PoolingLayer
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random
import torch

class cVAE(nn.Module):
    def __init__(self, encoder, decoder, device,
                 n_gcn_hidden_dim, n_gru_hidden_dim,
                 latent_dim, vocab_size, embedding_dim,
                 teacher_forcing_ratio=0.5):
        super(cVAE, self).__init__()

        self.device = device

        self.encoder = encoder
        self.decoder = decoder

        self.gcn_hidden_dim = n_gcn_hidden_dim # size of the final GCN hidden layer
        self.gru_hidden_dim = n_gru_hidden_dim # size of the GRU hidden layer

        self.vocab_size = vocab_size # number of characters that can be found in a SMILES string
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim

        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Latent space projections
        self.encoder_mu = nn.Linear(self.gcn_hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(self.gcn_hidden_dim, latent_dim)

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
        x: [batch, seq_len] token indices for the characters of the SMILES string
        y: [batch, 1] property vector
        """
        batch_size, target_len = x.size()
        device = x.device

        outputs = torch.zeros(batch_size, target_len, self.vocab_size).to(device)

        # Initialize first token as <STR> (index 2)
        input_token = torch.ones(batch_size, dtype=torch.long).to(device) * 2
        outputs[:,0,2] = 1

        # Initialize hidden state (zeros)
        hidden = torch.zeros(self.decoder.n_gru_layers, batch_size, self.decoder.hidden_size).to(device)

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

    def forward(self, node_mat, adj_mat, smiles, gap):
        """
        Full forward pass
        x: [batch, seq_len] token indices
        y: [batch, 1] property vector
        """
        # Encode graph to hidden representation
        hidden_encoder = self.encoder(node_mat, adj_mat, gap)  # should output [batch, gcn_hidden_dim + condition_dim]

        # Sample latent vector
        z = self._sample_latent(hidden_encoder)

        # Decode sequence
        recon_x = self.forward_decoder(z, smiles, gap)

        return recon_x
    
    @torch.no_grad()
    def sample(self, z, y, seq_length=100, temperature=0.8):
        """
        Generate a SMILES sequence from a latent vector z and property y.
        z: [batch, latent_dim]
        y: [batch, 1]
        temperature: 
            - low T (T<1): conservative, high-probability tokens
            - T=1: Normal behaviour
            - high T (T>1): More random, diverse, riskier
        """

        str_idx = 2
        end_idx = 1
        pad_idx = 0

        batch_size = z.size(0)
        device = z.device

        # Start with <STR> token = index 2
        input_token = torch.full((batch_size,), str_idx, dtype=torch.long, device=device)

        # Initialize hidden state to zeros
        hidden = torch.zeros(self.decoder.n_gru_layers,
                            batch_size,
                            self.decoder.hidden_size,
                            device=device)

        outputs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(seq_length):
            # One decoding step
            logits, hidden = self.decoder(input_token, z, y, hidden)

            # Mask illegal tokens
            logits[:, pad_idx] = -1e9
            logits[:, str_idx] = -1e9

            # Convert to probabilities
            probs = F.softmax(logits / temperature, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            outputs.append(next_token)

            # Feed token back to decoder
            input_token = next_token

            finished |= (next_token == end_idx)
            if finished.all():
                break

        # Concatenate tokens (batch, seq_length)
        outputs = torch.stack(outputs, dim=1)
        return outputs

class GCN_Encoder(nn.Module):

    """
    Class which combines the convolution, pooling and neural network layers to create a general GCN model. This model also first applies a transformation from a node matrix N to a node features matrix F, through the use of a weight matrix W. This weight matrix is NOT node specific, it is a single matrix used to transform all node vectors (F = N @ W).
    """
    def __init__(
        self,
        node_vec_len: int,
        node_fea_len: int,
        hidden_fea_len: int,
        n_conv: int,
        n_hidden: int,
        n_outputs: int,
        p_dropout: float = 0.0,
    ):
        # Call constructor of base class
        super().__init__()

        # Define layers
        # Initial transformation from node matrix to node features
        self.init_transform = nn.Linear(node_vec_len, node_fea_len)

        # Convolution layers
        self.conv_layers = nn.ModuleList(
            [
                ConvolutionLayer(
                    node_in_len=node_fea_len,
                    node_out_len=node_fea_len,
                )
                for i in range(n_conv)
            ]
        )

        # Pool convolution outputs
        self.pooling = PoolingLayer()
        pooled_node_fea_len = node_fea_len

        # Pooling activation
        self.pooling_activation = nn.LeakyReLU()

        # From pooled vector to hidden layers
        self.pooled_to_hidden = nn.Linear(pooled_node_fea_len, hidden_fea_len)

        # Hidden layer
        self.hidden_layer = nn.Linear(hidden_fea_len, hidden_fea_len)

        # Hidden layer activation function
        self.hidden_activation = nn.LeakyReLU()

        # Hidden layer dropout
        self.dropout = nn.Dropout(p=p_dropout)

        # If hidden layers more than 1, add more hidden layers
        self.n_hidden = n_hidden
        if self.n_hidden > 1:
            self.hidden_layers = nn.ModuleList(
                [self.hidden_layer for _ in range(n_hidden - 1)]
            )
            self.hidden_activation_layers = nn.ModuleList(
                [self.hidden_activation for _ in range(n_hidden - 1)]
            )
            self.hidden_dropout_layers = nn.ModuleList(
                [self.dropout for _ in range(n_hidden - 1)]
            )

        # Final layer going to the output
        self.hidden_to_output = nn.Linear(hidden_fea_len, n_outputs)

    def forward(self, node_mat, adj_mat, y):
        # Perform initial transform on node_mat
        node_fea = self.init_transform(node_mat)

        # Perform convolutions
        for conv in self.conv_layers:
            node_fea = conv(node_fea, adj_mat)

        # Perform pooling
        pooled_node_fea = self.pooling(node_fea)
        pooled_node_fea = self.pooling_activation(pooled_node_fea)

        # First hidden layer
        hidden_node_fea = self.pooled_to_hidden(pooled_node_fea)
        hidden_node_fea = self.hidden_activation(hidden_node_fea)
        hidden_node_fea = self.dropout(hidden_node_fea)

        # Subsequent hidden layers
        if self.n_hidden > 1:
            for i in range(self.n_hidden - 1):
                hidden_node_fea = self.hidden_layers[i](hidden_node_fea)
                hidden_node_fea = self.hidden_activation_layers[i](hidden_node_fea)
                hidden_node_fea = self.hidden_dropout_layers[i](hidden_node_fea)

        # Output is last hidden node + property as it will be used in the cVAE to extract the latent representation
        out = torch.cat([hidden_node_fea, y], dim=1) # ( batch_size , hidden_dim + condition_dim)

        return out
    

class GRU_Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        latent_dim,
        hidden_size,
        n_gru_layers,
        n_fc_layers
        ):

        super(GRU_Decoder, self).__init__()
        self.vocab_size = vocab_size # number of characters that can be found in a SMILES string
        self.latent_dim = latent_dim # size of the latent space
        self.property_dim = 1 # number of properties that serve as conditions (shape of condition vector)

        self.hidden_size = hidden_size
        self.n_gru_layers = n_gru_layers
        self.n_fc_layers = n_fc_layers
        self.embedding_dim = embedding_dim

        # Embedding layer for input tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # GRU input size: token embedding + latent vector + property
        self.gru = nn.GRU(embedding_dim + latent_dim + self.property_dim, hidden_size, n_gru_layers, batch_first=True)

        fc_layers = []
        for _ in range(n_fc_layers):
            fc_layers.append(nn.Linear(hidden_size, hidden_size))
            fc_layers.append(nn.ReLU())

        self.fc_stack = nn.Sequential(*fc_layers)

        # Output layer: project GRU hidden state to vocab size
        self.fc_out = nn.Linear(hidden_size, vocab_size)

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
        decoder_input = torch.cat([token_embed, z, y], dim=1).unsqueeze(1)  # [batch, 1, embed+latent+property]

        # GRU forward
        gru_out, hidden = self.gru(decoder_input, hidden)  # output: [batch, 1, hidden_size]

        h = gru_out.squeeze(1)

        h = self.fc_stack(h)

        # Project to vocab
        output = self.fc_out(h.squeeze(1))  # [batch, vocab_size]

        return output, hidden