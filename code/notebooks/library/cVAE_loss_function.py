import torch.nn as nn
import torch

def loss_function(model, logits, targets, batch_size, beta=1):
    recon_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    loss_recon = recon_loss_fn(logits, targets)
    
    kl_loss = -0.5 * torch.sum(1 + model.z_logvar - model.z_mean.pow(2) - model.z_logvar.exp()) / batch_size
    loss = loss_recon + beta * kl_loss

    return loss