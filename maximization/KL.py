import torch
from torch.nn import Module
from abc import abstractmethod, ABC
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data

class KL(Module):
    def __init__(self, variational_model, lr=0.01, optimizer='SGD', num_iterations=1):
        super().__init__()
        self.variational_model = variational_model
        self.optimizer = getattr(optim, optimizer)(variational_model.parameters(), lr=lr)
        self.num_iterations = num_iterations
        
    def KLD(elf, variational_model, lr=0.01, optimizer='Adam', num_iterations=1):
        KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
              + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))
              
# Distribution Alignment

distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
                      torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1)) 
        
    
        f1 = 1.0*(self.current_epoch - self.warmup.cross_reconstruction.start_epoch)/(1.0*( self.warmup.cross_reconstruction.end_epoch- self.warmup.cross_reconstruction.start_epoch))
        f1 = f1*(1.0*self.warmup.cross_reconstruction.factor)
        cross_reconstruction_factor = torch.FloatTensor([min(max(f1,0),self.warmup.cross_reconstruction.factor)])
        
        f2 = 1.0 * (self.current_epoch - self.warmup.beta.start_epoch) / ( 1.0 * (self.warmup.beta.end_epoch - self.warmup.beta.start_epoch))
        f2 = f2 * (1.0 * self.warmup.beta.factor)
        beta = torch.FloatTensor([min(max(f2, 0), self.warmup.beta.factor)])
        
        f3 = 1.0*(self.current_epoch - self.warmup.distance.start_epoch )/(1.0*( self.warmup.distance.end_epoch- self.warmup.distance.start_epoch))
        f3 = f3*(1.0*self.warmup.distance.factor)
        distance_factor = torch.FloatTensor([min(max(f3,0),self.warmup.distance.factor)])
        
        self.optimizer.zero_grad()
        
        loss = reconstruction_loss - beta * KLD
        
        if cross_reconstruction_loss > 0:
            loss += cross_reconstruction_factor*cross_reconstruction_loss
        if distance_factor > 0:
            loss += distance_factor*distance
        loss.backward()
        
        self.optimizer.step()
        
        return loss.item()