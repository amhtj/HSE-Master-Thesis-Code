import numpy as np
from torch.nn import Module
import torch
import torch.nn as nn
import torch.nn.functional as F
from setuptools import setup, Extension
from torch.utils import cpp_extension
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data
import ELBO
import KL


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.device = config.device
        self.auxiliary_data_source = config.specific_parameters.auxiliary_data_source
        self.all_data_sources = ['reconstruction_features', self.auxiliary_data_source]
        self.dataset = config.dataset_name
        self.generalized = config.generalized
        self.classifier_batch_size = 32
        self.img_seen_samples = config.samples_per_class[0]          
        self.att_seen_samples = config.samples_per_class[1]
        self.att_unseen_samples = config.samples_per_class[2]
        self.img_unseen_samples = config.samples_per_class[3]
        self.reco_loss_function = config.specific_parameters.loss
        self.nepoch = config.nepoch
        self.lr_cls = config.specific_parameters.lr_cls
        self.cross_reconstruction = config.specific_parameters.warmup.cross_reconstruction
        self.cls_train_epochs = config.specific_parameters.cls_train_steps

        self.dataset = dataloader(self.dataset,
                                  copy.deepcopy(self.auxiliary_data_source),
                                  device=self.device)

        if self.dataset == 'noise_signal':
            self.num_classes = 20
            self.num_novel_classes = 10

        self.encoder = {}
        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.encoder[datatype] = EncoderTemplate(
                dim, self.latent_size, self.hidden_size_rule[datatype], self.device)
            print(str(datatype) + ' ' + str(dim))

        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.decoder[datatype] = DecoderTemplate(
                self.latent_size, dim, self.hidden_size_rule[datatype], self.device)

        # An optimizer       
        parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            parameters_to_optimize += list(self.encoder[datatype].parameters())
            parameters_to_optimize += list(self.decoder[datatype].parameters())

        self.optimizer = optim.SGD(parameters_to_optimize, lr=config.specific_parameters.lr_gen_model, betas=(
            0.9, 0.999), eps=1e-08, weight_decay=0, momentum=True)

        if self.reco_loss_function == 'l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=False)

        elif self.reco_loss_function == 'l1':
            self.reconstruction_criterion = nn.L1Loss(size_average=False)

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.FloatTensor(logvar.size()[0],1).normal_(0,1)
            eps  = eps.expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu

    def trainstep(self, img, att):

        mu_img, logvar_img = self.encoder['reconstruction_features'](img)
        z_from_img = self.reparameterize(mu_img, logvar_img)

        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)
        z_from_att = self.reparameterize(mu_att, logvar_att)

        img_from_img = self.decoder['denoise_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)

        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) \
                              + self.reconstruction_criterion(att_from_att, att)

        # Loss

        img_from_att = self.decoder['denoise_features'](z_from_att)
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)

        cross_reconstruction_loss = self.reconstruction_criterion(img_from_att, img) \
                                    + self.reconstruction_criterion(att_from_img, att)

       self.elbo = ELBO(self.encoder, **elbo_kwargs)
       self.kld = KL(self.encoder, **kld_kwargs)


        class ELBO(Module):
            def __init__(self, variational_model, lr=0.01, optimizer='SGD', num_iterations=1):
                super().__init__()
                self.variational_model = variational_model
                self.optimizer = getattr(optim, optimizer)(variational_model.parameters(), lr=lr)
                self.num_iterations = num_iterations

    def forward(self, p_model, q_model, batch_size=64):
        assert q_model.has_latents, "Error: Q Model does not have latent variables"

        p_samples = p_model.sample(batch_size)
        # Sample from q(z | x) and compute_logprob
        loss = 0
        for i in range(self.num_iterations):
            latents, variational_log_prob = self.variational_model.sample(p_samples, compute_logprob=True)
            # Compute log p(x, z) = log p(x | z) + log p(z)
            q_log_prob = q_model.log_prob(p_samples, latents)
            loss += -(q_log_prob - variational_log_prob).mean()

        return loss / self.num_iterations

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
