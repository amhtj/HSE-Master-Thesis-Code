import torch
from torch.utils import data

class Main:
    def __init__(self):
        self.data = {}
        
    def train_vae(self):
        losses = []

        self.dataloader = data.DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)  # ,num_workers = 4)

        self.dataset.novelclasses = self.dataset.novelclasses.long()
        self.dataset.seenclasses = self.dataset.seenclasses.long()
        
        # Leave both statements
        
        self.train()
        self.reparameterize_with_noise = True

        print('Train for reconstruction')
        for epoch in range(0, self.nepoch):
            self.current_epoch = epoch

            i = -1
            for iters in range(0, self.dataset.ntrain, self.batch_size):
                i += 1

                label, data_from_modalities = self.dataset.next_batch(
                    self.batch_size)

                label = label.long().to(self.device)
                for j in range(len(data_from_modalities)):
                    data_from_modalities[j] = data_from_modalities[j].to(
                        self.device)
                    data_from_modalities[j].requires_grad = False

                loss = self.trainstep(
                    data_from_modalities[0], data_from_modalities[1])

                if i % 50 == 0:

                    print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t' +
                          ' | loss ' + str(loss)[:5])

                if i % 50 == 0 and i > 0:
                    losses.append(loss)

                for key, value in self.encoder.items():
                    self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()

        return losses

    def predict(self):

        iter_idx = 0
        embeddings = torch.Tensor()
        for batch in self.dataset.gen_next_batch(self.batch_size, dset_part='test'):
            iter_idx += 1
            label, data_from_modalities = batch

            label = label.long().to(self.device)
            for j in range(len(data_from_modalities)):
                data_from_modalities[j] = data_from_modalities[j].to(
                    self.device)
            
            mu_att, logvar_att = self.encoder[self.auxiliary_data_source](data_from_modalities[1])
            z_from_att = self.reparameterize(mu_att, logvar_att)

            embeddings = torch.cat((embeddings, z_from_att), 0)
        
        return embeddings
    
    