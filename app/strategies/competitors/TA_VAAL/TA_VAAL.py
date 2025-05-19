
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import numpy as np

from ActiveLearner import ActiveLearner
from strategies.competitors.TA_VAAL.ta_vaal_query_model import Discriminator, VAE

from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)


class TA_VAAL(ActiveLearner):
    
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any]) -> None:
        super().__init__(ct_p, t_p, self.__class__.__name__)
        
        
    def read_data(self, dataloader: DataLoader, labels=True):
        if labels:
            while True:
                for _, img, label in dataloader: yield img, label
        else:
            while True:
                for _, img, _ in dataloader: yield img

        
    def vae_loss(self, x, recon, mu, logvar, beta):
        mse_loss = nn.MSELoss()
        MSE = mse_loss(recon, x)
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
    
    
    def train_vaal(self, optimizers: Dict[str, optim.Adam], cycle: int, n_top_k_obs: int, subset_len: int) -> None:
        
        EPOCHV = 120
        
        self.load_best_checkpoint()
        self.model.eval()

        self.vae.train()
        self.discriminator.train()

        adversary_param, beta, num_adv_steps, num_vae_steps = 1, 1, 1, 1

        bce_loss = nn.BCELoss()
        
        labelled_data = self.read_data(self.lab_train_dl)
        unlabelled_data = self.read_data(self.unlab_train_dl)

        #self.train_vaal(optimizers, self.iter+1, n_top_k_obs, len(sample_unlab_subset))
        train_iterations = int( (n_top_k_obs * cycle + subset_len) * EPOCHV / self.batch_size )


        for iter_count in range(train_iterations):
            labelled_imgs, labels = next(labelled_data)
            unlabelled_imgs = next(unlabelled_data)[0]

            labelled_imgs = labelled_imgs.to(self.device)
            unlabelled_imgs = unlabelled_imgs.to(self.device)
            labels = labels.to(self.device)
            
            if iter_count == 0 :
                r_l_0 = torch.from_numpy(np.random.uniform(0, 1, size=(labelled_imgs.shape[0],1))).type('torch.FloatTensor').to(self.device)
                r_u_0 = torch.from_numpy(np.random.uniform(0, 1, size=(unlabelled_imgs.shape[0],1))).type('torch.FloatTensor').to(self.device)

                r_l = r_l_0.detach()
                r_u = r_u_0.detach()
                r_l_s = r_l_0.detach()
                r_u_s = r_u_0.detach()
            else:
                with torch.no_grad():
                    r_l = self.model(labelled_imgs, mode = 'module_out')
                    r_u = self.model(unlabelled_imgs, mode = 'module_out')
                    
                r_l_s = torch.sigmoid(r_l).detach()
                r_u_s = torch.sigmoid(r_u).detach()
                    
                           
            # VAE step
            for count in range(num_vae_steps): # num_vae_steps
                recon, _, mu, logvar = self.vae(r_l_s, labelled_imgs)
                unsup_loss = self.vae_loss(labelled_imgs, recon, mu, logvar, beta)
                unlab_recon, _, unlab_mu, unlab_logvar = self.vae(r_u_s, unlabelled_imgs)
                transductive_loss = self.vae_loss(unlabelled_imgs, unlab_recon, unlab_mu, unlab_logvar, beta)
            
                labelled_preds = self.discriminator(r_l, mu)
                unlabelled_preds = self.discriminator(r_u, unlab_mu)
                
                lab_real_preds = torch.ones(labelled_imgs.size(0))
                unlab_real_preds = torch.ones(unlabelled_imgs.size(0))
                    
                lab_real_preds = lab_real_preds.to(self.device)
                unlab_real_preds = unlab_real_preds.to(self.device)

                dsc_loss = bce_loss(labelled_preds[:, 0], lab_real_preds) + \
                           bce_loss(unlabelled_preds[:, 0], unlab_real_preds)
                           
                total_vae_loss = unsup_loss + transductive_loss + adversary_param * dsc_loss
                
                optimizers["vae"].zero_grad()
                total_vae_loss.backward()
                optimizers["vae"].step()

                # sample new batch if needed to train the adversarial network
                if count < (num_vae_steps - 1):
                    labelled_imgs, _ = next(labelled_data)
                    unlabelled_imgs = next(unlabelled_data)[0]

                    labelled_imgs = labelled_imgs.to(self.device)
                    unlabelled_imgs = unlabelled_imgs.to(self.device)
                    labels = labels.to(self.device)


            # Discriminator step
            for count in range(num_adv_steps):
                with torch.no_grad():
                    _, _, mu, _ = self.vae(r_l_s, labelled_imgs)
                    _, _, unlab_mu, _ = self.vae(r_u_s, unlabelled_imgs)
                
                labelled_preds = self.discriminator(r_l, mu)
                unlabelled_preds = self.discriminator(r_u, unlab_mu)
                
                lab_real_preds = torch.ones(labelled_imgs.size(0))
                unlab_fake_preds = torch.zeros(unlabelled_imgs.size(0))

                lab_real_preds = lab_real_preds.to(self.device)
                unlab_fake_preds = unlab_fake_preds.to(self.device)
                
                dsc_loss = bce_loss(labelled_preds[:, 0], lab_real_preds) + \
                           bce_loss(unlabelled_preds[:, 0], unlab_fake_preds)

                optimizers["discriminator"].zero_grad()
                dsc_loss.backward()
                optimizers["discriminator"].step()

                # sample new batch if needed to train the adversarial network
                if count < (num_adv_steps-1):
                    labelled_imgs, _ = next(labelled_data)
                    unlabelled_imgs = next(unlabelled_data)[0]

                    labelled_imgs = labelled_imgs.to(self.device)
                    unlabelled_imgs = unlabelled_imgs.to(self.device)
                    labels = labels.to(self.device)
                if iter_count % 100 == 0:
                    logger.info("Iteration: " + str(iter_count) + " / " + str(train_iterations) + \
                        " vae_loss: " + str(total_vae_loss.item()) + " dsc_loss: " + str(dsc_loss.item())) # type: ignore

    
    
    def query(self, sample_unlab_subset: Subset, n_top_k_obs: int) -> Tuple[List[int], List[int]]:
            
        # set the entire batch size to the dimension of the sampled unlabelled set
        dl_dict = dict( batch_size=self.batch_size, shuffle=False, pin_memory=True )
        
        self.unlab_train_dl = DataLoader(sample_unlab_subset, **dl_dict)
        self.lab_train_dl = DataLoader(Subset(self.dataset.unlab_train_ds, self.labelled_indices), **dl_dict)
        
        self.vae = VAE(
            z_dim=self.dataset.image_size,
            nc=self.dataset.n_channels,
            f_filt=3 if self.ct_p["dataset_name"] == 'fmnist' else 4
        ).to(self.device)
        
        self.discriminator = Discriminator(z_dim=self.dataset.image_size).to(self.device)
             
        optimizers = {
            'vae': optim.Adam(self.vae.parameters(), lr=5e-4), 
            'discriminator': optim.Adam(self.discriminator.parameters(), lr=5e-4)
        }

        self.train_vaal(optimizers, self.iter+1, n_top_k_obs, len(sample_unlab_subset))
        
        self.vae.eval() # <---- since we have finisched our training we can now starts the evaluation step
              
        all_preds, all_indices = [], []

        for indices, images, _ in self.unlab_train_dl:
            images = images.to(self.device)
            
            with torch.no_grad():
                r = self.model(images, mode = 'module_out')
                _, _, mu, _ = self.vae(torch.sigmoid(r), images)
                preds = self.discriminator(r, mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        
        # need to multiply by -1 to be able to use torch.topk 
        #all_preds *= -1 # -> technically now should be correct, commenting this I can use torch.topk(, largest=False)
        
        # select the topk points which the discriminator things are the most likely to be unlabelled
        overall_topk = torch.topk(all_preds, n_top_k_obs, largest=False).indices.tolist()
        
        return overall_topk, [self.rand_unlab_sample[id] for id in overall_topk]
        
    