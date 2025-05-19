
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from auto_encoder.BackBone_Decoder import BackBone_Decoder

from typing import List, Tuple

import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


import logging
logger = logging.getLogger(__name__)



def make_random_split(dataset: Dataset, perc_val: float, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    ds_len = len(dataset) # type: ignore
    val_ds_len = int(ds_len * perc_val)
    
    train_ds_len = ds_len - val_ds_len
    
    train_ds, va_ds = random_split(dataset, (train_ds_len, val_ds_len))
    train_dl = DataLoader(train_ds, shuffle=True, pin_memory=True, batch_size=batch_size)
    val_dl = DataLoader(va_ds, shuffle=False, pin_memory=True, batch_size=batch_size)
    
    return train_dl, val_dl
    

def fit_ae(model: BackBone_Decoder, device: torch.device, train_ds: Dataset, perc_val=0.2, num_epochs=200, bs=128, lr=0.001, momentum=0.5):

    assert 0 < lr < 1 and num_epochs > 0 and bs > 0 and 0 <= momentum < 1
    
    model = model.to(device)

    # set optimizer, loss type and datasets (depending on the type of AE)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    criterion = nn.MSELoss()
    
    train_dl, val_dl = make_random_split(train_ds, perc_val, bs)
    
    # training cycle
    loss = None  # just to avoid reference before assigment
    history = {'tr_loss': [], 'val_loss': []}
    for epoch in range(num_epochs):
        # training
        model.train()
        tr_loss = 0

        for _, images, _, _ in train_dl:
            # zero the gradient
            optimizer.zero_grad()
                    
            images = images.to(device)
            
            _, outputs = model(images)
            # compute loss (flatten output in case of ConvAE. targets already flat)
            
            loss = criterion(torch.flatten(outputs, 1), torch.flatten(images, 1))
            tr_loss += loss.item()
            # propagate back the loss
            loss.backward()
            optimizer.step()
        
        tr_loss /= len(train_dl)
        history['tr_loss'].append(round(tr_loss, 5))

        # validation
        val_loss = evaluate(model=model, val_dl=val_dl, criterion=criterion, device=device)
        history['val_loss'].append(round(val_loss, 5))
        torch.cuda.empty_cache()

        logger.info(f'AE Epoch {epoch} | tr_loss: {round(tr_loss, 5)} | val_loss: {round(val_loss, 5)}')

        # simple early stopping mechanism
        '''if epoch >= 10:
            last_values = history['val_loss'][-10:]
            if (abs(last_values[-10] - last_values[-1]) <= 2e-5) or (last_values[-3] < last_values[-2] < last_values[-1]):
                return''' 



def evaluate(model: BackBone_Decoder, criterion: nn.MSELoss, device: torch.device, val_dl: DataLoader) -> float:

    model.eval()

    with torch.no_grad():
        val_loss = 0
        for _, images, _, _ in val_dl:
            images = images.to(device)
            
            _, outputs = model(images)
            
            loss = criterion(torch.flatten(outputs, 1), torch.flatten(images, 1))
            val_loss += loss.item()
            
    return val_loss / len(val_dl)



def get_initial_sample_higher_MSE(model: BackBone_Decoder, dl: DataLoader, device: torch.device, n_lab_obs: int) -> List[int]:
    
    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss()

    idxs_to_sort = torch.empty((0,), dtype=torch.float32, device=torch.device('cpu'))
    losses = torch.empty((0,), dtype=torch.float32, device=torch.device('cpu'))

    with torch.no_grad():
        for idxs, images, _, _ in dl:
            idxs_to_sort = torch.cat((idxs_to_sort, idxs.cpu()), dim=0)
            
            images = images.to(device)
            _, outputs = model(images)
            
            loss = criterion(torch.flatten(outputs, 1), torch.flatten(images, 1))
            losses = torch.cat((losses, loss.cpu), dim=0)
            
    harder_sample = torch.topk(losses, k=n_lab_obs, largest=True, sorted=False).indices
    
    return [int(idxs_to_sort[id].item()) for id in harder_sample]



def get_initial_sample_farthest_KMeans(model: BackBone_Decoder, dl: DataLoader, device:torch.device, n_classes: int, n_obs_per_class: int) -> List[int]:
     
    model = model.to(device)
    model.eval()
    
    unlab_embedding = torch.empty((0, model.encoder.get_embedding_dim()), dtype=torch.float32, device=torch.device('cpu'))

    with torch.no_grad():
        for _, images, _, _ in dl:
            images = images.to(device)
            embedds, _ = model(images)
            unlab_embedding = torch.cat((unlab_embedding, embedds.cpu()), dim=0)
            
    unlab_embedding = unlab_embedding.numpy()
    
    kmeans = KMeans(n_clusters=n_classes)
    kmeans.fit(unlab_embedding)
    
    # Find the farthest points to each centroid (cluster)
    initial_samples = []
    for i in range(n_classes):
        cluster_samples = np.where(kmeans.labels_ == i)[0]
        centroid = kmeans.cluster_centers_[i]
        distances = pairwise_distances_argmin_min(np.array(unlab_embedding[cluster_samples]), np.array([centroid]))[1]
        farthest_samples = cluster_samples[np.argsort(distances)[-n_obs_per_class:]]
        initial_samples.extend(farthest_samples.tolist())
    
    return initial_samples
    
    