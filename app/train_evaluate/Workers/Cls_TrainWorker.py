
import torch
import torch.nn as nn

from models.BBone_Module import Master_Model
from models.modules.LossNet import LossPredLoss
from utils import accuracy_score, log_assert
from config import al_params

from torch.utils.data import DataLoader, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from typing import Tuple, Dict, Any, List

import logging
logger = logging.getLogger(__name__)


class Cls_TrainWorker():
    def __init__(self, gpu_id: int, params: Dict[str, Any], world_size: int = 1) -> None:
        
        self.device = torch.device(gpu_id)
        self.iter: int = params["iter"]
        
        self.world_size: int = world_size

        self.model: Master_Model | DDP = params["ct_p"]["Master_Model"]
        
        self.dataset_name: str = params["ct_p"]["dataset_name"]
        self.strategy_name: str = params["strategy_name"]
        
        self.is_gtg_module = self.model.only_module_name == 'GTGModule'
        self.method_name: str = self.strategy_name.split(f'{self.model.name}_')[1]
        if self.is_gtg_module: 
            self.net_type = self.model.added_module.GTG_Model # type: ignore
            logger.info(f'GTGNet Type: {self.net_type}')
        else: self.net_type = 'LL'
        
        self.epochs = params["t_p"]["epochs"]
        self.batch_size = params["batch_size"]
        self.ds_t_p = params["t_p"][self.dataset_name]
        self.bbone_pt = params["ct_p"]["bbone_init_weights"]
        #if 'perc_labelled_batch' in params: self.perc_labelled_batch = params['perc_labelled_batch']
        if 'batch_size_gtg_online' in params: self.batch_size_gtg_online = params['batch_size_gtg_online']
        
        self.train_dl: DataLoader | Tuple[Subset, Subset] = params["train_dl"]
        self.test_dl: DataLoader = params["test_dl"]
        
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='none').to(self.device)
        self.mse_loss_fn = nn.MSELoss(reduction='none').to(self.device)
        self.l1_loss_fn = nn.L1Loss(reduction='none').to(self.device)
        self.kld_loss_fn = nn.KLDivLoss(reduction='batchmean').to(self.device)
        self.ll_loss_fn = LossPredLoss(self.device).to(self.device)
        self.bce_loss_fn = nn.BCELoss(reduction='none').to(self.device)
        
        self.score_fn = accuracy_score
        self.init_check_filename = f'app/checkpoints/{self.dataset_name}/{self.model.module.name if self.world_size > 1 else self.model.name}_init.pth.tar' # type: ignore
        self.check_best_path = f'app/checkpoints/{self.dataset_name}/{self.strategy_name}_{self.device}.pth.tar'
        
        # RETRAIN FROM SCRATCH THE NETWORK (different from what LL4AL have done)
        self.__load_checkpoint(self.init_check_filename)
        
        # BACKBONE PRETRAINING
        if self.bbone_pt != None and self.is_gtg_module:
            self.model.backbone.module.load_state_dict(self.bbone_pt) if self.world_size > 1 else self.model.backbone.load_state_dict(self.bbone_pt) # type: ignore
            
        #########################################################################################################################################################
        if isinstance(self.train_dl, tuple):
            
            lab_subset, unlab_subset = self.train_dl
            
            self.unlab_train_dl = DataLoader(dataset=unlab_subset, batch_size=self.batch_size_gtg_online * al_params["al_iters"], shuffle=True, pin_memory=True)            
            self.lab_train_dl = DataLoader(dataset=lab_subset, batch_size=self.batch_size_gtg_online * self.iter, shuffle=True, pin_memory=True)

            self.n_batches = len(self.lab_train_dl)
            
        else: self.n_batches = len(self.train_dl)
        #########################################################################################################################################################
        
        self.init_opt_sched()
        self.i = 0
        

    def init_opt_sched(self):
        optimizers = self.ds_t_p["optimizers"]
        module_name = self.model.only_module_name
        self.decay = optimizers["modules"]["decay"][module_name] if module_name != None and self.method_name != 'TiDAL' else None
        
        dict_optim_bb = {**optimizers["backbone"]["optim_p"][module_name]}
        
        self.optimizers: List[torch.optim.SGD | torch.optim.Adam] = []
        self.lr_schedulers = []
        self.optimizers.append(optimizers["backbone"]["type"][module_name](self.model.backbone.parameters(), **dict_optim_bb)) # type: ignore
        self.lr_schedulers.append(torch.optim.lr_scheduler.MultiStepLR(self.optimizers[0], milestones=[optimizers["milestones"]["backbone"]], gamma=0.1))
        if module_name != None:
            self.optimizers.append(optimizers["modules"]["type"][module_name](self.model.added_module.parameters(), **optimizers["modules"]["optim_p"][module_name])) # type: ignore
            
            #if self.method_name == 'TiDAL' or self.net_type == 'llmlp_gtg': 
            # TiDAL has milestone 160 and runs until epoch 200, all GTG Module have milestone to 60 and run until epoch 120
            logger.info(f'module_name {module_name}')
            self.lr_schedulers.append(torch.optim.lr_scheduler.MultiStepLR(self.optimizers[-1], milestones=[optimizers["milestones"][module_name]], gamma=0.1))
            
            
    
    def __save_checkpoint(self, filename: str) -> None:
        logger.info(f' => Saving {filename} Checkpoint')
        checkpoint = dict(state_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()) # type: ignore
        torch.save(checkpoint, filename)
    
    
    def __load_checkpoint(self, filename: str) -> None:
        logger.info(f' => Loading {filename} Checkpoint')
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.module.load_state_dict(checkpoint["state_dict"]) if self.world_size > 1 else self.model.load_state_dict(checkpoint["state_dict"]) # type: ignore
        

    def compute_losses(self, weight: float, module_out: torch.Tensor | None, outputs: torch.Tensor, labels: torch.Tensor, epoch: int,
                       tidal: None | Tuple[torch.Tensor, torch.Tensor, int] = None,) -> Tuple[float, torch.Tensor, float, float]:
                
        ce_loss = self.ce_loss_fn(outputs, labels)
        backbone = torch.mean(ce_loss)
        self.i += 1
            
        if module_out == None:
            return self.score_fn(outputs, labels), backbone, backbone.item(), 0
        
        elif self.is_gtg_module:
                        
            (pred_entr, true_entr), labelled_mask = module_out
            if self.i % 20 == 0: logger.info(f'y_pred {pred_entr}\ny_true {true_entr}') 
            
            if self.net_type != 'lstmbc': entr_loss = weight * self.mse_loss_fn(pred_entr, true_entr.detach())
            else: entr_loss = weight * self.bce_loss_fn(pred_entr, true_entr.detach())

            lab_ce_loss = torch.mean(ce_loss[labelled_mask])
            
            ############################################################################################################################
            # NO LOSS SCALING SINCE I HAVE REMOVED THE OVERSAMPLING STEP
            #lab_ce_loss /= self.rateo_unlab_lab_batch
            # ce loss scaled by the ratio between unlabelled and labelled batches
            # es: ma1000 labelled samples, 10000 unlabelled samples 128 batch_size -> scaling factor = (10000/128) / (1000/128) = 9.875
            ############################################################################################################################
            
            entr_loss = torch.mean(entr_loss[labelled_mask]) + torch.mean(entr_loss[~labelled_mask])
            
            loss = lab_ce_loss + entr_loss
            
            return self.score_fn(outputs[labelled_mask], labels[labelled_mask]), loss, lab_ce_loss.item(), entr_loss.item()
            
        
        elif self.method_name in ['LearningLoss', 'TA_VAAL'] or 'GTG_off' in self.method_name:
            loss_weird = weight * self.ll_loss_fn(module_out, ce_loss)
            loss = backbone + loss_weird
                
            return self.score_fn(outputs, labels), loss, backbone.item(), loss_weird.item()
        
        elif self.method_name == 'TiDAL':
            log_assert(tidal != None, 'TiDAL parameters are None')
            idxs, moving_prob, epoch = tidal # type: ignore
            moving_prob = moving_prob.to(self.device)
            moving_prob = (moving_prob * epoch + torch.softmax(outputs, dim=1) * 1) / (epoch + 1)
            
            self.train_dl.dataset.moving_prob[idxs.tolist(), :] = moving_prob.detach().cpu() # type: ignore
                        
            m_module_loss = weight * self.kld_loss_fn(F.log_softmax(module_out, 1), moving_prob.detach())
            loss = backbone + m_module_loss
                
            return self.score_fn(outputs, labels), loss, backbone.item(), m_module_loss.item()

        else: raise AttributeError('Invalid method_name')
    
    
    def return_moved_imgs_labs(self, images: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.world_size > 1:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
        else: images, labels = images.to(self.device), labels.to(self.device)
        return images, labels
    
    
    def train_batch(self, images, labels,  weight, idxs, moving_prob, epoch, train_loss_ce, train_loss_pred) -> Tuple[float, float, float, float]:
        images, labels = self.return_moved_imgs_labs(images, labels)
                
        for optimizer in self.optimizers: optimizer.zero_grad(set_to_none=True)
                    
        outputs, module_out = self.model(images, weight=weight, labels=labels) if not self.is_gtg_module \
            else self.model(images, weight=weight, labels=labels, iteration=self.iter)
                                                                        
        score, loss, train_loss_ce, train_loss_pred = self.compute_losses(
            weight=weight, module_out=module_out, outputs=outputs, labels=labels,
            epoch=epoch, tidal=(idxs, moving_prob, epoch),
        )
                
        loss.backward()
        for optimizer in self.optimizers: optimizer.step()

        return score, loss.item(), train_loss_ce, train_loss_pred
    
    
    
    def train(self) -> torch.Tensor:
        
        weight = 1.
        results = torch.zeros((4, self.epochs), device=self.device)
                
        self.model.train()
                
        for epoch in range(self.epochs):
                        
            train_loss, train_loss_ce, train_loss_pred, train_accuracy = .0, .0, .0, .0
            
            if self.world_size > 1: self.train_dl.sampler.set_epoch(epoch) # type: ignore
            if self.decay != None and epoch >= self.decay: 
                if self.net_type != 'llmlp_gtg' or self.method_name != 'TiDAL': weight = 0. # try llmlp_gtg to be performed for 200 epochs
            
            if isinstance(self.train_dl, tuple):
                for b_idx, ((idxs_l, images_l, labels_l, _), (idxs_u, images_u, labels_u, _)) in enumerate(zip(self.lab_train_dl, self.unlab_train_dl)):
                    
                    
                    idxs = torch.cat((idxs_l, idxs_u), dim=0)
                    images = torch.cat((images_l, images_u), dim=0)
                    labels = torch.cat((labels_l, labels_u), dim=0)
                    
                    train_a, train_l, train_l_ce, train_l_pred = self.train_batch(images, labels, weight, idxs, None, epoch, train_loss_ce, train_loss_pred)
                    
                    logger.info(f'{b_idx} | accuracy -> {train_a}\tloss -> {train_l}\ttrain_loss_ce -> {train_l_ce}\ttrain_loss_pred -> {train_l_pred}')
                    
                    train_loss += train_l
                    train_accuracy += train_a
                    train_loss_ce += train_l_ce
                    train_loss_pred += train_l_pred

            else:
                for idxs, images, labels, moving_prob in self.train_dl:
                    
                    train_a, train_l, train_l_ce, train_l_pred = self.train_batch(images, labels, weight, idxs, moving_prob, epoch, train_loss_ce, train_loss_pred)
                    
                    train_accuracy += train_a
                    train_loss += train_l
                    train_loss_ce += train_l_ce
                    train_loss_pred += train_l_pred
                    

            train_accuracy /= self.n_batches
            train_loss /= self.n_batches
            train_loss_ce /= self.n_batches
            train_loss_pred /= self.n_batches
            
            logger.info(f' Epoch: {epoch} | train_accuracy -> {train_accuracy}\ttrain_loss -> {train_loss}\ttrain_loss_ce -> {train_loss_ce}\ttrain_pred -> {train_loss_pred}')
            
            for pos, metric in zip(range(results.shape[0]), [train_accuracy, train_loss, train_loss_ce, train_loss_pred]):
                results[pos][epoch] = metric

            # MultiStepLR
            for lr_scheduler in self.lr_schedulers: lr_scheduler.step()
            
        # Save checkpoint
        self.__save_checkpoint(self.check_best_path)

        
        return results



    def test(self) -> torch.Tensor:
        
        self.__load_checkpoint(self.check_best_path)
        test_accuracy = .0
        
        self.model.backbone.eval() # type: ignore

        with torch.inference_mode():
            for _, images, labels in self.test_dl:
                
                images, labels = self.return_moved_imgs_labs(images, labels)
                outputs, _ = self.model.backbone(images) # type: ignore
                test_accuracy += self.score_fn(outputs, labels)

            test_accuracy /= len(self.test_dl)
            
        return torch.tensor([test_accuracy], device=self.device)
