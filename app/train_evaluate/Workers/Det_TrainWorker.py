
import pickle
import torch

from torch.nn.parallel import DistributedDataParallel as DDP

from models.backbones.ssd_pytorch.detect_eval import do_python_eval, write_voc_results_file
from models.backbones.ssd_pytorch.ssd_layers.modules.multibox_loss import MultiBoxLoss
from models.modules.LossNet import LossPredLoss
from models.BBone_Module import Master_Model


from datasets_creation.Detection import Det_Dataset
from torch.utils.data import DataLoader

from typing import Dict, Any, List, Tuple
import os
import numpy as np

from utils import create_directory, cycle

import logging
logger = logging.getLogger(__name__)


class Det_TrainWorker():
    def __init__(self, gpu_id: int, params: Dict[str, Any], world_size: int = 1) -> None:
        
        self.device = torch.device(gpu_id)
        self.iter: int = params["iter"]
        self.ct_p, self.t_p = params["ct_p"], params["t_p"]
        self.epoch_size: int = self.t_p["epoch_size"]
        self.dataset: Det_Dataset = self.ct_p["Dataset"]
        
        self.world_size: int = world_size
        self.wandb_run = params["wandb_p"] if 'wandb_p' in params else None
        
        self.model: Master_Model | DDP = self.ct_p["Model_train"] if self.world_size > 1 else self.ct_p["Master_Model"]
        
        self.strategy_name: str = params["strategy_name"]
        
        self.train_dl: DataLoader = params["train_dl"]
        self.test_dl: DataLoader = params["test_dl"]
        
        self.backbone_loss_fn = MultiBoxLoss(self.dataset.n_classes, 0.5, True, 0, True, 3, 0.5, False, self.device)
        self.ll_loss_fn = LossPredLoss(self.device).to(self.device)


        self.best_check_filename = f'app/checkpoints/{self.ct_p["dataset_name"]}'
        self.init_check_filename = f'{self.best_check_filename}/{self.model.module.name if self.world_size > 1 else self.model.name}_init.pth.tar'
        # same name ActiveLearner.load_best_checkpoint reads
        self.check_best_path = f'{self.best_check_filename}/{self.strategy_name}_{self.device}.pth.tar'
        
        if self.iter > 1: 
            self.__load_checkpoint(self.check_best_path)
            logger.info(' => Continuing Training the Best Model from the Previous Iteration')
        else:
            self.__load_checkpoint(self.init_check_filename)
            logger.info(' => Loading Initial Checkpoint')
        logger.info(' DONE')
        
        # set device for priors
        self.backbone = self.model.module.backbone if self.world_size > 1 else self.model.backbone
        self.backbone.set_device_priors(self.device)
    
    
    def init_opt_sched(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[240], gamma=0.1)
    
    
    def __save_checkpoint(self, filename: str) -> None:
        checkpoint = dict(state_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict())
        torch.save(checkpoint, filename)
    

    def __load_checkpoint(self, filename: str) -> None:
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.module.load_state_dict(checkpoint["state_dict"]) if self.world_size > 1 else self.model.load_state_dict(checkpoint["state_dict"])
        self.init_opt_sched()
        
        
    def compute_losses(self, module_out: torch.Tensor, outputs: torch.Tensor, \
                       labels: List[torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
                
        loss_l, loss_c, N = self.backbone_loss_fn(outputs, labels)
        target_loss = loss_l + loss_c
        original_loss = (loss_l + loss_c).sum() / N
        loss_l = torch.mean(loss_l)
        loss_c = torch.mean(loss_c)
        
        if module_out is None:
            return original_loss, (loss_l, loss_c), torch.tensor(0.)
        else:
            pred_loss = self.ll_loss_fn(module_out, target_loss)
            loss = original_loss + pred_loss
                
            return loss, (loss_l, loss_c), pred_loss

    

    def train(self) -> torch.Tensor:
        
        if self.wandb_run != None: self.wandb_run.watch(self.model, log="all", log_freq=10)
                
        self.model.train()
        
        # loss counters
        loc_loss, conf_loss, train_loss, train_pred_loss = .0, .0, .0, .0
        
        epoch, step_index = 0, 0
                
        results = torch.zeros((4, self.t_p["epochs"]), device=self.device)
        
        if self.world_size > 1: self.train_dl.sampler.set_epoch(epoch) # type: ignore
        
        batch_iterator = iter(cycle(self.train_dl))
        
        for iteration in range(0, self.t_p["max_iter"]):

            if iteration in self.t_p["lr_steps"]:
                step_index += 1
                self.adjust_learning_rate(step_index)

            _, images, labels = next(batch_iterator)
            
            images = images.requires_grad_(True).to(self.device, non_blocking=True)
            labels = [ann.to(self.device, non_blocking=True) for ann in labels]
            
            self.optimizer.zero_grad()
            
            outputs, module_out = self.model(images, labels)
            
            loss, (loss_l, loss_c), module_loss = self.compute_losses(
                module_out=module_out, outputs=outputs, labels=labels
            )

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., norm_type=2)
            
            self.optimizer.step()

            train_loss += loss.item()
            train_pred_loss += module_loss.item()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            
            if self.wandb_run != None:
                self.wandb_run.log({
                    "train_loss": loss.item(),
                    "loc_loss": loss_l.mean().item(),
                    "conf_loss": loss_c.mean().item(),
                    "train_pred_loss": module_loss.item(),
                })
            
            # end of epoch: record the mean losses, checkpoint and reset the counters
            if (iteration + 1) % self.epoch_size == 0:
                
                self.__save_checkpoint(self.check_best_path)
                self.scheduler.step()
                    
                logger.info(f'GPU: {self.device} ||| Epoch {epoch} | Iteration {iteration + 1} -> train_loss: {train_loss / self.epoch_size}\tloc_loss: {loc_loss / self.epoch_size}\tconf_loss: {conf_loss / self.epoch_size}\ttrain_pred_loss: {train_pred_loss / self.epoch_size}')
                                
                for pos, metric in zip(range(results.shape[0]), [train_loss / self.epoch_size,
                                                                 loc_loss / self.epoch_size, 
                                                                 conf_loss / self.epoch_size, 
                                                                 train_pred_loss / self.epoch_size]): 
                    if metric != 0.0: results[pos][epoch] = metric
                
                loc_loss, conf_loss = .0, .0
                train_loss, train_pred_loss = .0, .0
                epoch += 1
                if self.world_size > 1: self.train_dl.sampler.set_epoch(epoch) # type: ignore
            
        self.__load_checkpoint(self.check_best_path)

        return results



    def test(self) -> torch.Tensor:
        
        self.backbone.change_phase() # -> test phase, the SSD outputs the detections
        
        self.model.eval()

        # evaluation
        num_images = len(self.test_dl.dataset) # type: ignore
        
        # all detections are collected into:
        all_boxes: List[List[np.ndarray | None]] = [[ None for _ in range(num_images)] for _ in range(self.dataset.n_classes)]
        output_dir = f'results/{self.ct_p["exp_path"]}/{self.ct_p["dataset_name"]}/{self.ct_p["trial"]}/{self.strategy_name}/test'
        create_directory(output_dir)        
        det_file = os.path.join(output_dir, f'detections_{self.device}.pkl')
        
        with torch.no_grad(): # Allow inference mode
            logger.info(f'GPU: {self.device} |||  => Detection Phase On Going...')
            for i in range(num_images):
                im, _, h, w = self.test_dl.dataset.pull_item(i) # type: ignore
                x = im.unsqueeze(0).to(self.device, non_blocking=True)
                detection = self.model(x, mode='outs')
                
                # skip j = 0, because it's the background class
                for j in range(1, detection.size(1)):
                    dets = detection[0, j, :]
                    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                    dets = torch.masked_select(dets, mask).view(-1, 5)
                    if dets.size(0) == 0: continue
                    boxes = dets[:, 1:]
                    boxes[:, 0] *= w
                    boxes[:, 2] *= w
                    boxes[:, 1] *= h
                    boxes[:, 3] *= h
                    scores = dets[:, 0].cpu().numpy()
                    cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32, copy=False)
                    all_boxes[j][i] = cls_dets
      
        
        logger.info(f'GPU: {self.device} |||  => Saving Detections')
        with open(det_file, 'wb') as f: pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        logger.info(f'GPU: {self.device} |||  DONE\n')

        logger.info(f'GPU: {self.device} |||  => Evaluating detections...')
        results_prefix = str(self.device).replace(':', '_')
        write_voc_results_file(all_boxes, self.test_dl.dataset, self.dataset.classes, output_dir, results_prefix)
        mAP = do_python_eval(self.dataset.classes, self.dataset.dataset_path, output_dir, results_prefix)
        logger.info(f'GPU: {self.device} |||  DONE\n')
        
        self.backbone.change_phase() # -> back to the train phase for the query and the next AL iteration
        
        torch.cuda.empty_cache()

        return torch.tensor([mAP], device=self.device)
    


    def adjust_learning_rate(self, step: int) -> None:
        """Sets the learning rate to the initial LR decayed by 10 at every
            specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        # lr = args.lr * (gamma ** (step))
        lr = 0.01 * (0.1 ** (step))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr