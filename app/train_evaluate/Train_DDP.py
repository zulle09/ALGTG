
import copy
import torch

from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SubsetRandomSampler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from models.BBone_Module import Master_Model
from utils import set_seeds
from datasets_creation.Detection import detection_collate
from .Workers.Cls_TrainWorker import Cls_TrainWorker
from .Workers.Det_TrainWorker import Det_TrainWorker

import os
import gc
from typing import Tuple, Dict, Any, List
from multiprocessing import connection 



def initialize_preocess(rank: int, world_size: int) -> None:
    
    torch.cuda.set_device(rank)
    
    init_process_group(backend='nccl', 
                       init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}', 
                       rank=rank, world_size=world_size)
    
    
    
def clear_process() -> None: destroy_process_group()



def train_ddp(rank: int, world_size: int, params: Dict[str, Any], conn: connection.Connection) -> None:
    set_seeds()
    
    initialize_preocess(rank, world_size)
    
    ct_p = params["ct_p"]
    t_p = params["t_p"]
    
    moved_model: Master_Model = copy.deepcopy(ct_p["Master_Model"]).to(rank)
    batch_size = t_p[ct_p["dataset_name"]]["batch_size"][params["module_name_only"]]
    num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    
    # deep copy the model (it is in the RAM) and then move it to the realive gpu
    
    moved_model = DDP(moved_model, device_ids=[rank], output_device=rank, find_unused_parameters=True) # type: ignore
    ct_p["Model_train"] = moved_model
    
    dict_dl = dict(
        batch_size=batch_size, shuffle=False, pin_memory=True, 
        persistent_workers=True, num_workers=num_workers
    )
    
    if ct_p["task"] == 'detection': dict_dl["collate_fn"] = detection_collate
        
    if world_size % 2 != 0: dict_dl["drop_last"] = True
    
    train_results = [torch.zeros((4, t_p["epochs"]), device=torch.device(rank)) for _ in range(world_size)]
    test_results = [torch.zeros(1, device=torch.device(rank)) for _ in range(world_size)]
    
    
    lab_train_ds = Subset(ct_p["Dataset"].lab_train_ds, params["labelled_subset"])
    params["train_dl"] = DataLoader(
        lab_train_ds,
        #sampler=DistributedSampler(params["labelled_subset"], num_replicas=world_size,
        
        # TiDAL WON'T WORK SINCE MMOVING_PROBS IS NOT ACCESSIBLE FROM SUBSET
        sampler=DistributedSampler(lab_train_ds, num_replicas=world_size,
                                   rank=rank, shuffle=True, seed=100001),
        **dict_dl
    )
    
    params["test_dl"] = DataLoader(
        ct_p["Dataset"].test_ds, 
        sampler=DistributedSampler(ct_p["Dataset"].test_ds, num_replicas=world_size, 
                                   rank=rank, shuffle=False, seed=100001),
        **dict_dl
    )
    
    
    if ct_p["task"] == 'detection':  
        # determine the number of iterations by -> iterations = epochs * (len(splitted_train_ds) // batch_size)
        #                                               x     =  300   * (len(train_ds)/world_size) / batch_size)
        t_p["epoch_size"] = len(params["train_dl"].dataset) // batch_size # type: ignore
        t_p["max_iter"] = t_p["epochs"] * t_p["epoch_size"]
        
        train_test = Det_TrainWorker(rank, params, world_size)
        
    else: train_test = Cls_TrainWorker(rank, params, world_size)
        
    
    if rank == 0:
        dist.gather(train_test.train(), train_results)
        dist.gather(train_test.test(), test_results)
    else:
        dist.gather(train_test.train())
        dist.gather(train_test.test())     
    
    dist.barrier()
    
    
    # shutdown the worker
    del params["train_dl"]._iterator
    del params["test_dl"]._iterator
    
    gc.collect()
    
    if rank == 0:
        train_results = (torch.sum(torch.stack(train_results), dim=0) / world_size).cpu().tolist()
        test_results = (torch.sum(torch.stack(test_results), dim=0) / world_size).cpu().tolist()
        
        conn.send((train_results, test_results))       

    dist.barrier()

    clear_process()

    
    
def train(params: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    set_seeds()
    
    ct_p = params["ct_p"]
    t_p = params["t_p"]   
    
    ct_p["Master_Model"] = ct_p["Master_Model"].to(params["ct_p"]["device"])
    batch_size = t_p[ct_p["dataset_name"]]["batch_size"][params["module_name_only"]]
    params["batch_size"] = batch_size
    
    dict_dl = dict(batch_size=batch_size, pin_memory=True)
    
    if ct_p["task"] == 'detection': 
        dict_dl["collate_fn"] = detection_collate
        
        t_p["epoch_size"] = len(params["labelled_subset"]) // batch_size
        t_p["max_iter"] = t_p["epochs"] * t_p["epoch_size"]
        
        TrainWorker = Det_TrainWorker
    
    else: TrainWorker = Cls_TrainWorker
    
    if params["module_name_only"] == 'GTGModule':
        train_dl = (
            Subset(ct_p["Dataset"].train_ds, params["labelled_indices"]),
            Subset(ct_p["Dataset"].train_ds, params["rand_unlab_sample"])
        )
    else: 
        train_dl = DataLoader(ct_p["Dataset"].train_ds, sampler=SubsetRandomSampler(params["labelled_indices"]), **dict_dl)
        
    params["train_dl"] = train_dl
    params["test_dl"] = DataLoader(ct_p["Dataset"].test_ds, shuffle=False, **dict_dl)
    
    train_test = TrainWorker(params["ct_p"]["device"], params)
    
        
    train_results = train_test.train().cpu().tolist()
    test_results = train_test.test().cpu().tolist()
    
    return train_results, test_results


