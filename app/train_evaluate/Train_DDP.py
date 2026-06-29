
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
from .Worker import Cls_TrainWorker


import os
import gc
from typing import Tuple, Dict, Any, List
from multiprocessing import connection 



def initialize_preocess(rank: int, world_size: int) -> None:
    """Initialize the distributed process group for multi-GPU training.
    Sets up the CUDA device for the specified process rank and initializes
    the PyTorch distributed process group using NCCL backend for efficient
    GPU-to-GPU communication.
    Args:
        rank: The rank of the current process within the distributed group.
        world_size: The total number of processes participating in the
            distributed training.
    Returns:
        None.
    Raises:
        RuntimeError: If the distributed process group fails to initialize.
    """
    
    torch.cuda.set_device(rank)
    
    init_process_group(backend='nccl', 
                       init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}', 
                       rank=rank, world_size=world_size)
    
    
# Clean up the distributed process group after training is complete
def clear_process() -> None: destroy_process_group()



def train_ddp(rank: int, world_size: int, params: Dict[str, Any], conn: connection.Connection) -> None:
    """
    Trains and evaluates a model using Distributed Data Parallel (DDP) in a multi-GPU setting.
    This function initializes a distributed training process, sets up the model with
    PyTorch DDP, creates data loaders with distributed samplers, runs training and
    evaluation via a Cls_TrainWorker, and gathers results across all processes.
    
    Args:
        rank (int): The rank of the current process within the distributed group.
        world_size (int): The total number of processes in the distributed group.
        params (Dict[str, Any]): A dictionary containing configuration parameters including:
            - ct_p: Contains Master_Model and Dataset
            - t_p: Contains training parameters like batch_size and epochs
            - module_name_only: Module identifier for batch size lookup
            - labelled_subset: Subset indices for labeled training data
        conn (connection.Connection): Connection object used to send results back
            to the main process.
    
    Returns:
        None. Results are sent via the connection object to rank 0.
    
    Raises:
        Exception: Any exceptions during training/evaluation are propagated.
    
    Note:
        - The function uses DistributedSampler for both train and test data loaders.
        - If world_size is odd, drop_last is set to True to handle uneven batches.
        - Results are averaged across all ranks before being sent via connection.
        - A barrier is applied before cleanup to ensure all processes synchronize.
    """
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
            
    if world_size % 2 != 0: dict_dl["drop_last"] = True
    
    train_results = [torch.zeros((4, t_p["epochs"]), device=torch.device(rank)) for _ in range(world_size)]
    test_results = [torch.zeros(1, device=torch.device(rank)) for _ in range(world_size)]
    
    
    lab_train_ds = Subset(ct_p["Dataset"].train_ds, params["labelled_indices"])
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
    
    
    train_test = Cls_TrainWorker(rank, params, world_size)
        
    
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
    """Trains and evaluates a model using the provided parameters.
    This function sets up the training environment, initializes data loaders,
    and executes the training and evaluation processes using a TrainWorker.
    Args:
        params (Dict[str, Any]): A dictionary containing all necessary parameters
            for training and evaluation, including:
            - ct_p: Configuration parameters containing dataset, model, and device info
            - t_p: Training parameters including batch sizes
            - module_name_only: The name of the module to use (e.g., 'GTGModule')
            - labelled_indices: Indices of labeled samples for training
            - rand_unlab_sample: Indices of randomly sampled unlabeled samples
    Returns:
        Tuple[List[float], List[float]]: A tuple containing:
            - train_results: List of training metrics (e.g., loss values)
            - test_results: List of test metrics (e.g., loss values)
    Raises:
        KeyError: If required keys are missing from the params dictionary.
    """
    set_seeds()
    
    ct_p = params["ct_p"]
    t_p = params["t_p"]   
    
    ct_p["Master_Model"] = ct_p["Master_Model"].to(params["ct_p"]["device"])
    batch_size = t_p[ct_p["dataset_name"]]["batch_size"][params["module_name_only"]]
    params["batch_size"] = batch_size
    
    dict_dl = dict(batch_size=batch_size, pin_memory=True)
    
    TrainWorker = Cls_TrainWorker
    
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


