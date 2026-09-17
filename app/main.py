# -*- coding: utf-8 -*-

import torch

torch.autograd.set_detect_anomaly(True) # type: ignore
import torch.distributed as dist

from models.BBone_Module import Master_Model

from init import get_backbone, get_dataset, get_ll_module_params, get_masters, get_strategies_object
from utils import create_directory, create_exp_path, plot_trail_acc, plot_res_std_mean, set_seeds, save_yamal

    	
from config import al_params, det_config
from datetime import datetime
import argparse
import time
from typing import Dict, Any, List, Tuple

import os
import logging
logger = logging.getLogger(__name__)



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpus', '--gpus', type=int, required=False, default=1, help='Number of GPUs to use during training')
    parser.add_argument('-m', '--methods', type=str, nargs='+', required=True, choices=['random', 'll'], help='Possible methods to choose')
    parser.add_argument('-ds', '--datasets', type=str, nargs='+', required=True, choices=['voc'], help='Possible datasets to choose')
    parser.add_argument('-tr', '--trials', type=int, required=False, default=5, help='AL trials')
    parser.add_argument('-tulp', '--temp_unlab_pool', required=False, action='store_true', help='Temporary Unlabelled Pool')

    args = parser.parse_args()
    return args




def run_strategies(ct_p: Dict[str, Any], t_p: Dict[str, Any],
                   Masters: Dict[str, Master_Model], methods: List[str]) -> Tuple[Dict[str, Dict[str, List[float]]], List[int]]:
    #Tuple[Dict[str, Dict[str, List[float]]], Dict[str, Dict[int, Dict[str, int]]], List[int]]:

    results = { }
    #count_classes = { }
    n_lab_obs = [al_params["init_lab_obs"] + (iter * al_params["n_top_k_obs"]) for iter in range(al_params["al_iters"])]
     
    # get the strategis object to run them
    strategies = get_strategies_object(methods, Masters, ct_p, t_p)
    
    ##########################################################################################################
    # START THE ACTIVE LEARNING PROECSS
    for strategy in strategies:
        logger.info(f'-------------------------- {strategy.strategy_name} --------------------------\n')
        #results[strategy.strategy_name], count_classes[strategy.strategy_name] = strategy.run()
        results[strategy.strategy_name] = strategy.run()
    ##########################################################################################################
    
    #return results, count_classes, n_lab_obs               
    return results, n_lab_obs               


def get_device(args) -> torch.device:
    if torch.cuda.is_available():
        logger.info(f'Using {args.gpus} / {torch.cuda.device_count()} of the available GPUs')
        device = torch.device('cuda')
        if dist.is_available(): logger.info('We can train on multiple GPU')
        if dist.is_nccl_available(): logger.info('NCCL backend available')
    else:
        logger.info('CUDA is not available. Using CPU')
        device = torch.device('cpu')

    logger.info(f'Application running on {device}\n')
    
    return device



def main() -> None:
    args = get_args()
    
    exp_path = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    create_directory(f'results/{exp_path}')
    
    logging.basicConfig(filename = f'results/{exp_path}/exp_log.log',
                    filemode = 'a',
                    format = '%(asctime)s - %(levelname)s: %(message)s',
                    datefmt = '%H:%M:%S',
                    level = logging.INFO)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    device = get_device(args)
    
    # setting seed and deterministic behaviour of pytorch for reproducibility
    set_seeds()
            
        
    task_params = det_config

    for dataset_name in args.datasets:

        logger.info(f'----------------------- RUNNING ACTIVE LEARNING BENCHMARK ON {dataset_name} -----------------------\n')
        
        Dataset = get_dataset(dataset_name) # get the dataset
        BBone = get_backbone(Dataset.n_classes) # the backbone is the same for all

        ll_module_params = get_ll_module_params() # create learning loss dictionary parameters
            
        Masters = get_masters(args.methods, BBone, ll_module_params, dataset_name) # obtain the master models
                                                
        common_training_params = {
            'device': device, 'exp_path': exp_path,
            'dataset_name': dataset_name, 'gpus': args.gpus,
            'temp_unlab_pool': args.temp_unlab_pool,
        }
        
        for trial in range(args.trials):
            common_training_params["trial"] = trial
            
            logger.info(f'----------------------- SAMPLE ITERATION {trial + 1} / {args.trials} -----------------------\n')
            
            create_exp_path(exp_path, dataset_name, str(trial))
            
            Dataset.get_initial_subsets(trial)

            results, n_lab_obs = run_strategies(
                ct_p = {**common_training_params, 'Dataset': Dataset}, t_p = task_params, Masters = Masters, methods = args.methods
            )
            
            plot_trail_acc(dataset_name, trial, results, n_lab_obs, exp_path,
                             list(task_params["results_dict"]["test"].keys())[0],
                             plot_png_name=f'{dataset_name}/{trial}/results.png')
        
        plot_res_std_mean(exp_path, dataset_name)

        del common_training_params["trial"]
        # saving yamal configuration file
        save_yamal(
            common_training_params, 
            {'epochs': task_params['epochs'], 'datasets_optim': task_params[dataset_name]}, 
            al_params, exp_path, dataset_name
        )
            

if __name__ == "__main__":
    try:
        start = time.time()
        main()
        end = time.time()
        
        logger.info(f'####################################################### TOTAL ELAPSED SECONDS: {end-start} #######################################################')
    except Exception as e:
        logger.exception("main.py has crashed. Error: %s", e)
        raise e