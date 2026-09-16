# -*- coding: utf-8 -*-

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch

from dataset import Cls_Datasets

# Enable only when debugging NaNs/inf in the backward pass — it roughly doubles
# training time, so keep it off for full experimental runs.
# torch.autograd.set_detect_anomaly(True)

from models.BBone_Module import Master_Model

from init import get_backbone, get_ll_module_params, get_masters, get_strategies_object, dict_backbone, dict_strategies_GTG
from strategies.gtg.GTG import GTG
from utils import (
    create_directory, create_exp_path, plot_trail_acc, plot_res_std_mean, set_seeds, save_yamal,
    get_device, read_rows, purge_rows, get_git_commit
)

    	
from config import cls_config, al_params
from datetime import datetime
import argparse
import time
from typing import Dict, Any, List, Tuple

import logging
logger = logging.getLogger(__name__)



def get_args() -> argparse.Namespace:
    """Parse and return the command-line arguments for the application.
    Returns:
        argparse.Namespace: Parsed command-line arguments containing the selected
            methods, datasets, and configuration options for training and active
            learning experiments.
    Raises:
        SystemExit: If argument parsing fails or required arguments are missing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpus', '--gpus', type=int, required=False, default=1, help='Number of GPUs to use during training')
    parser.add_argument('-ma', '--master_address', type=str, required=False, default='localhost', help='Master address for distributed training')
    parser.add_argument('-mp', '--master_port', type=str, required=False, default='12355', help='Master port for distributed training')
    parser.add_argument('-m', '--methods', type=str, nargs='+', required=True, choices=[
            'random', 'entropy', 'coreset', 'badge', 'bald', 'cdal', 'tavaal', 'alphamix', 'tidal',
            'uncertaingcn', 'coregcn',
            'll', 'gtg', 'll_gtg', 'llmlp_gtg', 'lsmlps_gtg', 'lsmlpsA_gtg', 'lstmreg_gtg', 'lstmbc_gtg'
        ], help='Possible methods to choose')
    parser.add_argument('-ds', '--datasets', type=str, nargs='+', required=True, choices=['cifar10', 'cifar100', 'svhn', 'fmnist', 'caltech256', 'tinyimagenet'],
                        help='Possible datasets to choose')
    parser.add_argument('-tr', '--trials', type=int, required=False, default=5, help='AL trials')
    parser.add_argument('-o', '--exp_path', type=str, required=False, default=None,
                        help='Output folder under results/ (default: current timestamp). Pass an '
                             'existing one to accumulate several invocations into a single '
                             'results.csv — this is what lets a sweep run one method per process '
                             'and be resumed after a crash')
    parser.add_argument('-bbone', '--bbone_pre_train', required=False, action='store_true', help='BackBone pre-train for GTG Module')
    parser.add_argument('-cs_s', '--cold_start_strategy', type=str, required=False, choices=['none', 'mse', 'kmeans'], default='none', help='Cold start strategy')
    
   
    parser.add_argument('-tulp', '--temp_unlab_pool', required=False, action='store_true', help='Temporary Unlabelled Pool')
    parser.add_argument('-nw', '--num_workers', type=int, required=False, default=4,
                        help='DataLoader worker processes for the single-GPU path. 0 loads in the '
                             'main process and leaves the GPU waiting on the CPU; set it to the '
                             'number of available cores. Use 0 if workers misbehave in your container')
    parser.add_argument('-diag', '--diagnostics', required=False, action='store_true',
                        help='Enable the expensive analysis artefacts (t-SNE plots, affinity-matrix plots, '
                             'entropy-history plots, per-sample labelled image dumps). OFF by default: they can '
                             'dominate the wall-clock of a full benchmark run. Accuracy CSVs and train/val '
                             'curves are always written')
    parser.add_argument('-am', '--affinity_matrix', type=str, nargs='+', required=False, choices=['corr', 'cos_sim', 'rbfk'], default=['corr'], help='Affinity matrix to choose')
    
    parser.add_argument('-am_md', '--affinity_matrix_mixed_distance', type=str, nargs='+', required=False, choices=['LL', 'UU', 'LU', 'UL'], default=['LL'], help='Affinity matrix portion to convert into distance measure')
    parser.add_argument('-am_sl', '--affnity_matrix_self_loop', required=False, action='store_true', help='Affinity matrix cells self loop (default: no self loop)')
    
    parser.add_argument('-am_s', '--affinity_matrix_strategy', type=str, required=False, choices=['uncertanity', 'diversity', 'mixed'], default='mixed', 
                       help='Different affinity matrix modification')
    parser.add_argument('-am_ts', '--affinity_matrix_threshold_strategies', type=str, required=False, nargs='+', choices=['threshold', 'mean', 'none'], default=['mean'],
                        help='Possible treshold strategy types to choose to apply in the affinity matrix. '
                             'Every GTG method is run once per -am x -am_ts combination')
    parser.add_argument('-am_t', '--affinity_matrix_threshold', type=float, required=False, default=0.5,
                        help='Affinity Matrix Threshold for our method, when threshold_strategy = mean, this is ignored')
    parser.add_argument('-am_r', '--affinity_matrix_redundancy', type=float, required=False, default=0.0,
                        help='Redundancy weight lambda >= 0: amplifies the labelled->unlabelled coupling so that '
                             'samples redundant with the labelled set converge sooner and are less likely to be '
                             'queried. 0 (default) recovers standard GTG')
    parser.add_argument('-e_s', '--entropy_strategy', type=str, required=False, choices=['mean', 'integral'], default='mean',
                        help='Entropy strategy to sum up the entropy history')
    
    parser.add_argument('-gtg_iter', '--gtg_iterations', type=int, required=False, default=30, help='Maximum GTG iterations to perorm')
    parser.add_argument('-gtg_t', '--gtg_tollerance', type=float, required=False, default=0.0001, help='GTG tollerance')
        
    #parser.add_argument('-plb', '--perc_labelled_batch', type=float,  required=False, default=0.5,
    #                    help='Number of labelled observations to mantain in each batch during GTG end-to-end version')
    parser.add_argument('-bsgtgo', '--batch_size_gtg_online', type=int,  required=False, default=32, help='Initial batch size for the online GTG version')
    parser.add_argument('-dbg_gtg', '--debug_gtg', required=False, action='store_true',
                        help='Log the GTG replicator dynamics at DEBUG: per-iteration state and the '
                             'gradient hooks used to trace weights diverging to NaN. Off by default '
                             'because each log formats a tensor and synchronises with the GPU '
                             '(~11 KB and ~136 syncs per training batch)')
    parser.add_argument('-ex', '--existing', type=str, required=False, choices=['append', 'resume', 'overwrite'], default='append',
                        help='What to do with rows already in the -o results.csv for a strategy and trial. '
                             'append: run and append (duplicates rows). resume: skip a trial whose rows are '
                             'complete, discard a partial one and re-run it. overwrite: discard and re-run')

    args = parser.parse_args()

    # Online GTG strategies share one GTGModule whose affinity config is set when each strategy is
    # constructed, and all are constructed before any runs: with several combinations every online
    # strategy would silently train and query with the last one while its name advertises its own.
    online_gtg = [m for m in args.methods if dict_strategies_GTG.get(m) is GTG]
    n_configs = len(args.affinity_matrix) * len(args.affinity_matrix_threshold_strategies)
    if online_gtg and n_configs > 1:
        parser.error(f'{", ".join(online_gtg)} would all run with the last of the {n_configs} -am x -am_ts '
                     'combinations; pass exactly one -am and one -am_ts value')
    return args




def run_strategy(
        strategy: Any,
        trial: int,
        exp_path: str,
        dataset_name: str,
        existing: str,
        n_lab_obs: List[int]
    ) -> Dict[str, List[float]]:
    """Run one strategy for one trial, honouring the `--existing` policy for rows already on disk.

    Trials are independent (the initial split and every pool are seeded by dataset, trial and
    cycle; each cycle reloads the `_init` checkpoint and resets the seeds), so a skipped or re-run
    trial does not change the others.

    Args:
        strategy: A constructed strategy object exposing `strategy_name` and `run()`.
        trial: The trial index (written to the `iter` column).
        exp_path: The experiment folder under results/.
        dataset_name: The dataset name.
        existing: 'append', 'resume' or 'overwrite' (see `--existing`).
        n_lab_obs: The labelled-set size of every AL cycle, i.e. the expected `lab_obs` values.

    Returns:
        Dict[str, List[float]]: The test metrics per cycle, read back from results.csv when the
            trial is skipped.
    """
    name = strategy.strategy_name

    if existing == 'resume':
        rows = [row for row in read_rows('results.csv', exp_path, dataset_name) if row[:2] == [name, str(trial)]]
        if sorted(int(row[2]) for row in rows) == n_lab_obs:
            logger.info(f' => {name} trial {trial} is already complete in results.csv: skipping it')
            test_key = list(cls_config["results_dict"]["test"].keys())[0]
            return {test_key: [float(row[3]) for row in sorted(rows, key=lambda row: int(row[2]))]}

    if existing in ('resume', 'overwrite'):
        for filename in ('results.csv', 'count_classes.csv', '_count_classes.csv'):
            n_removed = purge_rows(filename, exp_path, dataset_name, name, trial)
            if n_removed: logger.info(f' => Removed {n_removed} rows of {name} trial {trial} from {filename} before running it')

    return strategy.run()



def run_strategies(
        ct_p: Dict[str, Any],
        t_p: Dict[str, Any],
        gtg_p: Dict[str, Any],
        Masters: Dict[str, Master_Model],
        methods: List[str],
        existing: str = 'append'
    ) -> Tuple[Dict[str, Dict[str, List[float]]], List[int]]:
    """
    Run the selected active learning strategies and collect their results.
    This function builds the strategy objects from the provided configuration,
    executes each strategy sequentially, and returns a mapping of strategy names
    to their outputs along with the expected number of labeled observations at
    each active learning iteration.
    Args:
        ct_p: Configuration parameters for the CT strategy family.
        t_p: Configuration parameters for the T strategy family.
        gtg_p: Configuration parameters for the GTG strategy family.
        Masters: Mapping of master model names to their corresponding model instances.
        methods: List of strategy method names to instantiate and run.
        existing: Policy for rows already in results.csv ('append', 'resume', 'overwrite').
    Returns:
        A tuple containing:
         - A dictionary mapping each strategy name to the result returned by
                          that strategy's ``run()`` method.
                        - A list of labeled-observation counts, one per active learning
                          iteration.
                """
    results = { }
    n_lab_obs = [al_params["init_lab_obs"] + (iter * al_params["n_top_k_obs"]) for iter in range(al_params["al_iters"])]
     
    # get the strategis object to run them
    strategies = get_strategies_object(methods=methods, Masters=Masters, ct_p=ct_p, t_p=t_p, gtg_p=gtg_p)
    
    ##########################################################################################################
    # START THE ACTIVE LEARNING PROECSS
    for strategy in strategies:
        logger.info(f'-------------------------- {strategy.strategy_name} --------------------------\n')
        results[strategy.strategy_name] = run_strategy(
            strategy=strategy, trial=ct_p["trial"], exp_path=ct_p["exp_path"],
            dataset_name=ct_p["dataset_name"], existing=existing, n_lab_obs=n_lab_obs
        )
    ##########################################################################################################
    
    return results, n_lab_obs               



def main() -> None:
    """Run the active learning benchmark workflow.

    This entry point parses command-line arguments, initializes runtime
    configuration (logging, device, reproducibility), builds datasets and model
    components, executes selected strategies across trials, and saves artifacts
    (plots and YAML configs).

    Raises:
        ValueError: If backbone pre-training is enabled without a cold-start
            strategy.
    """
    args = get_args()
    
    exp_path = args.exp_path or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    create_directory(dir=f'results/{exp_path}')
    
    logging.basicConfig(filename = f'results/{exp_path}/exp_log.log',
                    filemode = 'a',
                    format = '%(asctime)s - %(levelname)s: %(message)s',
                    datefmt = '%d-%m-%y - %H:%M:%S',
                    level = logging.INFO)

    # Raise only the GTG loggers: the rest of the run stays at INFO.
    if args.debug_gtg:
        for gtg_logger in ('models.modules.GTG.GTG_Cls', 'strategies.gtg.GTG_off'):
            logging.getLogger(gtg_logger).setLevel(logging.DEBUG)

    logger.info(f'Code version: {get_git_commit()}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    device = get_device(n_gpus=args.gpus)
    
    # setting seed and deterministic behaviour of pytorch for reproducibility
    set_seeds()
            
        
    gtg_params = {
        'gtg_t': args.gtg_tollerance,
        'gtg_i': args.gtg_iterations,
        
        'am': args.affinity_matrix,
        'am_s': args.affinity_matrix_strategy,
        'am_ts': args.affinity_matrix_threshold_strategies,
        'am_t': args.affinity_matrix_threshold,
        'am_r': args.affinity_matrix_redundancy,
        'e_s': args.entropy_strategy,
        'am_md': args.affinity_matrix_mixed_distance,
        'am_sl': args.affnity_matrix_self_loop,
        
        #'plb': args.perc_labelled_batch,
        'bsgtgo': args.batch_size_gtg_online,
    }
    
        
    for dataset_name in args.datasets:
        
        logger.info(f'----------------------- RUNNING ACTIVE LEARNING BENCHMARK ON {dataset_name} -----------------------\n')
        
        Dataset = Cls_Datasets(dataset_name) # get the dataset
        BBone = get_backbone(
            n_classes=Dataset.n_classes, 
            n_channels=Dataset.n_channels, 
            bbone=dict_backbone[dataset_name], 
            img_size=Dataset.image_size
        ) # the backbone is the same for all

        # create gtg dictionary parameters
        gtg_module_params = dict(
            **gtg_params, n_classes = Dataset.n_classes, 
            embedding_dim = BBone.get_rich_features_shape(),
            device = device
        )
        
        ll_module_params = get_ll_module_params(image_size=Dataset.image_size, dataset_name=dataset_name) # create learning loss dictionary parameters
            
        Masters = get_masters(
            methods=args.methods, 
            BBone=BBone, 
            ll_module_params=ll_module_params, 
            gtg_module_params=gtg_module_params, 
            dataset_name=dataset_name, 
            n_classes=Dataset.n_classes
        ) # obtain the master models
                                                
        common_training_params = {
            'device': device, 'exp_path': exp_path,
            'dataset_name': dataset_name, 'gpus': args.gpus, 
            'master_address': args.master_address, 'master_port': args.master_port,
            'temp_unlab_pool': args.temp_unlab_pool, 'bbone_pre_train': args.bbone_pre_train,
            'diagnostics': args.diagnostics, 'num_workers': args.num_workers,
        }
        
        if args.cold_start_strategy == 'none' and args.bbone_pre_train: raise ValueError('You must specify a cold start strategy to pre-train the backbone')
        if args.cold_start_strategy != 'none': args.trials = 1
        
        for trial in range(args.trials):
            common_training_params["trial"] = trial
            
            logger.info(f'----------------------- SAMPLE ITERATION {trial + 1} / {args.trials} -----------------------\n')
            
            create_exp_path(exp_path=exp_path, dataset_name=dataset_name, iter=str(trial))
            
            #intial_weights = Dataset.get_initial_subsets(device)  # type: ignore
            if args.cold_start_strategy != 'none':
                # adding the initial weights to the common_training_params to be loaded in the backbnone
                bbone_init_weights = Dataset.get_initial_subsets_cold_start(device=device, strategy=args.cold_start_strategy) # type: ignore 
                common_training_params["bbone_init_weights"] = bbone_init_weights if args.bbone_pre_train else None
            else: 
                Dataset.get_initial_subsets(trial_id=trial)
                common_training_params["bbone_init_weights"] = None

            results, n_lab_obs = run_strategies(
                ct_p = {**common_training_params, 'Dataset': Dataset},
                t_p = cls_config,
                gtg_p = gtg_params,
                Masters = Masters,
                methods = args.methods,
                existing = args.existing
            )
            
            plot_trail_acc(
                dataset_name=dataset_name, 
                trial=trial, 
                methods_results=results, 
                n_lab_obs=n_lab_obs, 
                exp_path=exp_path,
                key=list(cls_config["results_dict"]["test"].keys())[0],
                plot_png_name=f'{dataset_name}/{trial}/results.png'
            )
        
        plot_res_std_mean(exp_path=exp_path, dataset_name=dataset_name)

        del common_training_params["trial"]
        # saving yamal configuration file
        save_yamal(
            common_training_params=common_training_params, 
            task_params={
                'epochs': cls_config['epochs'], 
                'datasets_optim': cls_config[dataset_name]
            }, 
            al_params=al_params, 
            gtg_params=gtg_params, 
            exp_path=exp_path, 
            dataset_name=dataset_name
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