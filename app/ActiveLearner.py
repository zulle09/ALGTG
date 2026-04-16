
from models.BBone_Module import Master_Model
from dataset import Cls_Datasets
from train_evaluate.Train_DDP import train, train_ddp
from utils import (
    count_class_observation, plot_cumulative_train_results, set_seeds,
    create_class_dir, create_method_res_dir, plot_new_labelled_tsne, save_train_val_curves, write_csv,
    plot_classes_count_iterations, plot_entropy_iterations_classes, get_device
)
from config import al_params

from torch.utils.data import Subset, DataLoader
import torch.multiprocessing as mp
from torchvision.utils import save_image
import torch
import numpy as np
from typing import List, Dict, Any
import copy
import os

import logging
logger = logging.getLogger(__name__)



class ActiveLearner():
    def __init__(self, ct_p: Dict[str, Any], t_p: Dict[str, Any], method_name: str) -> None:
        
        self.iter = 1
        
        self.ct_p: Dict[str, Any] = ct_p
        self.t_p: Dict[str, Any] = t_p
        
        self.device: torch.device = self.ct_p["device"]
        self.model: Master_Model = self.ct_p["Master_Model"]
        self.model = self.model.to(self.device)
        
        self.dataset: Cls_Datasets = self.ct_p["Dataset"]
        
        self.ds_t_p = self.t_p[self.ct_p["dataset_name"]]
        self.batch_size = self.ds_t_p["batch_size"][self.model.only_module_name]

        # each strategy will have a separate labelled and unlabelled set to work independently from the original sets
        self.labelled_indices: List[int] = copy.deepcopy(self.dataset.labelled_indices)
        self.unlabelled_indices: List[int] = copy.deepcopy(self.dataset.unlabelled_indices)
        self.temp_unlab_pool = []

        self.train_results: Dict[int, Any] = {}
        
        self.method_name = method_name
        self.strategy_name = f'{self.model.name}_{method_name}' # define strategy name    
        self.best_check_filename: str = f'app/checkpoints/{self.ct_p["dataset_name"]}/{self.strategy_name}'
        
        self.world_size: int = self.ct_p["gpus"]
        self.master_address: str = self.ct_p["master_address"]
        self.master_port: str = self.ct_p["master_port"]
        
        self.path = f'results/{self.ct_p["exp_path"]}/{self.ct_p["dataset_name"]}/{self.ct_p["trial"]}/{self.strategy_name}'
        create_method_res_dir(path=self.path)
        
        # save initial labelled images
        self.save_labelled_images(new_labelled_idxs=self.labelled_indices)
        
        self.count_classes: Dict[int, List[int]] = {}
        self.count_classes[self.iter * al_params["n_top_k_obs"]] = list(
            count_class_observation(
                classes=self.dataset.classes, 
                dataset=Subset(self.dataset.train_ds, self.labelled_indices)).values()
        )

        # save the initial count classes in the csv file
        write_csv(
            filename = '_count_classes.csv',
            exp_path = self.ct_p["exp_path"],
            dataset_name = self.ct_p["dataset_name"],
            head = ['method', 'iter', 'lab_obs'] + self.dataset.classes,
            values = [self.strategy_name, self.ct_p["trial"], self.iter * al_params["n_top_k_obs"]] + self.count_classes[self.iter * al_params["n_top_k_obs"]]
        )
        
        
    
    def save_labelled_images(self, new_labelled_idxs: List[int]) -> None:
        """Save newly labelled images to disk for visual analysis.
        Iterates through the newly labelled images from the active learning
        iteration and saves them to class-specific directories within the
        iteration folder. Handles both classification and other tasks (e.g.,
        segmentation) by extracting unique labels from ground truth annotations
        when applicable.
        
        Args:
            new_labelled_idxs: List of indices corresponding to the newly labelled
                images in the unlabeled training dataset.
        
        Returns:
            None.
        
        Raises:
            OSError: If there are issues creating directories or saving files.

        """
        logger.info(f' => Iteration {self.iter} Method {self.strategy_name} - Saving the new labelled images for further visual analysis...')
        create_class_dir(base_path=self.path, iter=self.iter, classes=self.dataset.classes)
        
        for idx_top, (_, img, gt) in enumerate(Subset(self.dataset.unlab_train_ds, new_labelled_idxs)): # type: ignore
            save_image(img, f'{self.path}/new_labelled_images/{self.iter}/{self.dataset.classes[gt]}/{idx_top}.png')
        logger.info(' DONE\n')
        
        
    
    def get_rand_unlab_sample(self) -> None:
        """
        Get a random sample of unlabelled indices for active learning.
        Selects a random subset of unlabelled indices to be used in the current
        active learning iteration. The selection process depends on the task type.
            - Sets a reproducible seed based on dataset_id, trial number, and
              current iteration to ensure deterministic sampling across trials.
            - Randomly permutes the unlabelled indices and selects the first
              'unlab_sample_dim' indices as the random sample.
            - Logs the seed and last 10 permuted indices for debugging.
            - Resets the seed to its original value after sampling.
            - If 'temp_unlab_pool' is True, removes the selected samples from
              the unlabelled indices pool for the current iteration.

        
        Returns:
            None: Updates self.rand_unlab_sample with the selected indices
                  and optionally modifies self.unlabelled_indices.
        
        Raises:
            KeyError: If required keys are missing in al_params or ct_p dictionaries.
        """
        # set seed for reproducibility            
        seed = self.dataset.dataset_id * (self.ct_p["trial"] * al_params["al_iters"] + (self.iter - 1))
        set_seeds(seed)
        
        rand_perm = torch.randperm(len(self.unlabelled_indices)).tolist()
        # it has the indices of the unlabelled sample
        self.rand_unlab_sample = [self.unlabelled_indices[idx] for idx in rand_perm[:al_params["unlab_sample_dim"]]]
        
        logger.info(f' SEED: {seed} - Last 10 permuted indices are: {rand_perm[-10:]}')
        
        #reset the original seed
        set_seeds()
        
        if self.ct_p["temp_unlab_pool"]:
            # removing the whole observation sample fromt the unlabelled indices list
            for idx in self.rand_unlab_sample: self.unlabelled_indices.remove(idx) # - 10000


    def load_best_checkpoint(self):
        """Load the best model checkpoint from disk.

        Determines the appropriate device (CUDA or CPU) based on the distributed
        training configuration and world size, then loads the corresponding
        checkpoint file. The model state dictionary is restored from the
        checkpoint.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the checkpoint file is corrupted or missing
                required keys (e.g., 'state_dict').
        """
        device = get_device(n_gpus=self.world_size)

        filename = f'{self.best_check_filename}_{device}.pth.tar'
        checkpoint: Dict = torch.load(filename, map_location=self.device)
        logger.info(f' => Loading {filename} Checkpoint')
        self.model.load_state_dict(checkpoint["state_dict"])
            
    
    def get_embeddings(self, dataloader: DataLoader, dict_to_modify: Dict[str, Any], embedds2cpu: bool = False) -> None:
        """Extract embeddings and model outputs from a dataloader.

        Runs the model in evaluation mode to extract embeddings, outputs,
        module outputs, labels, and indices from the given dataloader.
        Results are concatenated and stored in the provided dictionary.

        Args:
            dataloader: DataLoader containing the dataset to process.
            dict_to_modify: Dictionary to store extracted data. Expected keys
                may include 'embedds', 'outs', 'module_out', 'labels', and/or
                'idxs' depending on desired outputs.
            embedds2cpu: If True, moves embeddings to CPU before concatenating.
                Defaults to False.

        Returns:
            None. The dict_to_modify dictionary is modified in place.

        Raises:
            AttributeError: If 'module_out' is requested but no additional
                module is specified in the model.
        """
                    
        self.model.eval()

        # again no gradients needed
        with torch.inference_mode():
            
            for data in dataloader:
                if len(data) > 3: idxs, images, labels, _ = data # in case on TiDAL
                else: idxs, images, labels = data  
                
                images = images.to(self.device)
                
                if 'embedds' in dict_to_modify:
                    embed = self.model(images, mode='embedds')
                    dict_to_modify["embedds"] = torch.cat((dict_to_modify["embedds"], embed.cpu() if embedds2cpu else embed), dim=0)

                if 'outs' in dict_to_modify:
                    dict_to_modify["outs"] = torch.cat((dict_to_modify["outs"], self.model(images, mode='outs').cpu()), dim=0)

                # could be both LL (1 output) and GTG (3 outputs)
                if 'module_out' in dict_to_modify:
                    if self.model.added_module != None:
                        dict_to_modify["module_out"] = torch.cat((
                            dict_to_modify["module_out"], 
                            self.model(images, mode='module_out',
                                        labels=labels.to(self.device) if self.model.only_module_name == 'GTGModule'
                                        else None, iteration=self.iter).cpu().squeeze()
                        ), dim=0)    
                    else:
                        raise AttributeError("Can't get the module_out if there is no additional module specified")    

                if 'labels' in dict_to_modify: dict_to_modify["labels"] = torch.cat((dict_to_modify["labels"], labels), dim=0)
                if 'idxs' in dict_to_modify: dict_to_modify["idxs"] = torch.cat((dict_to_modify["idxs"], idxs), dim=0)

    
    
    def save_tsne(
            self, 
            idxs_new_labels: List[int], 
            d_labels: Dict[str, int], 
            al_iter: int, 
            gtg_result_prediction = None
        ) -> None:
        """Saves a t-SNE visualization of the embeddings for the current active learning iteration.
        This method generates a t-SNE plot showing the labelled, unlabelled, and newly labelled
        observations for visualization purposes during active learning. It loads the best model
        checkpoint, extracts embeddings for both labelled and unlabelled training data, and
        creates a visualization using the provided parameters.
        Args:
            idxs_new_labels (List[int]): Indices of the newly labelled samples in this iteration.
            d_labels (Dict[str, int]): Dictionary mapping class names to their corresponding integer labels.
            al_iter (int): The current active learning iteration number.
            gtg_result_prediction (optional): Predictions from the GTG model. Defaults to None.
        Returns:
            None: This method saves the t-SNE plot to disk and does not return any value.
        Note:
            - This method loads the best checkpoint before extracting embeddings.
            - It requires the dataset to have 'unlab_train_ds' attribute and 'labelled_indices'.
            - The plot is saved to the experiment path specified in the configuration.
        """
        # plot the tsne graph for each iteration
        
        logger.info(' => Saving the TSNE embeddings plot with labelled, unlabelled and new labelled observations')
        
        dl_dict = dict(batch_size=self.batch_size, shuffle=False)
        unlab_train_dl = DataLoader(Subset(self.dataset.unlab_train_ds, self.rand_unlab_sample), **dl_dict)
        lab_train_dl = DataLoader(Subset(self.dataset.unlab_train_ds, self.labelled_indices), **dl_dict)
        
        lab_embedds_dict = {
            'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=torch.device('cpu')),
            'labels': torch.empty(0, dtype=torch.int8, device=torch.device('cpu'))
        }
        unlab_embedds_dict = {
            'embedds': torch.empty((0, self.model.backbone.get_embedding_dim()), dtype=torch.float32, device=torch.device('cpu')),
            'labels': torch.empty(0, dtype=torch.int8, device=torch.device('cpu'))
        }
        self.load_best_checkpoint()
        logger.info(' Getting the embeddings...')
        self.get_embeddings(lab_train_dl, lab_embedds_dict, embedds2cpu=True)
        self.get_embeddings(unlab_train_dl, unlab_embedds_dict, embedds2cpu=True)
        

        logger.info(' Plotting...')
        plot_new_labelled_tsne(
            lab_embedds_dict, unlab_embedds_dict,
            al_iter, self.strategy_name,
            self.ct_p["dataset_name"], idxs_new_labels, self.dataset.classes, 
            self.ct_p["exp_path"], self.ct_p["trial"], d_labels,
            gtg_result_prediction
        )
        
        logger.info(' DONE\n')
        
        
        
    def train_evaluate_save(
            self, 
            lab_obs: int, 
            iter: int,
            results_format: Dict[str, Dict[str, List[float]]]
        ) -> Dict[str, Any]:
        """
        Train, evaluate, and save results for the current iteration.
        This method executes the training process, evaluates the model on test data,
        and saves the results to CSV and curve files for further analysis.
        Args:
            lab_obs (int): Number of labeled observations used in this iteration.
            iter (int): Current iteration number in the active learning loop.
            results_format (Dict[str, Dict[str, List[float]]]): Dictionary containing
                the structure for storing results, with keys "train" and "test" mapping
                to metric names and their value lists.
        Returns:
            Dict[str, Any]: Dictionary containing training results for the current
                iteration, keyed by metric names.
        Raises:
            Exception: Re-raises any exception from the training process in distributed
                or non-distributed mode.
        Note:
            - If self.world_size > 1, runs distributed training using DDP (Distributed
                Data Parallel) with multiprocessing spawning.
            - For single GPU training, calls train() directly.
            - Results are saved to CSV file named '{task}_results.csv' in the
                experiment path.
            - Training and validation curves are saved using save_train_val_curves().
        """
        
        test_res_keys = list(results_format["test"].keys())
        train_res_keys = list(results_format["train"].keys())
        
        params = { 
            'ct_p': self.ct_p, 't_p': self.t_p, 'strategy_name': self.strategy_name, 
            'iter': iter, 'labelled_indices': self.labelled_indices, 'rand_unlab_sample': self.rand_unlab_sample,
            'module_name_only': self.model.only_module_name
        }
                
        #if self.model.only_module_name == 'GTGModule': params["perc_labelled_batch"] = self.perc_labelled_batch
        if self.model.only_module_name == 'GTGModule': params["batch_size_gtg_online"] = self.batch_size_gtg_online # type: ignore
                
        # if we are using multiple gpus
        if self.world_size > 1:
            
            os.environ["MASTER_ADDR"] = self.master_address
            os.environ["MASTER_PORT"] = self.master_port
            
            # Pipe for the itra-process communication of the results
            parent_conn, child_conn = mp.Pipe()
            
            logger.info(' => RUNNING DISTRIBUTED TRAINING')
            
            # spawn the process
            mp.spawn(fn=train_ddp, args=(self.world_size, params, child_conn, ), nprocs=self.world_size, join=True) # type: ignore
            # obtain the results
            while parent_conn.poll(): train_recv, test_recv = parent_conn.recv()
                            
        else:
            logger.info(' => RUNNING TRAINING')
            # add the already created labeeld train dataloader
                
            train_recv, test_recv = train(params)
            
        
        logger.info(' DONE\n')
        
        iter_train_results, iter_test_results = {}, {}
        for idx, metrics in enumerate(train_recv):  # type: ignore
            iter_train_results[train_res_keys[idx]] = metrics
            
        for idx, metrics in enumerate(test_recv):  # type: ignore
            iter_test_results[test_res_keys[idx]] = metrics
            results_format["test"][test_res_keys[idx]].append(metrics)
        
        logger.info(f'TESTING RESULTS -> {iter_test_results}')
        
        # save the results in the csv file for further analysis and plotting of cumulative results after the end of all iterations
        write_csv(
            filename = 'results.csv',
            exp_path = self.ct_p["exp_path"],
            dataset_name = self.ct_p["dataset_name"],
            head = ['method', 'iter', 'lab_obs'] + test_res_keys,
            values = [self.strategy_name, self.ct_p["trial"], lab_obs] + list(iter_test_results.values())
        )
        
        # save the train and val curves for the current iteration
        save_train_val_curves(
            list_dict_keys=list(self.t_p["results_dict"]["train"].keys()), 
            results_info=iter_train_results,
            strategy_name=self.strategy_name,
            exp_path=self.ct_p["exp_path"],
            dataset_name=self.ct_p["dataset_name"],
            al_iter=iter,
            cicle_iter=self.ct_p["trial"]
        )

        return iter_train_results
        
        
    
    def update_sets(self, overall_topk: List[int]) -> None:
        """
        Update the labelled and unlabelled indices after an active learning iteration.

        This method updates the internal datasets by moving samples from the unlabelled
        pool to the labelled pool based on the query strategy results. It also handles
        the temporary unlabelled pool if configured, saves newly labelled images for
        visual analysis, and performs sanity checks to ensure no index overlap.

        Args:
            overall_topk: List of indices corresponding to the samples selected by the
                query strategy to be added to the labelled set.

        Returns:
            None. Updates self.labelled_indices, self.unlabelled_indices, and
                optionally self.temp_unlab_pool in place.

        Raises:
            Exception: If there is a non-empty intersection between the labelled,
                unlabelled, or temp_unlab_pool index sets (indicating data leakage).

        Note:
            - If temp_unlab_pool is enabled, the randomly sampled unlabelled indices
              are temporarily removed and then re-added after the iteration.
            - The method logs the new lengths of all index lists for debugging.
        """
            
        # save the new labelled images to further visual analysis
        self.save_labelled_images(overall_topk)
        
        # Update the labeeld and unlabelled training set
        logger.info(' => Modifing the labelled and Unlabelled Indices Lists')
        
        sample_len = len(self.rand_unlab_sample)
                
        # if we are using the temporary unlabelled pool we have to update the unlabelled indices list and temp pool
        if self.ct_p["temp_unlab_pool"]:
            # the sample have been already removed from the labelled set
            for idx in overall_topk: self.rand_unlab_sample.remove(idx) # - 1000 = 9000
            self.temp_unlab_pool.extend(self.rand_unlab_sample) # + 9000
        else:
            for idx in overall_topk: self.unlabelled_indices.remove(idx) # - 1000 = 9000
            
        # extend with the overall_topk
        self.labelled_indices.extend(overall_topk) # + 1000
        
        
        # sanity check
        if len(list(set(self.unlabelled_indices) & set(self.labelled_indices))) == 0 and \
                len(list(set(self.temp_unlab_pool) & set(self.labelled_indices))) == 0 and \
                len(list(set(self.unlabelled_indices) & set(self.temp_unlab_pool))) == 0:
            logger.info(' Intersection between indices lists are EMPTY')
        else: 
            logger.exception('NON EMPTY INDICES INTERSECTION')
            raise Exception('NON EMPTY INDICES INTERSECTION')
        
        
        if self.ct_p["temp_unlab_pool"] and len(self.unlabelled_indices) < sample_len and len(self.temp_unlab_pool) > 0: 

            # reinsert all teh observations from the pool inside the original unlabelled pool
            self.unlabelled_indices.extend(self.temp_unlab_pool)
            self.temp_unlab_pool.clear() # empty the temp unlabelled pool list


        
        logger.info(f' New labelled_indices lenght: {len(self.labelled_indices)} - new unlabelled_indices lenght: {len(self.unlabelled_indices)}' + \
            f' - new temp_unlab_pool lenght: {len(self.temp_unlab_pool)}' if self.ct_p["temp_unlab_pool"] else '')
        
        logger.info(' DONE\n')

        
        
    def run(self) -> Dict[str, List[float]]:
        """
        Run the active learning loop for the current strategy.

        Executes the complete active learning process across multiple iterations,
        including training, evaluation, query strategy selection, and result saving.

        Returns:
            Dict[str, List[float]]: A dictionary containing test results for each
                metric, where keys are metric names and values are lists of results
                across all iterations.

        Raises:
            Exception: If any critical error occurs during training, evaluation,
                or query strategy execution.

        Note:
            - The loop runs for 'al_iters' iterations as defined in al_params.
            - Each iteration trains the model, queries new labels, and updates
              the labelled/unlabelled sets.
            - Results are saved to CSV and visualization plots are generated.
        """
            
        results_format = copy.deepcopy(self.t_p["results_dict"])
                
        # start of the loop
        while self.iter <= al_params["al_iters"]:
            
            #self.iter += 1
            
            logger.info(f'----------------------- ITERATION {self.iter} / {al_params["al_iters"]} -----------------------\n')
            
            logger.info(f' => Getting the sampled unalbeled indices for the current iteration...')
            self.get_rand_unlab_sample()
            logger.info(' DONE\n')
            
            self.train_results[self.iter] = self.train_evaluate_save(
                lab_obs=al_params["init_lab_obs"] + ((self.iter - 1) * al_params["n_top_k_obs"]),
                iter=self.iter, 
                results_format=results_format
            )
            
            logger.info(' START QUERY PROCESS\n')
            
            # run method query strategy
            idxs_new_labels, topk_idx_obs = self.query( # type: ignore
                Subset(self.dataset.unlab_train_ds, self.rand_unlab_sample), al_params["n_top_k_obs"]
            )
            
            d_labels = count_class_observation(
                classes=self.dataset.classes, 
                dataset=self.dataset.train_ds, 
                topk_idx_obs=topk_idx_obs
            )
            logger.info(f' Number of observations per class added to the labelled set:\n {d_labels}\n')
            
            # Saving the tsne embeddings plot
            if 'GTG_off' in self.method_name:
                # if we are performing GTG Offline plot also the GTG predictions in the TSNE plot 
                self.save_tsne(
                    idxs_new_labels=idxs_new_labels, 
                    d_labels=d_labels, 
                    al_iter=self.iter, 
                    gtg_result_prediction=self.gtg_result_prediction # type: ignore
                )
            
            else: self.save_tsne(idxs_new_labels=idxs_new_labels, d_labels=d_labels, al_iter=self.iter)

            # modify the datasets and dataloader and plot the tsne
            self.update_sets(overall_topk=topk_idx_obs)
            
            self.iter += 1

            #if self.iter <= 10: self.count_classes[self.iter * al_params["n_top_k_obs"]] = list(d_labels.values())
            self.count_classes[self.iter * al_params["n_top_k_obs"]] = list(d_labels.values())

        
            write_csv(
                filename = 'count_classes.csv',
                exp_path = self.ct_p["exp_path"],
                dataset_name = self.ct_p["dataset_name"],
                head = ['method', 'iter', 'lab_obs'] + self.dataset.classes,
                values = [self.strategy_name, self.ct_p["trial"], self.iter * al_params["n_top_k_obs"]] + self.count_classes[self.iter * al_params["n_top_k_obs"]]
            )
                
        # plotting the number of classes in the train dataset for each trial
        if al_params["al_iters"] > 1:
            if self.dataset.n_classes <= 10: plot_classes_count_iterations(
                count_classes=self.count_classes, 
                class_labels=self.dataset.classes,
                strategy_name=self.strategy_name, 
                exp_path=self.ct_p["exp_path"], 
                dataset_name=self.ct_p["dataset_name"],
                cicle_iter=self.ct_p["trial"]
            )
            plot_entropy_iterations_classes(
                count_classes=self.count_classes, 
                strategy_name=self.strategy_name, 
                exp_path=self.ct_p["exp_path"], 
                dataset_name=self.ct_p["dataset_name"], 
                cicle_iter=self.ct_p["trial"],
            )
        
        # plotting the cumulative train results
        plot_cumulative_train_results(
            list_dict_keys=list(self.t_p["results_dict"]["train"].keys()), 
            cum_train_results=self.train_results, 
            strategy_name=self.strategy_name, 
            epochs=len(self.train_results[1]["train_pred_loss"]),
            exp_path=self.ct_p["exp_path"], 
            dataset_name=self.ct_p["dataset_name"], 
            cicle_iter=self.ct_p["trial"]
        )
    
        return results_format["test"]