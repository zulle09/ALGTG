
import torch
from torch.backends import cudnn
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.init as init

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import numpy as np
import csv
import os
import copy
import yaml
import random
from matplotlib.typing import ColorType
from typing import List, Dict, Any, Tuple
import argparse
import torch.distributed as dist



import logging
logger = logging.getLogger(__name__)



def accuracy_score(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute the accuracy score given the model outputs and the true labels.
    
    Args:
        outputs (torch.Tensor): The model outputs (logits or probabilities).
        labels (torch.Tensor): The true labels.
    
    Returns:
        float: The accuracy score.
    """
    outputs_class = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
    return (outputs_class == labels).sum().item()/len(outputs)



def entropy(tensor: torch.Tensor, dim=1) -> torch.Tensor:
    """
    Compute the entropy of a tensor along a specified dimension.
    Args:
        tensor (torch.Tensor): The input tensor.
        dim (int): The dimension along which to compute the entropy.
    Returns:
        torch.Tensor: The computed entropy.
    """
    x = torch.clamp(tensor, min=1e-8, max=1.)
    return -torch.sum(x * torch.log2(x), dim=dim)



def plot_trail_acc(
        dataset_name: str, 
        trial: int, 
        methods_results: Dict[str, Dict[str, List[float]]], 
        n_lab_obs: List[int], 
        exp_path: str, 
        key: str, 
        save_plot=True, 
        plot_png_name=None
    ) -> None:
    """
    Plot trial accuracy results for different methods.
    Generates a line plot comparing the performance of multiple methods
    across different numbers of labelled observations.
    
    Args:
        dataset_name (str): Name of the dataset being evaluated.
        trial (int): Trial number for the experiment.
        methods_results (Dict[str, Dict[str, List[float]]]): Dictionary mapping
            method names to their results. Each method's results is a dictionary
            with keys corresponding to metric names (e.g., 'test_map', 'accuracy')
            and values being lists of metric values for each number of labelled
            observations.
        n_lab_obs (List[int]): List of numbers of labelled observations used
            as x-axis values.
        exp_path (str): Path to the results directory for saving plots.
        key (str): Metric key to plot. If 'test_map', plots mAP; otherwise,
            plots Accuracy.
        save_plot (bool, optional): Whether to save the plot to a file.
            Defaults to True.
        plot_png_name (str, optional): Filename for saving the plot.
            Required if save_plot is True. Defaults to None.
    Returns:
        None
    
    Raises:
        ValueError: If save_plot is True but plot_png_name is None.
    """
    
    if key == 'test_map': measure = 'mAP'
    else: measure = 'Accuracy'
    
    plt.figure(figsize = (14,10))
    
    shapes = ['o', 's', '^', 'D', 'v', 'o', 's', '^', 'D', 'v', 'o', 's', '^', 'D', 'v', 'o', 's', '^', 'D', 'v']
    palette = get_palette(len(methods_results.items()))
    lines_handles = []
    
    # Plot each method
    for idx, (method_str, values) in enumerate(methods_results.items()):
        plt.plot(
            n_lab_obs, 
            values[key], 
            label = f'{method_str}',
            linestyle = 'dashed' if method_str.split('_')[1] in 'Random' else 'solid', 
            color=palette[idx]
        )
        plt.scatter(n_lab_obs, values[key], marker=shapes[idx], color=palette[idx], zorder=5)
        lines_handles.append(mlines.Line2D([], [], color=palette[idx], marker=shapes[idx], markersize=5, label=method_str))
  
    plt.title(f'{dataset_name} - Trial {trial} - {measure} labelled Obs', fontsize = 30)
    plt.xlabel('# labelled Obs', fontsize = 15)
    plt.ylabel(measure, fontsize = 15)
    plt.grid(True)
    plt.legend(handles=lines_handles)
    
    if save_plot: plt.savefig(f'results/{exp_path}/{plot_png_name}')
    else: plt.show()
    
    
    
def save_train_val_curves(
        list_dict_keys: List[str], 
        results_info: Dict[str, Any], 
        strategy_name: str, 
        exp_path: str, 
        dataset_name: str, 
        al_iter: int, 
        cicle_iter: int
    ) -> None:
    """
    Generate and save training and validation curves as a 2x2 subplot figure.
    
    Args:
        list_dict_keys (List[str]): List of dictionary keys corresponding to the
            metrics to be plotted (e.g., 'train_loss', 'val_loss', 'train_acc', 'val_acc').
        results_info (Dict[str, Any]): Dictionary containing the metric values for each
            epoch, where keys match those in list_dict_keys.
        strategy_name (str): Name of the active learning strategy used in the experiment.
        exp_path (str): Path to the experiment results directory.
        dataset_name (str): Name of the dataset being used.
        al_iter (int): Current active learning iteration number.
        cicle_iter (int): Current cycle iteration number.
    
    Returns:
        None: This function saves the plot to a file and does not return any value.
    
    Example:
        >>> results = {'train_loss': [0.5, 0.4, 0.3], 'val_loss': [0.6, 0.5, 0.4],
        ...            'train_acc': [0.7, 0.8, 0.9], 'val_acc': [0.65, 0.75, 0.85]}
        >>> keys = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
        >>> save_train_val_curves(keys, results, 'random_sampling', 'exp1', 'cifar10', 1, 1)
    """

    epochs = range(1, len(results_info[list_dict_keys[0]]) + 1)

    _, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (28,18))
        
    data = zip([(0,0), (0,1), (1,0), (1,1)], list_dict_keys)
    
    # plot each train/val curve in its respective subplot
    for (pos1, pos2), train_mes in data:
        ax[pos1][pos2].plot(epochs, results_info[train_mes], label = train_mes)            
        ax[pos1][pos2].set_title(f'{train_mes} - Epochs', fontsize = 15)
        ax[pos1][pos2].set_xlabel('Epochs', fontsize = 10)
        ax[pos1][pos2].set_ylabel(train_mes, fontsize = 10)
        ax[pos1][pos2].grid()
        ax[pos1][pos2].legend()
        
    # save the plot in the respective strategy folder
    plt.suptitle(f'Iteration: {al_iter} - {strategy_name}', fontsize = 30)
    plt.savefig(f'results/{exp_path}/{dataset_name}/{cicle_iter}/{strategy_name}/train_val_plots/{al_iter}.png')



def plot_cumulative_train_results(
        list_dict_keys: List[str], 
        cum_train_results: Dict[int, Any], 
        strategy_name: str,
        epochs: int,
        exp_path: str, 
        dataset_name: str, 
        cicle_iter: int
    ) -> None:

    """
    Plot cumulative training results in a 2x2 subplot figure.
    Generates a figure with four subplots displaying training and validation
    metrics over epochs for different iterations. Each subplot corresponds to
    a specific metric specified in list_dict_keys.
    
    Args:
        list_dict_keys (List[str]): List of metric keys to plot from the
            results dictionary (e.g., ['train_loss', 'val_loss', 'train_acc', 'val_acc']).
        cum_train_results (Dict[int, Any]): Dictionary mapping iteration numbers
            to their corresponding results dictionaries containing metric values
            for each epoch.
        strategy_name (str): Name of the strategy being evaluated (used for
            plot title and file naming).
        epochs (int): Total number of training epochs.
        exp_path (str): Base path for experiment results.
        dataset_name (str): Name of the dataset being used.
        cicle_iter (int): Current cycle/iteration number for the experiment.
    
    Returns:
        None: The function saves the plot to a file and does not return any value.
    
    Raises:
        KeyError: If any key in list_dict_keys is not present in the results dictionaries.
        ValueError: If list_dict_keys does not contain exactly 4 elements.
    """
    _, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (28,18))
        
    data = zip([(0,0), (0,1), (1,0), (1,1)], list_dict_keys)
    x = range(1, epochs + 1)
    
    palette = get_palette(len(cum_train_results.items()))
    
    # plot each train/val curve in its respective subplot
    for (pos1, pos2), train_mes in data:
        for idx, (iter, results_info) in enumerate(cum_train_results.items()):
            ax[pos1][pos2].plot(x, results_info[train_mes], label = f'{train_mes}_{iter}', color=palette[idx])
                        
        ax[pos1][pos2].set_title(f'{train_mes} - Epochs', fontsize = 15)
        ax[pos1][pos2].set_xlabel('Epochs', fontsize = 10)
        ax[pos1][pos2].set_ylabel(train_mes, fontsize = 10)
        ax[pos1][pos2].grid()
        ax[pos1][pos2].legend()

    # save the plot in the respective strategy folder
    plt.suptitle(f'Cumulative Train Results - {strategy_name}', fontsize = 30)
    plt.savefig(f'results/{exp_path}/{dataset_name}/{cicle_iter}/{strategy_name}/train_val_plots/cumulative_train_results.png')



def plot_entropy_iterations_classes(
        
        count_classes: Dict[int, List[int]], 
        strategy_name: str, 
        exp_path: str,
        dataset_name: str, 
        cicle_iter: int
    ) -> None:
    """
    Plot the entropy of class counts across iterations.
    Calculates the entropy for each class distribution in the provided
    dictionary of class counts, plots the entropy values, and saves the
    resulting figure to a specified path.
    
    Args:
        count_classes: A dictionary mapping class indices to lists of 
            observation counts for each iteration.
        strategy_name: The name of the strategy used (e.g., 'random', 
            'uncertainty'). Used for saving the plot.
        exp_path: The experiment path relative to the 'results/' directory.
        dataset_name: The name of the dataset being analyzed.
        cicle_iter: The current iteration number for the cycle.
    
    Returns:
        None. The function saves the plot directly to disk.
    
    Raises:
        ValueError: If count_classes is empty or contains empty lists.
    """

    entropy_val_to_plots = []
    for count in count_classes.values():
        count = torch.tensor(count).float()
        count = count / count.sum()
        entropy_val_to_plots.append(entropy(count, dim=0).item())
    
    plt.figure(figsize = (14,10))
    plt.plot(entropy_val_to_plots)
    plt.xlabel('Labelled Observations', fontsize=15)
    plt.ylabel('Entropy', fontsize=15)
    plt.title('Class Counts Entropy through Iteration', fontsize=30)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/{exp_path}/{dataset_name}/{cicle_iter}/{strategy_name}/entropy_class_iterations.png')



def plot_classes_count_iterations(
        
        count_classes: Dict[int, List[int]], 
        class_labels: List[str], 
        strategy_name: str, 
        exp_path: str, 
        dataset_name: str, 
        cicle_iter: int
    ) -> None:
    """
    Plot the class observations count across iterations.
    Generates a line plot showing the number of observations for each class
    across different iterations of a labeling process.

    Args:
        count_classes: A dictionary mapping iteration numbers to lists of class
            observation counts. Keys represent iteration steps, and values are
            lists where each index corresponds to a class ID.
        class_labels: A list of string labels for each class ID.
        strategy_name: The name of the active learning strategy used (e.g., 'random', 'uncertainty').
        exp_path: The experiment directory path for organizing results.
        dataset_name: The name of the dataset being used.
        cicle_iter: The current iteration cycle number.

    Returns:
        None. The function saves the plot directly to a file.
    """

    plt.figure(figsize=(14, 10))
    iters = list(count_classes.keys())
    for label_id in range(len(class_labels)):
        plt.plot(
            iters, [count_classes[it][label_id] for it in iters], 
            label=class_labels[label_id]
        )

    plt.xlabel('Labelled Observations', fontsize=15)
    plt.ylabel('Number of Class Observations', fontsize=15)
    plt.title('Class Observations Count by Iteration', fontsize=30)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/{exp_path}/{dataset_name}/{cicle_iter}/{strategy_name}/classes_count_iterations.png')



def write_csv(
        filename: str, 
        exp_path: str, 
        dataset_name: str, 
        head: List[str], 
        values: List[Any]
    ) -> None:
    """
    Write data to a CSV file in the results directory.
    Creates a CSV file at the specified path if it doesn't exist, writes the
    header row on first creation, and appends the provided values as a new row.
    
    Args:
        filename: The name of the CSV file to write.
        exp_path: The experiment path subdirectory within results.
        dataset_name: The dataset name subdirectory within the experiment path.
        head: List of column headers for the CSV file.
        values: List of values to write as a new row in the CSV file.
    
    Returns:
        None.
    
    Raises:
        OSError: If there is an issue creating or writing to the file.
    """
    res_path = os.path.join('results', exp_path, dataset_name, filename)
    
    if (not os.path.exists(res_path)):
        
        with open(res_path, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(head)
            f.close()
    
    with open(res_path, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(values)
        f.close()
            


def create_directory(dir: str) -> None: 
    """Create a directory at the specified path.

    Creates the directory and any necessary parent directories if they don't exist.
    If the directory already exists, no error is raised.

    Args:
        dir: The path of the directory to create.

    Returns:
        None.

    Raises:
        OSError: If the directory cannot be created.
    """
    os.makedirs(dir, exist_ok=True)


def create_exp_path(exp_path: str, dataset_name: str, iter: str) -> None:
    """
    Create the experiment directory path.

    Creates a directory structure for storing experiment results under
    'results/<exp_path>/<dataset_name>/<iter>'.

    Args:
        exp_path: The experiment path or name identifier.
        dataset_name: The name of the dataset used in the experiment.
        iter: The iteration number or identifier.

    Returns:
        None
    """
    create_directory(os.path.join('results', exp_path, dataset_name, iter))
    


def create_method_res_dir(path: str) -> None:
    """Create subdirectories for storing method results.

    Creates three subdirectories within the specified path:
    - 'train_val_plots': for storing training and validation plots
    - 'tsne_plots': for storing t-SNE visualization plots
    - 'new_labelled_images': for storing newly labelled images

    Args:
        path: The base directory path where the subdirectories will be created.

    Returns:
        None

    Raises:
        OSError: If directory creation fails.
    """
    for dir in ['train_val_plots', 'tsne_plots', 'new_labelled_images']:
        create_directory(os.path.join(path, dir))
    


def create_class_dir(base_path: str, iter: int, classes: List[str]) -> None:
    for cls in classes: create_directory(f'{base_path}/new_labelled_images/{iter}/{cls}')


    
def get_palette(n_classes: int) -> List[ColorType]:
    """Generate a color palette based on the number of classes.

    Selects an appropriate matplotlib color palette depending on the number
    of classes required. The palette is shuffled to provide randomized color
    assignments.

    Args:
        n_classes: The number of distinct classes for which colors are needed.

    Returns:
        A list of colors (ColorType) to be used for visualizing different classes.
        The palette is selected from:
            - TABLEAU_COLORS if n_classes <= 10
            - CSS4_COLORS if 10 < n_classes <= 100
            - XKCD_COLORS if n_classes > 100

    """

    if n_classes <= 10: palette = list(mcolors.TABLEAU_COLORS.values())
    elif n_classes <= 100: palette = list(mcolors.CSS4_COLORS.values())
    else: palette = list(mcolors.XKCD_COLORS.values())
    random.shuffle(palette)
    return palette
    
    

def plot_gtg_entropy_tensor(
        tensor: torch.Tensor,
        topk: List[int], 
        lab_unlabels: List[int], 
        classes: List[str], 
        path: str, 
        iter: int, 
        max_x: int, 
        dir: str
    ) -> None:
    """
    Plot entropy tensor for GTG (Guess The Gibbs) visualization.
    Generates two subplots visualizing entropy values over GTG iterations:
    1. First subplot shows entropy history/derivatives for all classes.
    2. Second subplot shows entropy for newly labelled samples vs unlabelled samples.
    
    Args:
        tensor: A torch.Tensor containing entropy values with shape 
            (num_samples, max_x) representing entropy over iterations.
        topk: List of indices indicating the top k newly labelled samples.
        lab_unlabels: List of integer labels corresponding to each sample 
            in the tensor.
        classes: List of class names as strings for legend labeling.
        path: Base directory path where plots will be saved.
        iter: Current iteration number used for filename and title.
        max_x: Maximum number of GTG iterations (x-axis range).
        dir: Subdirectory name determining plot type. If 'history', plots 
            entropy history; otherwise plots entropy derivatives.
    
    Returns:
        None. Saves the generated plot as a PNG file to 
        {path}/gtg_entropies_plots/{dir}/{iter}.png.
    
    Raises:
        FileNotFoundError: If the specified path cannot be created.
        ValueError: If tensor dimensions are incompatible with max_x.
    """
    
    create_directory(f'{path}/gtg_entropies_plots/{dir}')
    
    _, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    
    x = np.arange(max_x)
    array = tensor.cpu().numpy()
    
    new_labelled = array[topk]
    new_labelled_lab = [lab_unlabels[k] for k in topk]
    
    mask = np.ones(array.shape[0], dtype=bool)
    mask[topk] = False
    unlabelled = array[mask]

    title = 'Entropy History' if dir == 'history' else 'Entropy Derivatives'

    palette = get_palette(len(classes))
        
    pl_cls_1, pl2_cls_2 = set(), set()

    for i, lab in zip(range(len(array)), lab_unlabels):
        if lab not in pl_cls_1 and len(classes) <= 10 :
            axes[0].plot(x, array[i], linestyle="-", label=classes[lab], color=palette[lab])
            pl_cls_1.add(lab)
        else:
            axes[0].plot(x, array[i], linestyle="-", color=palette[lab])
    
    axes[0].set_title(f'{title} - Classes', fontsize = 25)
    axes[0].set_ylabel('Entropy', fontsize = 20)
    axes[0].set_xlabel('GTG Iterations', fontsize = 20)
    axes[0].grid()
    axes[0].legend()
    
    
    for array, color, style, label in [(unlabelled, 'lightblue', '--', 'unlabelled'),
                                       (new_labelled, None, '-', new_labelled_lab)]:
        
        labels_array = None
        if style == '-': labels_array = label
        
        for i in range(len(array)):
            
            if style == '-' and labels_array is not None:
                color = palette[labels_array[i]]
                label = classes[labels_array[i]]
                        
            if label not in pl2_cls_2 and len(classes) <= 10:
                axes[1].plot(x, array[i], linestyle=style, label=label, color=color)
                pl2_cls_2.add(label)
            else: axes[1].plot(x, array[i], linestyle=style, color=color)
        
    axes[1].set_ylabel('Entropy', fontsize = 20)
    axes[1].set_xlabel('GTG Iterations', fontsize = 20)
    axes[1].set_title(f'{title} - New_Lab / Unlab', fontsize = 25)
    axes[1].grid()
    axes[1].legend()
    
    plt.suptitle(f'Entropy History - Iteration {iter}', fontsize = 30)
    plt.savefig(f'{path}/gtg_entropies_plots/{dir}/{iter}.png')
    



def plot_res_std_mean(
        exp_path: str, 
        dataset_name: str
    ) -> None:
    """
    Plot mean and standard deviation of classification results with confidence intervals.
    Reads classification results from a CSV file, groups by method and labelled
    observations, calculates mean, standard deviation, and 95% confidence intervals,
    then generates a line plot with error bands.
   
    Args:
        exp_path: Path to the experiment results directory.
        dataset_name: Name of the dataset used in the experiment.
    
    Returns:
        None. Saves two files to the results directory:
            - mean_std_clf_results.csv: Grouped data with mean, std, count, and CI bounds
            - mean_std_results.png: Line plot showing test accuracy vs labelled observations
    
    Raises:
        FileNotFoundError: If the results CSV file does not exist at the expected path.
    """
    df = pd.read_csv(f'results/{exp_path}/{dataset_name}/clf_results.csv')

    df_grouped = (
        df[['method', 'lab_obs', 'test_accuracy']
        ].groupby(['method', 'lab_obs']).agg(['mean', 'std', 'count'])
    )
    df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
    
    df_grouped["ci"] = 1.96 * df_grouped["std"] / np.sqrt(df_grouped["count"])
    df_grouped["ci_lower"] = df_grouped["mean"] - df_grouped["ci"]
    df_grouped["ci_upper"] = df_grouped["mean"] + df_grouped["ci"]
    
    methods = df_grouped["method"].unique()
    shapes = ['o', 's', '^', 'D', 'v', 'o', 's', '^', 'D', 'v', 'o', 's', '^', 'D', 'v', 'o', 's', '^', 'D', 'v']

    df_grouped = df_grouped.drop(columns=["ci"])
    df_grouped.to_csv(f'results/{exp_path}/{dataset_name}/mean_std_clf_results.csv', index=False)

    lines_handles = []
    palette = get_palette(len(methods))

    # Plot each method
    plt.figure(figsize=(14, 10))
    for idx, method in enumerate(methods):
        method_data = df_grouped[df_grouped["method"] == method]
        plt.plot(method_data["lab_obs"], method_data["mean"], label=method, 
                 linestyle = 'dashed' if method.split('_')[1] in 'Random' else 'solid', color=palette[idx])
        plt.fill_between(method_data["lab_obs"], method_data["ci_lower"], method_data["ci_upper"], alpha=0.3, color=palette[idx])
        plt.scatter(method_data["lab_obs"], method_data["mean"], marker=shapes[idx], color=palette[idx], zorder=5)
        lines_handles.append(mlines.Line2D([], [], color=palette[idx], marker=shapes[idx], markersize=5, label=method))
        

    plt.xlabel('labelled Observations', fontsize = 15)
    plt.ylabel('Test Accuracy', fontsize = 15)
    plt.title(f'{dataset_name} results', fontsize = 30)
    plt.legend(handles=lines_handles)
    plt.grid(True)
    
    plt.savefig(f'results/{exp_path}/{dataset_name}/mean_std_results.png') 
    


def set_seeds(seed: int = 10001) -> None:
    """
    Set random seeds for reproducibility across various libraries and frameworks.
    This function ensures that the results of experiments are reproducible by
    setting the same seed for Python's built-in random module, NumPy, and PyTorch
    (both CPU and CUDA). It also configures PyTorch's cuDNN backend to be
    deterministic and disables benchmarking to further enhance reproducibility.
    
    Args:
        seed: An integer value to be used as the random seed for all libraries.
    
    Returns:
        None. This function does not return any value but sets the seeds and configurations in place for reproducibility.
    """
    # setting seed and deterministic behaviour of pytorch for reproducibility
    # https://discuss.pytorch.org/t/determinism-in-pytorch-across-multiple-files/156269
    #os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    


def plot_tsne_A(
        A: Tuple[torch.Tensor, torch.Tensor], 
        labels: Tuple[torch.Tensor, torch.Tensor],
        classes: List[str], 
        exp_path: str, 
        ds_name: str, 
        samp_iter: int, 
        method: str,
        affinity: str, 
        strategy: str, 
        iter: int
    ) -> None:
    """
    Generate t-SNE visualization plots for affinity matrices.
    This function applies t-SNE dimensionality reduction to the input affinity
    matrices and creates a 2x2 subplot figure showing both labelled and unlabelled
    data points, as well as class distributions.
    
    Args:
        A: Tuple of two torch.Tensor containing the original and sparsified
            affinity matrices.
        labels: Tuple of two torch.Tensor containing labels for labelled and
            unlabelled data points.
        classes: List of strings representing the class names.
        exp_path: String path to the experiment directory.
        ds_name: String name of the dataset.
        samp_iter: Integer representing the sample iteration.
        method: String specifying the method used (e.g., 'knn', 'ss').
        affinity: String specifying the affinity type (e.g., 'cosine', 'euclidean').
        strategy: String specifying the sparsification strategy used.
        iter: Integer representing the current iteration number.
    
    Returns:
        None. The function saves the generated plot to a PNG file in the
        specified results directory.
    
    Raises:
        AttributeError: If the input tensors do not have numpy() method.
        ValueError: If labels tuple length doesn't match expected format.
    """
    
    A_1, A_2 = A
    
    len_lab = len(labels[0])
    
    tsne_A1 = TSNE().fit_transform(A_1.numpy())
    tsne_A2 = TSNE().fit_transform(A_2.numpy())
    
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(23, 18))
        
    for idx, (tsne, name) in enumerate(zip([tsne_A1, tsne_A2], ['Original', f'{strategy} Sparsification'])):
        
        tsne_lab, tsne_unlab = tsne[:len_lab,:], tsne[len_lab:,:]
        x_lab, y_lab = tsne_lab[:,0], tsne_lab[:,1]
        x_unlab, y_unlab = tsne_unlab[:,0], tsne_unlab[:,1]
        x, y = np.hstack((x_lab, x_unlab)), np.hstack((y_lab, y_unlab))
        label = np.hstack(labels)
        
        sns.scatterplot(x=x_unlab, y=y_unlab, label='unlabelled', color='blue', s=17, ax=axes[idx][0])
        sns.scatterplot(x=x_lab, y=y_lab, label='labelled', color='orange', s=17, ax=axes[idx][0])
        axes[idx][0].set_title(f'{name} Affinity Matrix', fontsize = 20)
        axes[idx][0].legend()
        
        sns_p = sns.scatterplot(x=x, y=y, hue=[classes[l] for l in label], s=17, ax=axes[idx][1])
        axes[idx][1].set_title(f'{name} Affnity Matrix Classes', fontsize = 20)
        axes[idx][1].legend()
        if len(classes) > 10: sns_p.legend([],[], frameon=False)

    plt.suptitle(f'Affinity Matrix TSNE Plots - Iteration {iter}', fontsize=30)
    
    create_directory(f'results/{exp_path}/{ds_name}/{samp_iter}/{method}/tsne_plots/{affinity}_matrix')
    plt.savefig(f'results/{exp_path}/{ds_name}/{samp_iter}/{method}/tsne_plots/{affinity}_matrix/{iter}.png')
    
    
    
    
    
def plot_new_labelled_tsne(
        lab: Dict[str, torch.Tensor], 
        unlab: Dict[str, torch.Tensor],
        iter: int, 
        method: str, 
        ds_name: str, 
        idxs_new_labels: List[int], 
        classes: List[str],
        exp_path: str, 
        samp_iter: int, 
        d_labels: Dict[str, int],
        gtg_result_prediction=None
    ) -> None:
    """
    Generate t-SNE visualization plots for labelled, unlabelled, and new labelled data.
    This function creates a two-panel figure visualizing the t-SNE embeddings of
    labelled and unlabelled data points. The first panel shows the distribution
    of labelled, unlabelled, and newly labelled observations, while the second
    panel displays the same data colored by class labels.
    
    Args:
        lab (Dict[str, torch.Tensor]): Dictionary containing labelled data with
            keys 'embedds' (tensor of embeddings) and 'labels' (tensor of labels).
        unlab (Dict[str, torch.Tensor]): Dictionary containing unlabelled data with
            keys 'embedds' (tensor of embeddings) and 'labels' (tensor of labels).
        iter (int): Current iteration number for titling and saving the plot.
        method (str): The method used for labelling (e.g., 'random', 'uncertainty').
        ds_name (str): Name of the dataset being visualized.
        idxs_new_labels (List[int]): Indices of the newly labelled observations
            within the unlabelled data.
        classes (List[str]): List of class names corresponding to label indices.
        exp_path (str): Path to the experiment directory for saving results.
        samp_iter (int): Sample iteration number for organizing output files.
        d_labels (Dict[str, int]): Dictionary mapping labels to their counts,
            displayed in the figure text when there are 10 or fewer classes.
        gtg_result_prediction (np.ndarray, optional): Array of predictions for
            unlabelled data. If provided, the first panel distinguishes between
            true and false predictions for both unlabelled and new-labelled
            observations. Defaults to None.
    
    Returns:
        None: The function saves the generated plot to a file and does not
            return any value.
    
    Raises:
        ValueError: If the embeddings in 'lab' or 'unlab' are empty.
        TypeError: If input arguments are of incorrect types.
    """
    
    tsne = TSNE().fit_transform(np.vstack((lab["embedds"].numpy(), unlab["embedds"].numpy())))
    tsne_lab, tsne_unlab = tsne[:len(lab["embedds"]), :], tsne[len(lab["embedds"]):, :]
    
    x_lab, y_lab = tsne_lab[:, 0], tsne_lab[:, 1]
    x_unlab, y_unlab = tsne_unlab[:, 0], tsne_unlab[:, 1]
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 14))
    
    if isinstance(gtg_result_prediction, np.ndarray):
        
        idxs_unlab_again = [id for id in range(len(gtg_result_prediction)) if id not in idxs_new_labels]
        
        true_unlab = np.array(gtg_result_prediction[idxs_unlab_again]) == 1
        false_unlab = np.array(gtg_result_prediction[idxs_unlab_again]) == 0
        true_new_lab = np.array(gtg_result_prediction[idxs_new_labels]) == 1
        false_new_lab = np.array(gtg_result_prediction[idxs_new_labels]) == 0
        
        sns.scatterplot(x=x_unlab[idxs_unlab_again][true_unlab], y=y_unlab[idxs_unlab_again][true_unlab], 
                        color='blue', s=20, ax=axes[0], marker='o', label='True Unlabelled Obs')
        
        sns.scatterplot(x=x_unlab[idxs_unlab_again][false_unlab], y=y_unlab[idxs_unlab_again][false_unlab], 
                        color='blue', s=20, ax=axes[0], marker='X', label='False Unlabelled Obs')
        
        sns.scatterplot(x=x_lab, y=y_lab, color='orange', s=20, ax=axes[0], marker='o', label='Old Labeled Obs')
        
        sns.scatterplot(x=x_unlab[idxs_new_labels][true_new_lab], y=y_unlab[idxs_new_labels][true_new_lab], 
                        s=20, color='red', ax=axes[0], marker='o', label='True New-Labelled Obs')
        
        sns.scatterplot(x=x_unlab[idxs_new_labels][false_new_lab], y=y_unlab[idxs_new_labels][false_new_lab], 
                        s=20, color='red', ax=axes[0], marker='X', label='False New-Labelled Obs')
        
    else:
        
        sns.scatterplot(x=x_unlab, y=y_unlab, label='unlabelled', color='blue', s=20, ax=axes[0], marker='o')
        sns.scatterplot(x=x_lab, y=y_lab, label='labelled', color='orange', s=20, ax=axes[0], marker='o')
        sns.scatterplot(x=x_unlab[idxs_new_labels], y=y_unlab[idxs_new_labels], label='new_labelled', 
                        s=20, color='red', ax=axes[0], marker='o')
    
    axes[0].set_title('TSNE -- Labelled - Unlabelled - NewLabelled', fontsize=15)
    
    handles, labels = axes[0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    axes[0].legend(unique_labels.values(), unique_labels.keys(), markerscale=1)
    
    sns_p = sns.scatterplot(x=np.hstack((x_lab, x_unlab)), y=np.hstack((y_lab, y_unlab)), 
                            hue=[classes[l] for l in np.hstack((lab["labels"], unlab["labels"]))], s=20, ax=axes[1])
        
    axes[1].set_title('TSNE - Classes', fontsize=15)
    
    if len(classes) <= 10: 
        fig.text(0.5, 0.05, f'New Labelled Observations: {d_labels}', ha='center', va='center', fontdict={'size': 17})
    else: 
        sns_p.legend([], [], frameon=False)
    
    plt.suptitle(f'TSNE Plot - Iteration {iter}', fontsize=30)
    plt.savefig(f'results/{exp_path}/{ds_name}/{samp_iter}/{method}/tsne_plots/{iter}.png')

    
    
    
def count_class_observation(
        classes: List[str], 
        dataset: Dataset, 
        topk_idx_obs=None
    ) -> Dict[str, int]:
    """Count observations per class in a dataset.

    Counts the number of observations for each class in the provided dataset.
    If topk_idx_obs is provided, only counts observations at those indices;
    otherwise, counts all observations in the dataset.

    Args:
        classes: List of class names corresponding to class indices.
        dataset: PyTorch Dataset object containing the data.
        topk_idx_obs: Optional list of indices to count observations from.
            If None, counts all observations in the dataset.

    Returns:
        Dict[str, int]: Dictionary mapping class names to their observation counts.

    Raises:
        IndexError: If any index in topk_idx_obs is out of bounds.
        KeyError: If a label index is out of range of the classes list.
    """
     
    if topk_idx_obs == None: 
        if len(dataset[0]) == 3: labels = [lab for _, _ ,lab in dataset]
        else: labels = [lab for _, _ ,lab, _ in dataset] #-> for TiDAL
    else: labels = [dataset[k][2] for k in topk_idx_obs]
    
    d_labels = {}
    for cls in classes: d_labels[cls] = 0
    keys_d_labels = list(d_labels.keys())
    for l in labels: d_labels[keys_d_labels[l]] += 1
    return d_labels

        
           
def init_weights_apply(m: torch.nn.Module) -> None:
    """Apply weight initialization to a PyTorch module.

    Initializes weights and biases for different layer types using
    appropriate initialization strategies:
        - Conv2d: Kaiming normal initialization for weights
        - Linear: Normal initialization with std=1e-3 for weights
        - BatchNorm2d: Constant initialization (weight=1, bias=0)

    Args:
        m: A PyTorch module to initialize.

    Returns:
        None. The module is modified in-place.
    """
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None: init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=1e-3)
        if m.bias is not None: init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def log_assert(condition: bool | torch.Tensor, message: str):
    """Assert a condition and log the exception if the assertion fails.

    Args:
        condition: A boolean or torch.Tensor to be evaluated.
            If False or a tensor containing False values, the assertion fails.
        message: The message to be logged when the assertion fails.

    Raises:
        AssertionError: If the condition evaluates to False.
        Exception: Any exception raised by the logger during logging.
    """
    try: assert condition
    except AssertionError as err:
        logger.exception(message)
        raise err
    
    
def save_yamal(
        common_training_params: Dict[str, Any], 
        task_params: Dict[str, Any], 
        al_params: Dict[str, Any], 
        gtg_params: Dict[str, Any], 
        exp_path: str, 
        dataset_name: str
    ) -> None:
    """Save experiment configuration parameters to a YAML file.

    Combines all configuration parameters (training, task, active learning,
    and GTG parameters) into a single dictionary and writes it to a YAML
    file in the results directory.

    Args:
    common_training_params: Dictionary containing common training
        configuration parameters (e.g., batch_size, learning_rate).
    task_params: Dictionary containing task-specific parameters
        (e.g., num_classes, model architecture).
    al_params: Dictionary containing active learning parameters
        (e.g., query_strategy, budget).
    gt_params: Dictionary containing GTG (Greedy Threshold Gradient)
        specific parameters.
    exp_path: Path to the experiment results directory.
    dataset_name: Name of the dataset being used.

    Returns:
        None. The configuration is saved directly to a file.
    """
    
    yaml_dict = copy.deepcopy(common_training_params)
    yaml_dict.update(task_params)
    yaml_dict.update(al_params)
    yaml_dict.update(gtg_params)
    with open(f'results/{exp_path}/{dataset_name}/yamal_conf.yamal', 'w') as file: yaml.dump(yaml_dict, file)


def get_device(n_gpus: int) -> torch.device:
    """Select the best available computation device.

    Priority order:
        1. CUDA
        2. MPS (macOS)
        3. CPU

    Args: 
        n_gpus: The number of GPUs to use if CUDA is available.

    Returns:
        The selected torch device.
    """
    if torch.cuda.is_available():
        logger.info(f"Using {n_gpus} / {torch.cuda.device_count()} of the available GPUs")
        device = torch.device("cuda")

        if dist.is_available():
            logger.info("Distributed package is available")
        if dist.is_nccl_available():
            logger.info("NCCL backend available")

    elif torch.backends.mps.is_available():
        logger.info("CUDA is not available. Using MPS (macOS)")
        device = torch.device("mps")

    else:
        logger.info("CUDA and MPS are not available. Using CPU")
        device = torch.device("cpu")

    logger.info(f"Application running on {device}\n")
    return device
