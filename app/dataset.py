
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets
import torch
import numpy as np

from typing import List, Tuple, Dict, Any
import os
import subprocess
import shutil

from torchvision import transforms

from auto_encoder.Train_AE import fit_ae, get_initial_sample_higher_MSE, get_initial_sample_farthest_KMeans
from auto_encoder.BackBone_Decoder import BackBone_Decoder
from utils import count_class_observation, log_assert, set_seeds
from config import cls_datasets, al_params

import logging
logger = logging.getLogger(__name__)



def download_tinyimagenet() -> None:
    """Download the Tiny-IMAGENET-200 dataset if not already present.

    Checks if the Tiny-IMAGENET-200 dataset is already downloaded in the
    'datasets/tiny-imagenet-200' directory. If not present, downloads the
    dataset from the Stanford CS231n website, extracts it to the datasets
    directory, and removes the zip file after extraction.

    Logs the download progress and status using the logger.

    Returns:
        None: This function does not return any value.

    Raises:
        None: No explicit exceptions are raised, but subprocess errors from
            wget or unzip commands may occur if the download fails.
    """
    if not os.path.exists('datasets/tiny-imagenet-200'):
        logger.info(' => Downloading Tiny-IMAGENET Dataset')
        subprocess.run('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip')
        subprocess.run('unzip tiny-imagenet-200.zip -d datasets')
        subprocess.run('rm tiny-imagenet-200.zip')
        logger.info(' DONE\n')
    else:
        logger.info('Tiny-IMAGENET Dataset already downloaded')
        
        
def create_val_img_folder(dataset_dir: str) -> None:
    """
    Creates a validation image folder structure and organizes images based on annotations.

    This function reads a validation annotations file ('val_annotations.txt') and
    reorganizes validation images into their corresponding class folders. It creates
    the necessary directory structure if it doesn't already exist.

    Args:
        dataset_dir (str): The path to the dataset directory. This should be the
            parent directory containing (or to contain) the 'val' subdirectory.

    Raises:
        FileNotFoundError: If 'val_annotations.txt' does not exist in the val directory.
        OSError: If there are issues creating directories or moving files.

    Note:
        - The function expects 'val_annotations.txt' to be tab-separated with
          image filenames in the first column and folder/class names in the second.
        - If the dataset directory already exists, the function logs an info message
          and does nothing.
        - Images are moved from 'val/images/' to 'val/images/<folder_name>/' based
          on their annotations.
    """
    if not os.path.exists(dataset_dir):
        val_dir = os.path.join(dataset_dir, 'val')
        img_dir = os.path.join(val_dir, 'images')

        fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
        data = fp.readlines()
        val_img_dict = {}
        for line in data:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]
        fp.close()

        # Create folder if not present and move images into proper folders
        for img, folder in val_img_dict.items():
            newpath = (os.path.join(img_dir, folder))
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            if os.path.exists(os.path.join(img_dir, img)):
                os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))
    else:
        logger.info(f'{dataset_dir} already exists')


def preprocess_caltech256_dir(caltech256_path: str) -> int:
    """Preprocess Caltech256 dataset directory by renaming folders and files.

    Renames subdirectories in the Caltech256 dataset by extracting the class
    name (part after the last dot) and renames image files by extracting the
    image number (part after the last underscore).

    Args:
        caltech256_path: Path to the root Caltech256 dataset directory.

    Returns:
        int: Total number of images processed in all subdirectories.

    Raises:
        OSError: If directory operations fail (e.g., path not found,
            permission denied).
    """
    n_images = 0
    for dir_name in os.listdir(caltech256_path):
        new_dir_name = dir_name.split('.')[-1]
        os.rename(os.path.join(caltech256_path, dir_name), os.path.join(caltech256_path, new_dir_name))
        for file_name in os.listdir(os.path.join(caltech256_path, new_dir_name)):
            n_images += 1
            new_file_name = file_name.split('_')[-1]
            os.rename(os.path.join(caltech256_path, new_dir_name, file_name),
                      os.path.join(caltech256_path, new_dir_name, new_file_name))
    return n_images


def download_caltech256() -> int:
    """Download and extract the Caltech-256 dataset.
    Creates the dataset directory, downloads the Caltech-256 dataset
    from the official Caltech data repository, extracts the archive,
    and removes unnecessary files and directories (non-images and
    clutter class).
    Returns:
        int: The return value from preprocess_caltech256_dir, typically
            the number of processed images or classes.
    Raises:
        OSError: If wget fails to download or tar fails to extract.
    """
    os.makedirs('./datasets/caltech256', exist_ok=True)
    logger.info(' => Downloading Caltech256 Dataset')
    # downlaod dataset
    os.system('wget -P ./datasets/caltech256 https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar')
    # extract and remove it
    os.system('tar -xopf ./datasets/caltech256/256_ObjectCategories.tar -C ./datasets/caltech256')
    os.remove('./datasets/caltech256/256_ObjectCategories.tar')
    # remove non-images
    os.remove(os.path.join('./datasets/caltech256/256_ObjectCategories', '198.spider/RENAME2'))
    shutil.rmtree(os.path.join('./datasets/caltech256/256_ObjectCategories', '056.dog/greg'))
    # we don't need the class with noise
    shutil.rmtree(os.path.join('./datasets/caltech256/256_ObjectCategories', '257.clutter'))
        
    return preprocess_caltech256_dir('./datasets/caltech256/256_ObjectCategories')


def create_train_test_dir(
        list_idxs: Tuple[List[int], List[int]],
        train_data: Dataset,
        split_path: str, 
        split_types: Tuple[str, str], 
        classes: List[str]
    ) -> None:
    """Creates train and test directories and saves images from a dataset.

    Iterates through the provided indices for each split type (e.g., train/test),
    retrieves images and labels from the dataset, and saves them organized by
    class in the appropriate directory structure.

    Args:
        list_idxs: A tuple containing two lists of indices, one for each split
            (e.g., training and testing indices).
        train_data: A Dataset object from which to retrieve images and labels.
        split_path: The base path where train and test directories will be created.
        split_types: A tuple of two strings specifying the split type names
            (e.g., ('train', 'test')).
        classes: A list of class names corresponding to the label indices.

    Returns:
        None. The function saves images to disk and creates directories as needed.
    """
    for idxs, split_type in zip(list_idxs, split_types):
        for i in idxs:
            pil_image, label = train_data[i]
            image_path = os.path.join(split_path, split_type, classes[label])
            os.makedirs(image_path, exist_ok=True)
            id = len(os.listdir(image_path)) + 1
            pil_image.save(os.path.join(split_path, split_type, classes[int(label)], f'{id}.png'), 'PNG')



def compute_mean_std(data: Dataset, idxs=None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the mean and standard deviation of images in a dataset.

    Args:
        data: A dataset object that supports indexing (data[i]) and iteration.
            Each item should return a tuple where the first element is the image.
        idxs (optional): A sequence of indices specifying which samples to use.
            If None, all samples in the dataset are used. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - mean: The mean pixel values across all images, normalized by 255.
            - std: The standard deviation of pixel values across all images,
              normalized by 255.

    Raises:
        ValueError: If the dataset is empty or no valid images are found.

    Note:
        - Grayscale images (2D arrays) are automatically converted to 3-channel
          images by repeating the channel dimension.
        - The mean and std are computed per channel across all pixels in the
          selected images.
        - Values are normalized to [0, 1] range by dividing by 255.
    """
    logger.info(' => Computing mean and std for the train test split...')
    images = []
    if idxs != None:
        for i in idxs:
            if len(np.asarray(data[i][0]).shape) == 3: images.append(np.asarray(data[i][0]))
            else: images.append(np.repeat(np.asarray(data[i][0])[:, :, np.newaxis], 3, axis=2))
    else: 
        for img, _ in data: images.append(np.asarray(img))
    images = np.concatenate(images)
    mean = np.mean(images, axis=(0, 1)) / 255 
    std = np.std(images, axis=(0, 1)) / 255
    return mean, std


def complete_caltech_transforms(mean_std: Tuple[np.ndarray, np.ndarray]) -> None:
    """Add normalization transforms to the Caltech256 dataset.
    Appends a Normalize transform to both the training and testing transform
    pipelines for the Caltech256 dataset, using the provided mean and standard
    deviation values. Each transform list is then wrapped in a Compose object.
    Args:
        mean_std: A tuple containing:
            - mean: numpy array of mean values for each channel
            - std: numpy array of standard deviation values for each channel
    Returns:
        None
    """
    mean, std = mean_std
    cls_datasets["caltech256"]["transforms"]["train"].append(transforms.Normalize(mean=mean.tolist(), std=std.tolist()))
    cls_datasets["caltech256"]["transforms"]["train"] = transforms.Compose(cls_datasets["caltech256"]["transforms"]["train"])
                
    cls_datasets["caltech256"]["transforms"]["test"].append(transforms.Normalize(mean=mean.tolist(), std=std.tolist()))
    cls_datasets["caltech256"]["transforms"]["test"] = transforms.Compose(cls_datasets["caltech256"]["transforms"]["test"])


def init_caltech256() -> None:
    """Initialize and prepare the Caltech256 dataset for training.
    This function checks if the Caltech256 dataset exists locally. If not, it
    downloads the dataset, creates an 80/20 train/test split, and organizes
    the data into appropriate directories with computed normalization values.
    The function performs the following steps:
        1. Checks if the dataset already exists in the expected directory.
        2. If exists, loads the class names and sets up transforms.
        3. If not, downloads the dataset and splits it into train (80%) and
           test (20%) sets.
        4. Creates new directory structure for train/test split.
        5. Computes mean and standard deviation for image normalization.
        6. Replaces the original dataset directory with the split version.
    Returns:
        None: This function does not return any value. It modifies the
            cls_datasets dictionary in-place and creates/organizes dataset
            files on disk.
    Raises:
        OSError: If there are issues with file system operations like
            creating directories or removing/renaming files.
    """

    if os.path.exists('./datasets/caltech256/256_ObjectCategories'): 
        logger.info(' Caltech256 dataset already obtained and ready!')
        cls_datasets["caltech256"]["classes"] = sorted(os.listdir('./datasets/caltech256/256_ObjectCategories/train'))
        complete_caltech_transforms(compute_mean_std(datasets.ImageFolder('./datasets/caltech256/256_ObjectCategories')))
        return 
    
    n_images = download_caltech256()
    classes = sorted(os.listdir('./datasets/caltech256/256_ObjectCategories'))
    cls_datasets["caltech256"]["classes"] = classes    
        
    logger.info(' => Generating new train test split datasets...')
    
    rand_perm = torch.randperm(n_images)
    
    # 80% for the training set and the remaning 20% for the test set
    train_size = int(n_images * 0.8)
    train_idxs = rand_perm[:train_size].tolist()
    test_idxs = rand_perm[train_size:].tolist()
                
    train_data = datasets.ImageFolder('./datasets/caltech256/256_ObjectCategories', transform=transforms.Compose([transforms.Resize((224,224))]))
        
    split_path = './datasets/caltech256/256_ObjectCategories_new'
    os.mkdir(split_path)
    create_train_test_dir((train_idxs, test_idxs), train_data, split_path, ('train', 'test'), classes)
    
    complete_caltech_transforms(compute_mean_std(train_data, train_idxs))

    shutil.rmtree('./datasets/caltech256/256_ObjectCategories')
    os.rename(split_path, './datasets/caltech256/256_ObjectCategories')

    logger.info(' DONE\n')



class Cls_Datasets():
    
    def __init__(self, dataset_name: str) -> None:
        
        # we set these attributes to be used in the strategy objects and the training loop
        self.n_classes: int = cls_datasets[dataset_name]["n_classes"]
        self.n_channels: int = cls_datasets[dataset_name]["channels"]
        self.dataset_id: int = cls_datasets[dataset_name]["id"]
        self.image_size: int = cls_datasets[dataset_name]["image_size"]
        
        # set train and unlabelled dataset with and without transforms, respectively, and test dataset without transforms
        self.train_ds = Cls_Dataset(dataset_name=dataset_name, bool_train=True, bool_transform=True, n_classes=self.n_classes)
        self.unlab_train_ds = Cls_Dataset(dataset_name=dataset_name, bool_train=True, bool_transform=False)
        
        self.test_ds = Cls_Dataset(dataset_name=dataset_name, bool_train=False, bool_transform=False)
 
        if dataset_name == 'tinyimagenet':
            cls_datasets[dataset_name]["classes"] = os.listdir('./datasets/tiny-imagenet-200/train')
            
        self.classes: List[str] = cls_datasets[dataset_name]["classes"]
        
        
        
    def get_initial_subsets_cold_start(
            self, 
            device: torch.device,
            strategy: str,
            batch_size: int = 128,
            n_obs_per_class = 100
        ) -> Dict[str, Any]:
        """
        Selects initial labelled and unlabelled subsets for cold-start active learning.
        This method initializes the active learning process by training an autoencoder
        on the training dataset and selecting initial samples based on the specified
        strategy. Two strategies are supported: 'mse' (Mean Squared Error) for selecting
        samples with highest reconstruction error, and 'kmeans' (or other) for selecting
        samples farthest from cluster centers.
        Args:
            device: The torch device (CPU or CUDA) to use for training and inference.
            strategy: The selection strategy to use. Either 'mse' for highest MSE or
                any other value for farthest K-Means clustering.
            batch_size: The batch size for the DataLoader. Defaults to 128.
            n_obs_per_class: The number of observations per class to select when using
                K-Means strategy. Defaults to 100.
        Returns:
            Dict[str, Any]: A dictionary containing the state dictionary of the trained
                autoencoder's encoder, which can be used for further training or inference.
        Raises:
            ValueError: If the specified strategy is not supported.
            RuntimeError: If autoencoder training fails.

        """

        auto_encoder = BackBone_Decoder(classes=self.n_classes, n_channels=self.n_channels, image_size=self.image_size)
        fit_ae(model=auto_encoder, device=device, train_ds=self.train_ds)
        logger.info(device)
        logger.info(torch.device("cuda"))
        dl = DataLoader(self.train_ds, shuffle=True, pin_memory=(device != "mps"), batch_size=batch_size)
        
        self.labelled_indices = get_initial_sample_higher_MSE(auto_encoder, dl, device, al_params["init_lab_obs"]) if strategy == 'mse' \
            else get_initial_sample_farthest_KMeans(auto_encoder, dl, device, al_params["init_lab_obs"], n_obs_per_class)
        
        # the indices are created my araging
        self.unlabelled_indices = [int(idx) for idx in np.arange(len(self.train_ds)) if idx not in self.labelled_indices]
        
        return auto_encoder.encoder.state_dict()
    
    
    
    def get_initial_subsets(self, trial_id: int) -> None:
        """Initializes the initial subsets of labelled and unlabelled observations for active learning.
        This method creates a random split of the training dataset into labelled and unlabelled
        subsets for a given trial. The split is performed by randomly shuffling the training
        indices and selecting the first `init_lab_obs` indices as labelled, with the remaining
        indices as unlabelled.
        Args:
            trial_id (int): The trial identifier used for seed setting to ensure reproducibility
                across different trials.
        Returns:
            None: This method updates the instance attributes `labelled_indices` and
                `unlabelled_indices` with the selected indices.
        Raises:
            KeyError: If `init_lab_obs` is not found in `al_params`.
            IndexError: If `init_lab_obs` is larger than the training dataset size.
        Note:
            - The seed is set using `dataset_id * trial_id` before shuffling to ensure
              reproducibility for a specific trial.
            - The seed is reset after shuffling to avoid affecting subsequent operations.
            - The initial subset composition is logged for monitoring purposes.
        """

        init_lab_obs = al_params["init_lab_obs"]

        train_size = len(self.train_ds)

        set_seeds(self.dataset_id * trial_id)
        
        # random shuffle of the train indices
        shuffled_indices = torch.randperm(train_size)
        # each time should be a new shuffle, thus a new train-validation, labelled-unlabelled split
        
        set_seeds()

        # indices for the labelled and unlabelled sets
        self.labelled_indices: List[int] = shuffled_indices[:init_lab_obs].tolist()
        self.unlabelled_indices: List[int] = shuffled_indices[init_lab_obs:].tolist()
        
        logger.info(f' Initial subset of labelled observations composed with: {count_class_observation(self.classes, Subset(self.train_ds, self.labelled_indices))}')




class Cls_Dataset(Dataset):
    
    def __init__(
            self, 
            dataset_name: str, 
            bool_train: bool, 
            bool_transform: bool, 
            n_classes: int = -1
        ) -> None:
        
        """
        A custom dataset wrapper class that handles various image datasets.
        This class provides a unified interface for loading and accessing different
        image datasets (e.g., TinyImageNet, SVHN, Caltech256) with appropriate
        transforms. It supports both training and testing modes.
        Args:
            dataset_name (str): Name of the dataset to load. Supported values include
                'tinyimagenet', 'svhn', 'caltech256', and other datasets supported by
                torchvision (e.g., 'cifar10', 'cifar100', 'mnist').
            bool_train (bool): If True, loads the training split; otherwise loads the
                test split.
            bool_transform (bool): If True, applies training transforms; otherwise
                applies test transforms.
            n_classes (int, optional): Number of classes in the dataset. Required when
                both bool_train and bool_transform are True. Defaults to -1.
        Attributes:
            ds (Dataset): The underlying torchvision Dataset instance.
            moving_prob (torch.Tensor, optional): A tensor of shape (len(ds), n_classes)
                initialized to zeros. Only created when training with transforms enabled.
        Raises:
            AssertionError: If n_classes is -1 when both bool_train and bool_transform
                are True.
        Note:
            - For 'tinyimagenet', the dataset is downloaded and a validation folder is
            created if not already present.
            - For 'caltech256', the dataset is initialized if not already present.
            - The moving_prob attribute is used for the TiDAL method.
        Example:
            >>> dataset = Cls_Dataset('cifar10', bool_train=True, bool_transform=True, n_classes=10)
            >>> len(dataset)
            50000
            >>> index, image, label, prob = dataset[0]
    """
        str_train = 'train' if bool_train else 'test'
        str_transform = 'train' if bool_transform else 'test'
        
        if dataset_name == 'tinyimagenet':
            download_tinyimagenet()
            create_val_img_folder('./datasets/tiny-imagenet-200')
            self.ds: Dataset = datasets.ImageFolder(f'./datasets/tiny-imagenet-200/{str_train}',
                transform=cls_datasets["tinyimagenet"]["transforms"][str_transform]
            )
        elif dataset_name == 'svhn':
            self.ds: Dataset = cls_datasets["svhn"]["method"]('./datasets/svhn', 
                split=str_train, download=True,
                transform=cls_datasets["svhn"]["transforms"][str_transform]
            )
        elif dataset_name == 'caltech256':
            init_caltech256()
            self.ds: Dataset = datasets.ImageFolder(f'./datasets/caltech256/256_ObjectCategories/{str_train}',
                transform=cls_datasets["caltech256"]["transforms"][str_transform]
            )
        else:
            self.ds: Dataset = cls_datasets[dataset_name]["method"](f'./datasets/{dataset_name}',
                train=bool_train, download=True,
                transform=cls_datasets[dataset_name]["transforms"][str_transform]
            )
        
        if bool_train and bool_transform: 
            log_assert(n_classes != -1, 'Invalid n_classes')
            self.moving_prob = torch.zeros((len(self.ds), n_classes), dtype=torch.float32, device=torch.device('cpu')) # type: ignore  # -> for TiDAL
        
        
        
    def __len__(self) -> int:
        return len(self.ds) # type: ignore


    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor, int] | Tuple[int, torch.Tensor, int, torch.Tensor]:
        image, label = self.ds[index]
        if hasattr(self, 'moving_prob'):
            return index, image, label, self.moving_prob[index] # -> for TiDAL
        else: return index, image, label
        
    