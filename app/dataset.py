
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets
import torch
import numpy as np

from typing import List, Tuple, Dict, Any
import os
import shutil

from torchvision import transforms

from auto_encoder.Train_AE import fit_ae, get_initial_sample_higher_MSE, get_initial_sample_farthest_KMeans
from auto_encoder.BackBone_Decoder import BackBone_Decoder
from utils import count_class_observation, log_assert, set_seeds
from config import cls_datasets, al_params

import logging
logger = logging.getLogger(__name__)



def download_tinyimagenet() -> None:
    if not os.path.exists('datasets/tiny-imagenet-200'):
        logger.info(' => Downloading Tiny-IMAGENET Dataset')
        os.system('wget http://cs231n.stanford.edu/tiny-imagenet-200.zip')
        os.system('unzip tiny-imagenet-200.zip -d datasets')
        os.remove('tiny-imagenet-200.zip')
        logger.info(' DONE\n')
    else:
        logger.info('Tiny-IMAGENET Dataset already downloaded')
        
        
def create_val_img_folder(dataset_dir):
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


def preprocess_caltech256_dir(caltech256_path):
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


def download_caltech256():
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


def create_train_test_dir(list_idxs: Tuple[List[int], List[int]], train_data: Dataset, split_path: str, \
                          split_types: Tuple[str, str], classes: List[str]) -> None:
    for idxs, split_type in zip(list_idxs, split_types):
        for i in idxs:
            pil_image, label = train_data[i]
            image_path = os.path.join(split_path, split_type, classes[label])
            os.makedirs(image_path, exist_ok=True)
            id = len(os.listdir(image_path)) + 1
            pil_image.save(os.path.join(split_path, split_type, classes[int(label)], f'{id}.png'), 'PNG')


def compute_mean_std(data: Dataset, idxs=None) -> Tuple[np.ndarray, np.ndarray]:
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
    mean, std = mean_std
    cls_datasets["caltech256"]["transforms"]["train"].append(transforms.Normalize(mean=mean.tolist(), std=std.tolist()))
    cls_datasets["caltech256"]["transforms"]["train"] = transforms.Compose(cls_datasets["caltech256"]["transforms"]["train"])
                
    cls_datasets["caltech256"]["transforms"]["test"].append(transforms.Normalize(mean=mean.tolist(), std=std.tolist()))
    cls_datasets["caltech256"]["transforms"]["test"] = transforms.Compose(cls_datasets["caltech256"]["transforms"]["test"])


def init_caltech256() -> None:
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

        self.n_classes: int = cls_datasets[dataset_name]["n_classes"]
        self.n_channels: int = cls_datasets[dataset_name]["channels"]
        self.dataset_id: int = cls_datasets[dataset_name]["id"]
        self.image_size: int = cls_datasets[dataset_name]["image_size"]
        
        self.train_ds = Cls_Dataset(dataset_name=dataset_name, bool_train=True, bool_transform=True, n_classes=self.n_classes)
        self.unlab_train_ds = Cls_Dataset(dataset_name=dataset_name, bool_train=True, bool_transform=False)
        
        self.test_ds = Cls_Dataset(dataset_name=dataset_name, bool_train=False, bool_transform=False)
 
        if dataset_name == 'tinyimagenet':
            cls_datasets[dataset_name]["classes"] = os.listdir('./datasets/tiny-imagenet-200/train')
            
        self.classes: List[str] = cls_datasets[dataset_name]["classes"]
        
        
        
    def get_initial_subsets_cold_start(self, device: torch.device, strategy: str, batch_size: int = 128, n_obs_per_class = 100) -> Dict[str, Any]:

        auto_encoder = BackBone_Decoder(classes=self.n_classes, n_channels=self.n_channels, image_size=self.image_size)
        fit_ae(auto_encoder, device, self.train_ds)
        dl = DataLoader(self.train_ds, shuffle=True, pin_memory=True, batch_size=batch_size)
        
        self.labelled_indices = get_initial_sample_higher_MSE(auto_encoder, dl, device, al_params["init_lab_obs"]) if strategy == 'mse' \
            else get_initial_sample_farthest_KMeans(auto_encoder, dl, device, al_params["init_lab_obs"], n_obs_per_class)
        
        # the indices are created my araging
        self.unlabelled_indices = [idx for idx in np.arange(len(self.train_ds)) if idx not in self.labelled_indices]
        
        return auto_encoder.encoder.state_dict()
    
    
    
    def get_initial_subsets(self, trial_id: int) -> None:

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
    def __init__(self, dataset_name: str, bool_train: bool, bool_transform: bool, n_classes: int = -1) -> None:
        
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
        
    