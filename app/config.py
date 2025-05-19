
from torchvision import datasets
from torchvision import transforms
import torch


cls_datasets = {
    'cifar10': {
        'id': 1,
        'method': datasets.CIFAR10,
        'n_classes': 10,
        'channels': 3,
        'image_size': 32,
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        'transforms': {
            'train': transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                     std=[0.24703223, 0.24348513, 0.26158784])
                        
                        
                    ]),
            'test': transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.49139968, 0.48215841, 0.44653091],
                                     std=[0.24703223, 0.24348513, 0.26158784])
                    ])
        }
    },
    'cifar100': {
        'id': 2,
        'method': datasets.CIFAR100,
        'n_classes': 100,
        'channels': 3,
        'image_size': 32,
        'classes': ["apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle","bowl","boy","bridge",
                    "bus","butterfly","camel","can","castle","caterpillar","cattle","chair","chimpanzee","clock","cloud","cockroach",
                    "couch","crab","crocodile","cup","dinosaur","dolphin","elephant","flatfish","forest","fox","girl","hamster","house",
                    "kangaroo","keyboard","lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain",
                    "mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree","plain","plate",
                    "poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket","rose","sea","seal","shark","shrew","skunk",
                    "skyscraper","snail","snake","spider","squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone",
                    "television","tiger","tractor","train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm"],
        'transforms': {
            'train': transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.50707516, 0.48654887, 0.44091784],
                                     std=[0.26733429, 0.25643846, 0.27615047])
                    ]),
            'test': transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.50707516, 0.48654887, 0.44091784],
                                     std=[0.26733429, 0.25643846, 0.27615047])
                    ])
        }
    },
    'svhn': {
        'id': 3,
        'n_classes': 10,
        'method': datasets.SVHN,
        'image_size': 32,
        'channels': 3,
        'classes': ['0','1','2','3','4','5','6','7','8','9'],
        'transforms': {
            'train': transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, padding=4),
                        transforms.ToTensor(),
                        #transforms.Normalize(mean=[0.4376821 , 0.4437697 , 0.47280442], # -> remove normalization since it can produce black images
                        #             std=[0.19803012, 0.20101562, 0.19703614])
                    ]),
            'test': transforms.Compose([
                        transforms.ToTensor(),
                        #transforms.Normalize(mean=[0.4376821 , 0.4437697 , 0.47280442], # -> remove normalization since it can produce black images
                        #             std=[0.19803012, 0.20101562, 0.19703614])
                    ])
        }
    },
    'tinyimagenet': {
        'id': 4,
        'n_classes': 200,
        'channels': 3,
        'image_size': 64,
        'classes': [],
        'transforms': {
            'train': transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(64, padding=8),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.48042979, 0.44819701, 0.39755623],
                                     std=[0.27643974, 0.26888656, 0.28166852])
                    ]),
            'test': transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.48042979, 0.44819701, 0.39755623],
                                     std=[0.27643974, 0.26888656, 0.28166852])
                    ])
        }
    },
    'caltech256': {
        'id': 5,
        'n_classes': 256,
        'channels': 3,
        'image_size': 224,
        'classes':[],
        'transforms': {
            'train': [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(224, padding=28),
                        transforms.ToTensor(),
                    ],
            'test': [
                        transforms.ToTensor(),
                    ],
        }   
    },
    'fmnist': {
        'id': 6,
        'method': datasets.FashionMNIST,
        'n_classes': 10,
        'channels': 1,
        'image_size': 28,
        'classes': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
        'transforms': {
            'train': transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(28, padding=3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.2860405969887955], std=[0.3530242445149223])
                    ]),
            'test': transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.2860405969887955], std=[0.3530242445149223])
                    ]),
        } 
    }
}




al_params = {
    'init_lab_obs': 1000,
    'al_iters': 10, 
    'n_top_k_obs': 1000,
    'unlab_sample_dim': 10000,
}
    

cls_config = {
    'epochs': 200,
    'results_dict': { 'train': {'train_accuracy': [], 'train_loss': [], 'train_loss_ce': [], 'train_pred_loss': []},
                     'test': {'test_accuracy': []}},
    'cifar10': {
        'batch_size': { None: 128, 'LossNet': 128, 'GTGModule': 256 },
        'optimizers': {
            'backbone':{
                'type': { None: torch.optim.SGD, 'LossNet': torch.optim.SGD, 'GTGModule': torch.optim.SGD, },
                'optim_p': {
                    None: { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'LossNet': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'GTGModule': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                }
            },
            'modules': {
                'type': { 'LossNet': torch.optim.SGD, 'GTGModule': torch.optim.Adam, },
                'decay': { 'LossNet': 120, 'GTGModule': 120 },
                'optim_p': {
                    'LossNet': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'GTGModule': { 'lr': 0.001 ,}, #'momentum': 0.9, 'weight_decay': 5e-4 }, # SGD
                }
            },
            'milestones': { 'backbone': 160, 'LossNet': 160, 'GTGModule': 200 },
        }
    },

    'cifar100': {
        'batch_size': { None: 128, 'LossNet': 128, 'GTGModule': 256 },
        'optimizers': {
            'backbone':{
                'type': { None: torch.optim.SGD, 'LossNet': torch.optim.SGD, 'GTGModule': torch.optim.SGD, },
                'optim_p': {
                    None: { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'LossNet': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'GTGModule': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                }
            },
            'modules': {
                'type': { 'LossNet': torch.optim.SGD, 'GTGModule': torch.optim.Adam, },
                'decay': { 'LossNet': 120, 'GTGModule': 120, },
                'optim_p': {
                    'LossNet': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'GTGModule': { 'lr': 0.001, }
                }
            },
            'milestones': { 'backbone': 160, 'LossNet': 160, 'GTGModule': 200 },
        }
    },
    'svhn': {
        'batch_size': { None: 128, 'LossNet': 128, 'GTGModule': 256 },
        'optimizers': {
            'backbone':{
                'type': { None: torch.optim.SGD, 'LossNet': torch.optim.SGD, 'GTGModule': torch.optim.SGD, },
                'optim_p': {
                    None: { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'LossNet': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'GTGModule': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                }
            },
            'modules': {
                'type': { 'LossNet': torch.optim.SGD, 'GTGModule': torch.optim.Adam, },
                'decay': { 'LossNet': 120, 'GTGModule': 120, },
                'optim_p': {
                    'LossNet': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'GTGModule': { 'lr': 0.001, }
                }
            },
            'milestones': { 'backbone': 160, 'LossNet': 160, 'GTGModule': 200 },
        }
    },
    'fmnist': {
        'batch_size': { None: 128, 'LossNet': 128, 'GTGModule': 256 },
        'optimizers': {
            'backbone':{
                'type': { None: torch.optim.SGD, 'LossNet': torch.optim.SGD, 'GTGModule': torch.optim.SGD, },
                'optim_p': {
                    None: { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'LossNet': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'GTGModule': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 }, 
                }
            },
            'modules': {
                'type': { 'LossNet': torch.optim.SGD, 'GTGModule': torch.optim.Adam, },
                'decay': { 'LossNet': 120, 'GTGModule': 120, },
                'optim_p': {
                    'LossNet': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'GTGModule': { 'lr': 0.001, }
                }
            },
            'milestones': { 'backbone': 160, 'LossNet': 160, 'GTGModule': 200 },
        }
    },
    'caltech256': {
        'batch_size': { None: 16, 'LossNet': 16, 'GTGModule': 32 },
        'optimizers': {
            'backbone':{
                'type': { None: torch.optim.SGD, 'LossNet': torch.optim.SGD, 'GTGModule': torch.optim.SGD, },
                'optim_p': {
                    None: { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'LossNet': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'GTGModule': { 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4 },
                }
            },
            'modules': {
                'type': { 'LossNet': torch.optim.SGD, 'GTGModule': torch.optim.Adam, },
                'decay': { 'LossNet': 120, 'GTGModule': 120, },
                'optim_p': {
                    'LossNet': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'GTGModule': { 'lr': 0.001, }
                }
            },
            'milestones': { 'backbone': 160, 'LossNet': 160, 'GTGModule': 200 },
        }
    },
    'tinyimagenet': {
        'batch_size': { None: 128, 'LossNet': 128, 'GTGModule': 126 },
        'optimizers': {
            'backbone':{
                'type': { None: torch.optim.SGD, 'LossNet': torch.optim.SGD, 'GTGModule': torch.optim.SGD, },
                'optim_p': {
                    None: { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'LossNet': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'GTGModule': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                }
            },
            'modules': {
                'type': { 'LossNet': torch.optim.SGD, 'GTGModule': torch.optim.Adam, },
                'decay': { 'LossNet': 120, 'GTGModule': 120, },
                'optim_p': {
                    'LossNet': { 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4 },
                    'GTGModule': { 'lr': 0.001, }
                }
            },
            'milestones': { 'backbone': 160, 'LossNet': 160, 'GTGModule': 200 },
        }
    }
}







det_datasets = {
    'voc': {
        'id': 5,
        'n_classes': 21,
        'channels': 3,
        'image_size': 300,
        'dataset_path': 'datasets/voc/VOCdevkit', 
        
        'classes': [#'background',
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor'
            ],
    }
}


voc_config = {

    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
}


det_config = {
    'lr_steps': (80000, 100000, 120000),
    '3s': 300,
    'batch_size': 32,
    'results_dict': { 'train': { 'train_loss': [] , 'train_loc_loss': [], 'train_conf_loss': [], 'train_pred_loss': []},
                     'test': {'test_map': []}
                    }
}