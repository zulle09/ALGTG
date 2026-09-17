al_params = {
    'init_lab_obs': 1000,
    'al_iters': 10, 
    'n_top_k_obs': 1000,
    'unlab_sample_dim': 10000,
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
    'epochs': 300,
    'voc': { 'batch_size': { None: 32, 'LossNet': 32 } },
    'results_dict': { 'train': { 'train_loss': [] , 'train_loc_loss': [], 'train_conf_loss': [], 'train_pred_loss': []},
                     'test': {'test_map': []}
                    }
}