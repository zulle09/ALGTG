
from app.models.modules.GTG.GTG_Cls import GTGModule
from models.modules.LossNet import LossNet
from models.backbones.ResNet18 import ResNet, ResNet18
from models.backbones.VGG import VGG, VGG16_bn

from strategies.baselines.Random import Random
from strategies.baselines.Entropy import Entropy
from strategies.baselines.LearningLoss import LearningLoss
from strategies.competitors.CoreSet import CoreSet
from strategies.competitors.BADGE import BADGE
from strategies.competitors.BALD import BALD
from strategies.competitors.CDAL import CDAL
from strategies.competitors.TA_VAAL.TA_VAAL import TA_VAAL
from strategies.competitors.AlphaMix import AlphaMix
from strategies.competitors.TiDAL import TiDAL
from strategies.gtg.GTG import GTG
from strategies.gtg.GTG_off import GTG_off

from models.BBone_Module import Master_Model

from typing import Dict, Any, List, Tuple

    
def get_backbone(
        n_classes: int, 
        n_channels: int, 
        bbone: str, 
        img_size: int
    ) -> ResNet | VGG:
    """
    Selects and returns a backbone model based on the specified architecture.

    Args:
        n_classes (int): Number of classes for the classification head.
        n_channels (int): Number of input channels (e.g., 3 for RGB).
        bbone (str): Backbone architecture name. Must be 'ResNet' or 'VGG'.
        img_size (int): Input image size (used for ResNet only).

    Returns:
        ResNet | VGG: An instance of either ResNet18 or VGG16_bn model
            configured with the specified parameters.

    Raises:
        ValueError: If an unsupported backbone architecture is provided.
    """
    if bbone == 'ResNet': 
        return ResNet18(
            n_classes=n_classes, 
            n_channels=n_channels, 
            img_size=img_size
        )
    else: 
        return VGG16_bn(
            n_classes=n_classes,  
            n_channels=n_channels
        )
    

def get_module(module: str, module_params: Dict[str, Any] | Tuple[Dict[str, Any], Dict[str, Any]]) -> LossNet | GTGModule:
    if module == 'LL': return LossNet(module_params)
    else: return GTGModule(module_params)


def get_ll_module_params(image_size: int, dataset_name: str) -> Dict[str, int | str | List[int]]:
    if dataset_name != 'caltech256':   
        return dict(
                feature_sizes=[image_size, image_size // 2, image_size // 4, image_size // 8],
                num_channels=[64, 128, 256, 512],
                interm_dim=128, task='cls'
            )
    else: 
        return dict(
                feature_sizes=[224, 224, 112, 56],
                num_channels=[64, 64, 128, 256],
                interm_dim=128, task='cls'
            )
            
            
# to create the strategy objects based on the methods list
dict_strategies_BB = dict(
    random = Random, entropy = Entropy, coreset = CoreSet, badge = BADGE, # -> BB
    bald = BALD, cdal = CDAL, alphamix = AlphaMix, # -> BB
)

# to create the strategy objects based on the methods list
dict_strategies_LL = dict( ll = LearningLoss, tavaal = TA_VAAL, tidal = TiDAL ) # -> BB + LL

# to create the strategy objects based on the methods list
dict_strategies_GTG = dict(
    llmlp_gtg = GTG, lsmlps_gtg = GTG, lsmlpsA_gtg = GTG, lstmreg_gtg = GTG, lstmbc_gtg = GTG, # -> BB + GTG
    gtg = GTG_off, # -> BB
    ll_gtg = GTG_off # -> BB + LL
)

# to select the backbone based on the dataset
dict_backbone = dict(
    cifar10 = 'ResNet', cifar100 = 'ResNet', svhn = 'ResNet', fmnist = 'ResNet',
    caltech256 = 'VGG', tinyimagenet = 'ResNet'
)


def get_strategies_object(
        methods: List[str], 
        Masters: Dict[str, Master_Model], 
        ct_p: Dict[str, Any], 
        t_p: Dict[str, Any], 
        gtg_p: Dict[str, Any]
    ) -> List[Any]:

    """
    Generate a list of active learning strategy objects based on the specified methods.
    This function instantiates various active learning strategies based on the provided
    method names. It supports three categories of strategies:
    - GTG-based strategies (GTG, GTG_off, etc.)
    - Learning Loss based strategies (ll, tavaal, tidal)
    - Baseline strategies (Random, Entropy, CoreSet, AlphaMix, BADGE, BALD, CDAL)
    
    Args:
        methods (List[str]): List of method names to instantiate strategies for.
            Supported methods include: 'gtg', 'gtg_off', 'll', 'tavaal', 'tidal',
            'random', 'entropy', 'coreset', 'alphamix', 'badge', 'bald', 'cdal'.
        Masters (Dict[str, Master_Model]): Dictionary mapping model identifiers to
            Master_Model instances. Expected keys include 'M_None', 'M_LL',
            'M_LL_tidal', and 'M_GTG_*' variants.
        ct_p (Dict[str, Any]): Configuration parameters dictionary containing
            common settings for strategy initialization.
        t_p (Dict[str, Any]): Training parameters dictionary passed to strategies.
        gtg_p (Dict[str, Any]): GTG-specific parameters dictionary containing:
            - 'am_ts': List of am_ts values to iterate over
            - 'am': List of am values to iterate over
    
    Returns:
        List[Any]: A list of instantiated strategy objects. For GTG methods with
            multiple 'am_ts' and 'am' configurations, multiple strategy instances
            are returned (one for each combination).
    
    Raises:
        KeyError: If an unknown method is provided or required Master_Model
            is not found in the Masters dictionary.
        ValueError: If method name format is invalid for GTG-based strategies.
    
    Note:
        - For 'gtg' method, M_None Master_Model is used
        - For 'll' (learning loss) methods, M_LL Master_Model is used
        - For 'tidal' method, M_LL_tidal Master_Model is used
        - For other GTG variants (e.g., 'gtg_coral'), M_GTG_{variant} is used
        - GTG methods are expanded into multiple strategy instances based on
            the cartesian product of 'am_ts' and 'am' parameter lists
    """
    
    strategies: List[
        Random | Entropy | CoreSet | AlphaMix | BADGE | BALD | CDAL 
        | GTG_off | LearningLoss | GTG_off | TiDAL | TA_VAAL | GTG
    ] = []
    
    for method in methods:
        method_split = method.split('_')
        if 'gtg' in method_split:
            
            for id_am_ts in range(len(gtg_p['am_ts'])):
                
                for id_am in range(len(gtg_p['am'])):
                    if method == 'gtg': m_model = Masters['M_None']
                    elif method_split[0] == 'll': m_model = Masters['M_LL']
                    else: m_model = Masters[f'M_GTG_{method_split[0]}']

                    strategies.append(
                        dict_strategies_GTG[method](
                            {**ct_p, 'Master_Model': m_model}, t_p,
                            {**gtg_p, 'id_am_ts': id_am_ts, 'id_am': id_am, 'gtg_model': method_split[0]}
                        )
                    )
                
        elif method in ['ll', 'tavaal', 'tidal']:
            strategies.append(dict_strategies_LL[method]({ **ct_p, 'Master_Model': Masters['M_LL'] if method != 'tidal' else Masters['M_LL_tidal']}, t_p))
        else:
            strategies.append(dict_strategies_BB[method]({**ct_p, 'Master_Model': Masters['M_None']}, t_p))
    
    return strategies


# to create a single master model for each type
def get_masters(
        methods: List[str],
        BBone: ResNet | VGG,
        ll_module_params: Dict[str, Any], 
        gtg_module_params: Dict[str, Any],
        dataset_name: str, 
        n_classes: int
    ) -> Dict[str, Master_Model]:
    """
    Create and return a dictionary of master models based on the specified methods.
    This function initializes various master models by combining a backbone network
    (ResNet or VGG) with different modules (LL or GTG) based on the methods provided.
    Each master model represents a different approach for the learning task.
    
    Args:
        methods (List[str]): List of method names to create master models for.
            Supported methods: 'll', 'll_gtg', 'tavaal', 'tidal', 'llmlp_gtg',
            'lsmlps_gtg', 'lsmlpsA_gtg', 'lstmreg_gtg', 'lstmbc_gtg'.
        BBone (ResNet | VGG): The backbone neural network architecture to use.
        ll_module_params (Dict[str, Any]): Parameters for the LL (Linear Layer) module.
        gtg_module_params (Dict[str, Any]): Parameters for the GTG (Gated Task Graph) module.
        dataset_name (str): Name of the dataset being used.
        n_classes (int): Number of classes in the classification task.
    
    Returns:
        Dict[str, Master_Model]: A dictionary mapping master model names to their
            corresponding Master_Model instances. Keys include:
            - 'M_LL': Master with LL module (module_out=1) for 'll', 'll_gtg', 'tavaal'
            - 'M_LL_tidal': Master with LL module (module_out=n_classes) for 'tidal'
            - 'M_GTG_{gtg_module}': Master with GTG module for GTG-based methods
            - 'M_None': Backbone-only model (no module) if no specific method is used
    """
    
    ll, ll_tidal, only_bb = False, False, False
    
    Masters = { }
    
    # create and save the initial checkpoints of the masters
    for method in methods:
        if (method in ['ll', 'll_gtg', 'tavaal']) and not ll:
            
            Masters['M_LL'] = Master_Model(
                backbone=BBone, 
                added_module=get_module('LL', {**ll_module_params, 'module_out': 1}), 
                dataset_name=dataset_name
            )
            ll = True
        
        elif method == 'tidal' and not ll_tidal:

            Masters['M_LL_tidal'] = Master_Model(
                backbone=BBone,
                added_module=get_module('LL', {**ll_module_params, 'module_out': n_classes}),
                dataset_name=dataset_name
            )
            ll_tidal = True
        
        elif method in ['llmlp_gtg', 'lsmlps_gtg', 'lsmlpsA_gtg', 'lstmreg_gtg', 'lstmbc_gtg']:
            
            gtg_module = method.split('_')[0]
            Masters[f'M_GTG_{gtg_module}'] = Master_Model(
                backbone=BBone, 
                added_module=get_module(
                    'GTG', ({**gtg_module_params, 'gtg_module': gtg_module}, {**ll_module_params, 'module_out': 1})
                ), 
                dataset_name=dataset_name
            )

        elif not only_bb:
            
            Masters['M_None'] = Master_Model(backbone=BBone, added_module=None, dataset_name=dataset_name)
            only_bb = True
        
        else: continue
    
    # now the master: backbone + module initial checkpoint have been saved 
    
    return Masters
    