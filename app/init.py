from models.modules.LossNet import LossNet
from models.backbones.ssd_pytorch.SSD import SSD, build_ssd

from strategies.baselines.Random import Random
from strategies.baselines.LearningLoss import LearningLoss

from models.BBone_Module import Master_Model

from datasets_creation.Detection import Det_Dataset

from config import voc_config
from typing import Dict, Any, List


def get_dataset(dataset_name: str) -> Det_Dataset:
    return Det_Dataset(dataset_name)


def get_backbone(n_classes: int) -> SSD:
    return build_ssd('train', voc_config, num_classes=n_classes)


def get_ll_module_params() -> Dict[str, int | List[int]]:
    return dict(
                feature_sizes=[512, 1024, 512, 256, 256, 256],
                num_channels=[512, 1024, 512, 256, 256, 256],
                interm_dim=128
            )


dict_strategies = dict( random = Random, ll = LearningLoss )


def get_strategies_object(methods: List[str], Masters: Dict[str, Master_Model],
                          ct_p: Dict[str, Any], t_p: Dict[str, Any]) -> List[Random | LearningLoss]:
    strategies: List[Random | LearningLoss] = []

    for method in methods:
        m_model = Masters['M_LL'] if method == 'll' else Masters['M_None']
        strategies.append(dict_strategies[method]({**ct_p, 'Master_Model': m_model}, t_p))

    return strategies


# to create a single master model for each type
def get_masters(methods: List[str], BBone: SSD, ll_module_params: Dict[str, Any], dataset_name: str) -> Dict[str, Master_Model]:

    Masters = { }

    # create and save the initial checkpoints of the masters
    for method in methods:
        if method == 'll' and 'M_LL' not in Masters:
            Masters['M_LL'] = Master_Model(BBone, LossNet({**ll_module_params, 'module_out': 1}), dataset_name)
        elif method != 'll' and 'M_None' not in Masters:
            Masters['M_None'] = Master_Model(BBone, None, dataset_name)

    # now the master: backbone + module initial checkpoint have been saved

    return Masters
