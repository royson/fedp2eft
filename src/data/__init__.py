from .utils import *
from src.utils import get_func_from_config
from .meteor import Meteor
from .fedaya_dataset import FedAyaDataset
from .xnli_dataset import XNLIDataset
from .masakha_dataset import MasakhaNEWSDataset


def prepare_fl_partitioned_dataset(ckp):
    '''
        partition dataset (use a large `alpha` to make it IID;
        a small value (e.g. 1) will make it non-IID)
        This will create a new directory called "federated: in the directory where
        the dataset lives. Inside it, there will be N=pool_size sub-directories each with
        its own train/val/test split.
    '''
    data_config = ckp.config.data
    data_class = get_func_from_config(data_config)

    dataset = data_class(ckp, **data_config.args)
    if not dataset.pre_partition:
        dataset.create_fl_partitions()
