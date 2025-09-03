'''
Adopted from https://github.com/rui-ye/FedLLM-Bench
'''


import os
import json
import shutil

from datasets import Dataset, load_from_disk, concatenate_datasets 
from pathlib import Path
from .fl_dataset import FederatedDataset

import logging
logger = logging.getLogger(__name__)

def concat_dataset(d1, d2):
    if d1 is None:
        return d2
    elif d2 is None:
        return d1
    else:
        return concatenate_datasets([d1, d2])

class FedAyaDataset(FederatedDataset):
    def __init__(self, *args, 
                 test_ratio=0.2, 
                 languages = ['ar', 'en', 'es', 'fr', 'pt', 'ru', 'te', 'zh'],
                 pool='seen',
                 unseen_cids=[21, 22, 23, 24, 25, 26, 27, 34],
                 **kwargs): 
        super().__init__(*args, **kwargs)
        assert test_ratio > 0
        assert pool in ['seen', 'unseen']
        self.pool = pool
        if len(unseen_cids) == 0:
            assert pool == 'seen', 'no unseen clients are selected in an unseen pool'
        self.unseen_cids = unseen_cids
        self.seen_cids = [x for x in range(self.pool_size) if x not in unseen_cids]

        self.test_ratio = test_ratio
        if self.val_ratio > 0:
            self.partitions = ['train.json', 'val.json', 'test.json']
        else:
            self.partitions = ['train.json', 'test.json']

        self.dir_path = self.get_fed_dir()
        self.languages = languages
        self._create_fl_partition()

    def get_fed_dir(self):
        name = f'fedaya_valratio{self.val_ratio}'

        return os.path.join(self.dataset_fl_root, name)

    def _create_fl_partition(self):
        os.umask(0)

        if self.reset and os.path.exists(self.dir_path):
            logger.info(f'Reset flag is set for data federated splitting.. Deleting current {self.dir_path}')
            shutil.rmtree(self.dir_path)

        if self._has_fl_partition():
            logger.info(f"FL partitioned dataset {self.dir_path} found.")
            return 
        
        logger.info(f"Creating FL partitioned dataset {self.dir_path}..")

        if os.path.exists(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.makedirs(self.dir_path)

        fixed_seed = self.config.seed if hasattr(self.config, 'seed') else 42

        with open(self.path_to_data,'r',encoding='utf-8') as f:
            data = json.load(f)
        
        global_train_dataset = None
        if self.val_ratio > 0:
            global_val_dataset = None
        global_test_dataset = None

        # saving client data
        assert len(data) == self.pool_size
        for client_idx, value in enumerate(data.values()):
            dataset = Dataset.from_list(value)

            # prune all languages that are not in self.languages
            excluded_idx = []
            for d_idx, d in enumerate(dataset):
                if not d['language'] in self.languages:
                    excluded_idx.append(d_idx)
            dataset = dataset.select(
                (
                    i for i in range(len(dataset)) 
                    if i not in set(excluded_idx)
                )
            )

            # sanity check
            for d in dataset:
                assert d['language'] in self.languages

            dataset = dataset.train_test_split(test_size=self.test_ratio, seed=fixed_seed)
            dataset['test'].save_to_disk(f'{os.path.join(self.dir_path, str(client_idx), "test.json")}')
            # concat to global test dataset
            global_test_dataset = concat_dataset(global_test_dataset, dataset['test'])

            if self.val_ratio > 0:
                train_val_dataset = dataset['train'].train_test_split(test_size=self.val_ratio, seed=fixed_seed)
                train_val_dataset['train'].save_to_disk(f'{os.path.join(self.dir_path, str(client_idx), "train.json")}')
                train_val_dataset['test'].save_to_disk(f'{os.path.join(self.dir_path, str(client_idx), "val.json")}')
                global_train_dataset = concat_dataset(global_train_dataset, train_val_dataset['train'])
                global_val_dataset = concat_dataset(global_val_dataset, train_val_dataset['test'])
            else:
                dataset['train'].save_to_disk(f'{os.path.join(self.dir_path, str(client_idx), "train.json")}')
                global_train_dataset = concat_dataset(global_train_dataset, dataset['train'])
        
        # saving global data
        global_train_dataset.save_to_disk(f'{os.path.join(self.dir_path, "train.json")}')
        if self.val_ratio > 0:
            global_val_dataset.save_to_disk(f'{os.path.join(self.dir_path, "val.json")}')
        global_test_dataset.save_to_disk(f'{os.path.join(self.dir_path, "test.json")}')

    def download(self):
        assert os.path.exists(self.path_to_data), f'Fed-Aya dataset not found in {self.path_to_data}'
        return

    def _has_fl_partition(self):
        if self.val_ratio > 0 and not os.path.exists(os.path.join(self.dir_path, 'val.json')):
            return False
        
        for cid in range(self.pool_size):
            for partition in self.partitions:
                file_path = os.path.join(self.dir_path, str(cid), partition)
                if not os.path.exists(file_path):
                    return False    

        return True

    def get_available_training_clients(self):
        return self.seen_cids if self.pool == 'seen' else self.unseen_cids            
        # return list(range(self.pool_size))

    def get_dataset(self, 
                    data_pool,
                    partition,
                    cid=None, 
                    ):
        data_pool = data_pool.lower()
        partition = partition.lower()
        assert data_pool in ('server', 'client')
        assert partition in ('train', 'val', 'test')

        if data_pool == 'server':
            assert cid is None
            path = Path(self.dir_path) / f'{partition}.json'
        else:
            assert cid is not None
            path = Path(self.dir_path) / cid / f'{partition}.json'
        
        return load_from_disk(str(path))

    def get_dataloader(self, 
                    *args,
                    **kwargs):
        # not used. TODO: refactor
        raise NotImplementedError()