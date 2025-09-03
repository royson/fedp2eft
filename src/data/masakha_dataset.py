import os
import shutil
import evaluate
import numpy as np
from datasets import Dataset, load_from_disk, load_dataset, concatenate_datasets
from pathlib import Path
from .fl_dataset import FederatedDataset, create_lda_partitions
from transformers import EvalPrediction

import logging
logger = logging.getLogger(__name__)

def preprocess_function(tokenizer, max_length=256, padding='max_length'):
    def tokenize_examples(examples):
        return tokenizer(
            examples["headline_text"],
            padding=padding,
            max_length=max_length,
            truncation=True,
        )
    return tokenize_examples

def get_compute_metrics():
    metric = evaluate.load("accuracy")
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)
    return compute_metrics

class MasakhaNEWSDataset(FederatedDataset):
    def __init__(self, *args, 
                pool='seen',
                no_of_clients_per_language=10,
                alpha=0.5,
                 **kwargs): 
        super().__init__(*args, **kwargs)
        assert pool in ['seen', 'unseen']
        self.pool = pool
        self.alpha = alpha # dirichlet alpha
        self.no_of_clients_per_language = no_of_clients_per_language
        self.languages = ['amh', 
                          'eng', 
                          'fra', 
                          'hau', 
                          'ibo', 
                          'lin', 
                          'lug', 
                          'orm', 
                          'pcm', 
                          'run', 
                          'sna', 
                          'som', 
                          'swa', 
                          'tir', 
                          'xho', 
                          'yor']

        assert len(self.languages) * 2 * self.no_of_clients_per_language == self.pool_size

        if self.val_ratio > 0:
            self.partitions = ['train.json', 'val.json', 'test.json']
        else:
            self.partitions = ['train.json', 'test.json']

        self.dir_path = self.get_fed_dir()
        self._create_fl_partition()

    def get_fed_dir(self):
        name = f'masakha{self.no_of_clients_per_language}_valratio{self.val_ratio}'

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

        train_data, test_data = self.download()

        def _convert_to_dataset(ds):
            return Dataset.from_dict({
                'headline_text': ds['headline_text'],
                'labels': ds['label'],
                'language': [lan] * len(ds['label']),
            })
        
        def _save_client_dataset(client_train, client_test, cid):
            if self.val_ratio > 0:
                client_train_val = client_train.train_test_split(test_size=self.val_ratio, seed=fixed_seed)
                client_train_val['train'].save_to_disk(f'{os.path.join(self.dir_path, str(cid), "train.json")}')
                client_train_val['test'].save_to_disk(f'{os.path.join(self.dir_path, str(cid), "val.json")}')
            else:
                client_train.save_to_disk(f'{os.path.join(self.dir_path, str(cid), "train.json")}')
            
            client_test.save_to_disk(f'{os.path.join(self.dir_path, str(cid), "test.json")}')

        def _create_partition(train_data, test_data):
            dist = None
            train_indices = [np.asarray(range(len(train_data))), np.asarray(train_data['labels'])]
            test_indices = [np.asarray(range(len(test_data))), np.asarray(test_data['labels'])]

            train_par, dist = create_lda_partitions(train_indices, 
                                                    dist, 
                                                    None, 
                                                    num_partitions=self.no_of_clients_per_language, 
                                                    concentration=self.alpha, seed=fixed_seed,
                                                    accept_imbalanced=True)
            test_par, dist = create_lda_partitions(test_indices, 
                                                    dist, 
                                                    None, 
                                                    num_partitions=self.no_of_clients_per_language, 
                                                    concentration=self.alpha, seed=fixed_seed,
                                                    accept_imbalanced=True)
            
            return train_par, test_par

        ###
        for lan_idx, lan in enumerate(self.languages):
            lan_train_data = train_data[lan]
            lan_test_data = test_data[lan]

            shuffled_lan_train_data = lan_train_data.shuffle(seed=fixed_seed)
            shuffled_lan_test_data = lan_test_data.shuffle(seed=fixed_seed)

            train_samples = len(lan_train_data)
            test_samples = len(lan_test_data)

            seen_lan_train_data = _convert_to_dataset(shuffled_lan_train_data[:train_samples // 2])
            seen_lan_test_data = _convert_to_dataset(shuffled_lan_test_data[:test_samples // 2])

            unseen_lan_train_data = _convert_to_dataset(shuffled_lan_train_data[train_samples // 2:train_samples])
            unseen_lan_test_data = _convert_to_dataset(shuffled_lan_test_data[test_samples // 2:test_samples])

            seen_train_par, seen_test_par = _create_partition(seen_lan_train_data, seen_lan_test_data)
            unseen_train_par, unseen_test_par = _create_partition(unseen_lan_train_data, unseen_lan_test_data)


            for cid_lan, (tr_indices, _), (te_indices, _) in zip(range(self.no_of_clients_per_language), seen_train_par, seen_test_par):
                cid = cid_lan + (lan_idx * self.no_of_clients_per_language)
                client_train = seen_lan_train_data.select(list(tr_indices))
                client_test = seen_lan_test_data.select(list(te_indices))                

                _save_client_dataset(client_train, client_test, cid)
                print(f'[*] Seen Pool Language: {lan}. Client {cid}. No. of train: {len(client_train)}. No. of test: {len(client_test)}')
            
            for cid_lan, (tr_indices, _), (te_indices, _) in zip(range(self.no_of_clients_per_language), unseen_train_par, unseen_test_par):
                cid = cid_lan + (lan_idx * self.no_of_clients_per_language)
                client_train = unseen_lan_train_data.select(list(tr_indices))
                client_test = unseen_lan_test_data.select(list(te_indices))                
                        
                _save_client_dataset(client_train, client_test, cid + (len(self.languages) * self.no_of_clients_per_language))
                print(f'[*] Unseen Pool Language: {lan}. Client {cid+(len(self.languages) * self.no_of_clients_per_language)}. \
                    No. of train: {len(client_train)}. No. of test: {len(client_test)}')


    def download(self):
        train_data = {}
        test_data = {}
        for lan in self.languages:
            lan_train_data = load_dataset(
                "masakhane/masakhanews",
                lan,
                split="train",
            )
            lan_val_data = load_dataset(
                "masakhane/masakhanews",
                lan,
                split="validation",
            )

            lan_train_data = concatenate_datasets([lan_train_data, lan_val_data])
            lan_test_data = load_dataset(
                "masakhane/masakhanews",
                lan,
                split="test",
            )
            train_data[lan] = lan_train_data
            test_data[lan] = lan_test_data
        return train_data, test_data

    def _has_fl_partition(self):
        for cid in range(self.pool_size):
            for partition in self.partitions:
                file_path = os.path.join(self.dir_path, str(cid), partition)
                if not os.path.exists(file_path):
                    return False    

        return True

    def get_available_training_clients(self):
        if self.pool == 'seen':
            return list(range(self.pool_size // 2))
        else: # unseen
            return list(range(self.pool_size // 2, self.pool_size))
            
    def get_dataset(self, 
                    data_pool,
                    partition,
                    cid=None, 
                    ):
        data_pool = data_pool.lower()
        partition = partition.lower()
        assert data_pool == 'client'
        assert partition in ('train', 'val', 'test')
        assert cid is not None
        path = Path(self.dir_path) / cid / f'{partition}.json'
        
        return load_from_disk(str(path))

    def get_dataloader(self, 
                    *args,
                    **kwargs):
        # not used. TODO: refactor
        raise NotImplementedError()