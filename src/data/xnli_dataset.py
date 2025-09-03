import os
import shutil
import evaluate
import numpy as np
from datasets import Dataset, load_from_disk, load_dataset
from pathlib import Path
from .fl_dataset import FederatedDataset, create_lda_partitions
from transformers import EvalPrediction

import logging
logger = logging.getLogger(__name__)

def preprocess_function(tokenizer, max_length=128, padding='max_length'):
    def tokenize_examples(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding=padding,
            max_length=max_length,
            truncation=True,
        )
    return tokenize_examples

def get_compute_metrics():
    metric = evaluate.load("xnli")
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)
    return compute_metrics

class XNLIDataset(FederatedDataset):
    def __init__(self, *args, 
                train_samples=2000,
                test_samples=500,
                no_of_clients_per_language=20,
                alpha=0.5,
                pool='seen',
                 **kwargs): 
        super().__init__(*args, **kwargs)
        assert train_samples > 0 and test_samples > 0
        assert pool in ['seen', 'unseen']
        self.pool = pool
        self.alpha = alpha # dirichlet alpha

        self.train_samples = train_samples
        self.test_samples = test_samples
        self.no_of_clients_per_language = no_of_clients_per_language
        if self.val_ratio > 0:
            self.partitions = ['train.json', 'val.json', 'test.json']
        else:
            self.partitions = ['train.json', 'test.json']

        self.dir_path = self.get_fed_dir()
        self._create_fl_partition()

    def get_fed_dir(self):
        name = f'xnli{self.no_of_clients_per_language}_valratio{self.val_ratio}'

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
        all_languages = train_data[0]['hypothesis']['language']

        # randomly sample from datasets
        assert len(train_data) >= self.train_samples * 2 # 2 for both seen and unseen
        assert len(test_data) >= self.test_samples * 2
        assert len(all_languages) * 2 * self.no_of_clients_per_language == self.pool_size

        shuffled_train_data = train_data.shuffle(seed=fixed_seed)
        shuffled_test_data = test_data.shuffle(seed=fixed_seed)

        seen_train_data = shuffled_train_data[:self.train_samples]
        seen_test_data = shuffled_test_data[:self.test_samples]

        unseen_train_data = shuffled_train_data[self.train_samples:self.train_samples*2]
        unseen_test_data = shuffled_test_data[self.test_samples:self.test_samples*2]

        dist = None
        train_indices = [np.asarray(range(2000)), np.asarray(seen_train_data['label'])]
        test_indices = [np.asarray(range(500)), np.asarray(seen_test_data['label'])]

        # seen and unseen use the same label partitioning for all languages
        train_par, dist = create_lda_partitions(train_indices, 
                                                dist, 
                                                None, 
                                                num_partitions=self.no_of_clients_per_language, 
                                                concentration=self.alpha, seed=fixed_seed)
        test_par, dist = create_lda_partitions(test_indices, 
                                                dist, 
                                                None, 
                                                num_partitions=self.no_of_clients_per_language, 
                                                concentration=self.alpha, seed=fixed_seed)

        def _partition_xnli(dataset):
            partition_dataset = {}
            for lan_idx in range(len(all_languages)):
                partition_dataset[lan_idx] = {
                    'premise': [],
                    'hypothesis': [],
                    'labels': [],
                    'language': [],
                }
            for prem, hypo, label in zip(dataset['premise'], dataset['hypothesis'], dataset['label']):
                for lan_idx, lang in enumerate(all_languages):
                    partition_dataset[lan_idx]['premise'].append(prem[lang])
                    partition_dataset[lan_idx]['hypothesis'].append(hypo['translation'][lan_idx])
                    partition_dataset[lan_idx]['labels'].append(label)
                    partition_dataset[lan_idx]['language'].append(lang)

            return partition_dataset

        def _create_pool(pool_train_data, pool_test_data, starting_cid=0):
            partitioned_train_data = _partition_xnli(pool_train_data)
            partitioned_test_data = _partition_xnli(pool_test_data)

            for lan_idx in range(len(all_languages)):
                ds_train = Dataset.from_dict(partitioned_train_data[lan_idx])
                ds_test = Dataset.from_dict(partitioned_test_data[lan_idx])

                for cid_lan, (tr_indices, _), (te_indices, _) in zip(range(self.no_of_clients_per_language), train_par, test_par):
                    cid = cid_lan + starting_cid
                    client_train = ds_train.select(list(tr_indices))
                    client_test = ds_test.select(list(te_indices))

                    if self.val_ratio > 0:
                        client_train_val = client_train.train_test_split(test_size=self.val_ratio, seed=fixed_seed)
                        client_train_val['train'].save_to_disk(f'{os.path.join(self.dir_path, str(cid), "train.json")}')
                        client_train_val['test'].save_to_disk(f'{os.path.join(self.dir_path, str(cid), "val.json")}')
                    else:
                        client_train.save_to_disk(f'{os.path.join(self.dir_path, str(cid), "train.json")}')
                    
                    client_test.save_to_disk(f'{os.path.join(self.dir_path, str(cid), "test.json")}')

                starting_cid += self.no_of_clients_per_language

        _create_pool(seen_train_data, seen_test_data, starting_cid=0)
        _create_pool(unseen_train_data, unseen_test_data, starting_cid=len(all_languages) * self.no_of_clients_per_language)

    def download(self):
        train_data = load_dataset(
            "xnli",
            'all_languages',
            split="train",
            cache_dir=self.path_to_data,
        )

        test_data = load_dataset(
            "xnli",
            'all_languages',
            split="test",
            cache_dir=self.path_to_data,
        )
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