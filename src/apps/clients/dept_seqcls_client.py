import flwr as fl
import os
import torch
import numpy as np
import transformers
from collections import OrderedDict
from src.log import Checkpoint
from config import AttrDict
from src.utils import get_func_from_config
from typing import List

from transformers import TrainingArguments, Trainer, get_constant_schedule, default_data_collator, AutoTokenizer

class DEPTSeqClsClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        ckp: Checkpoint,
        num_train_epochs: float= 2., # overrides by max_step if max_step > 0
        max_steps: int = -1,
        batch_size: int = 32,
        gradient_accumulation_steps: int = 1,
        logging_steps: int=100,
        save_strategy: str='no',
        label_names: List[str]=['labels'],
        local_layer_suffix: str='_embeddings', 
        local_tokenizer: bool= False,
        reset_local_embed: bool= False, 
        **kwargs,
    ):
        transformers.logging.set_verbosity_error()
        self.cid = cid
        self.ckp = ckp
        self.config = ckp.config

        # data class
        data_config = self.config.data
        data_class = get_func_from_config(data_config)
        self.data_class = data_class(self.ckp, **data_config.args)
        self.get_dataset = self.data_class.get_dataset

        # local tokenizer and net
        if not local_tokenizer:
            tokenizer_config = self.config.models.tokenizer
            token_fn = get_func_from_config(tokenizer_config)
            self.tokenizer = token_fn(**tokenizer_config.args)
        else:
            self.local_tokenizer_path = f'{self.config.model_directory}/{self.ckp.name}/{self.cid}_tokenizer'

            if os.path.exists(self.local_tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_tokenizer_path)
            else:
                # create local tokenizer
                trainset = self.get_dataset(data_pool='client', 
                                partition='train',
                                cid=self.cid)
                tokenizer_config = self.config.models.tokenizer
                token_fn = get_func_from_config(tokenizer_config)
                global_tokenizer = token_fn(**tokenizer_config.args)

                '''
                hotfix training_corpus for specific datasets. TODO: refactor to take in a function instead
                '''
                if 'premise' in trainset.features and 'hypothesis' in trainset.features:
                    training_corpus = (
                        trainset["premise"][i : i + 10] + trainset["hypothesis"][i : i + 10] 
                        for i in range(0, len(trainset["premise"]), 10)
                    )
                elif 'headline_text' in trainset.features:
                    training_corpus = (
                        trainset["headline_text"][i : i + 10] 
                        for i in range(0, len(trainset["headline_text"]), 10)
                    )
                else:
                    raise NotImplementedError
                
                # vocab size will be smaller/bigger depending on yr dataset
                self.tokenizer = global_tokenizer.train_new_from_iterator(training_corpus, 50257)
                
                self.tokenizer.save_pretrained(self.local_tokenizer_path)

        self.net_config = self.config.models.net
        arch_fn = get_func_from_config(self.net_config)
        self.net = arch_fn(**self.net_config.args) 
        if local_tokenizer:   
            self.net.resize_token_embeddings(len(self.tokenizer))
        self.local_embedding_path = f'{self.config.model_directory}/{self.ckp.name}/{self.cid}_embeds.pt'
        if not os.path.exists(self.local_embedding_path) and reset_local_embed:
            '''
            hotfix for bert model only. TODO: refactor to take in a list of parameters
            '''
            self.net.bert.embeddings.word_embeddings.reset_parameters()
            self.net.bert.embeddings.position_embeddings.reset_parameters()
            self.net.bert.embeddings.token_type_embeddings.reset_parameters()

        self.local_layer_suffix = local_layer_suffix
                
        # prompt formatting
        self.data_collator = default_data_collator
        prompt_func = get_func_from_config(self.config.models.prompt.formatting_prompt_func)
        self.preprocess_function = prompt_func(self.tokenizer, **self.config.models.prompt.formatting_prompt_func.args)

        # evaluation metric
        metric_func = get_func_from_config(self.config.models.metric)
        self.compute_metric = metric_func(**self.config.models.metric.args)
        
        # hyperparameters
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.label_names = label_names

        # additional training kwargs
        self.logging_steps = logging_steps
        self.save_strategy = save_strategy

    def get_parameters(self, **kwargs):
        state_dict = {k:v for k,v in self.net.state_dict().items() if self.local_layer_suffix not in k}
        return [val.cpu().to(torch.float32).numpy() for _, val in state_dict.items()]

    def set_parameters(self, parameters):
        params_dict = zip([k for k in self.net.state_dict().keys() if self.local_layer_suffix not in k], parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        self.net.load_state_dict(state_dict, strict=False)

    def finetune(self, trainset, num_train_epochs, max_steps, lr):
        self.net.train()

        # define training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.model_directory, self.ckp.name),
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            logging_steps=self.logging_steps,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            report_to='none',
            save_strategy=self.save_strategy,
            gradient_checkpointing=self.config.models.net.args.gradient_checkpointing,
            label_names=self.label_names,
        )

        optimizer = torch.optim.AdamW([p for p in self.net.parameters() if p.requires_grad], lr=lr, weight_decay=0)
        scheduler = get_constant_schedule(optimizer)

        # Construct trainer
        trainer = Trainer(
            model=self.net,
            args=training_args, 
            optimizers=(optimizer, scheduler),
            train_dataset=trainset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        # Do local training
        results = trainer.train()
        return results

    def fit(self, parameters, round_config):
        # print(f"fit() on client cid={self.cid}")
        round_config = AttrDict(round_config)
        self.set_parameters(parameters)
        trainset = self.get_dataset(data_pool='client', 
                        partition='train',
                        cid=self.cid)

        # get sub-dataset based on total number of samples needed for finetuning
        if self.max_steps > 0:
            num_of_samples_needed = min(self.batch_size * self.gradient_accumulation_steps * self.max_steps, len(trainset))
            trainset = trainset.shuffle(seed=round_config.current_round).select(range(num_of_samples_needed))
        
        trainset = trainset.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=False,
        )

        # load local embed
        if os.path.exists(self.local_embedding_path):
            embeds_sd = torch.load(self.local_embedding_path)
            self.net.load_state_dict(embeds_sd, strict=False)

        results = self.finetune(trainset, self.num_train_epochs, self.max_steps, round_config.lr)

        # save local embed
        embeds_sd = {k:v for k,v in self.net.state_dict().items() if self.local_layer_suffix in k}

        torch.save(embeds_sd, self.local_embedding_path)

        return (
            self.get_parameters(),
            len(trainset),
            {"train_loss": results.training_loss},
        )

    def evaluate(self, parameters, round_config):
        # print(f"evaluate() on client cid={self.cid}")
        round_config = AttrDict(round_config)
        self.set_parameters(parameters)

        num_train_epochs = round_config.num_train_epochs if 'num_train_epochs' in round_config else self.num_train_epochs
        max_steps = round_config.max_steps if 'max_steps' in round_config else self.max_steps

        #### 2. evaluation
        # local test data
        testset = self.get_dataset(data_pool='client', 
                        partition='test',
                        cid=self.cid)
        testset = testset.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=False,
        )
        
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.model_directory, self.ckp.name),
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            logging_steps=self.logging_steps,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            report_to='none',
            save_strategy=self.save_strategy,
            gradient_checkpointing=self.config.models.net.args.gradient_checkpointing,
            label_names=self.label_names,
        )

        # load local embeds
        if os.path.exists(self.local_embedding_path):
            embeds_sd = torch.load(self.local_embedding_path)
            self.net.load_state_dict(embeds_sd, strict=False)

        self.net.eval()

        trainer = Trainer(
            model=self.net,
            args=training_args,
            eval_dataset=testset,
            compute_metrics=self.compute_metric,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )

        test_results = trainer.evaluate(eval_dataset=testset)
        
        metrics = {
            'accuracy': test_results['eval_accuracy'],
            'count': len(testset),
            'language': testset[0]['language'],
        }

        return 0., 0, metrics # server expects loss, num of samples, and dict

