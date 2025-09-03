import flwr as fl
import os
import torch
import numpy as np
import transformers
from collections import OrderedDict
from src.log import Checkpoint
from config import AttrDict
from src.utils import get_func_from_config
from peft import PeftModel, LoraConfig, get_peft_model
from typing import List, Dict
from src.apps.clients.fedp2eft_trainers import FedP2EFTSeqClsTrainer
from src.models.btlora.layer import fedbt_update_bts, fedbt_get_bts, fedbt_get_input_output_sizes
from copy import deepcopy
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments, Trainer, get_constant_schedule, default_data_collator


class FedDPASeqClsClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        ckp: Checkpoint,
        num_train_epochs: float = 2.,  # overrides by max_step if max_step > 0
        max_steps: int = -1,
        batch_size: int = 32,
        gradient_accumulation_steps: int = 1,
        logging_steps: int = 100,
        save_strategy: str = 'no',
        label_names: List[str] = ['labels'],
        local_lora_args: Dict = {},
        **kwargs,
    ):
        transformers.logging.set_verbosity_error()
        self.cid = cid
        self.ckp = ckp
        self.config = ckp.config

        # tokenizer
        self.tokenizer_config = self.config.models.tokenizer
        token_fn = get_func_from_config(self.tokenizer_config)
        self.tokenizer = token_fn(**self.tokenizer_config.args)

        # model
        self.net_config = self.config.models.net
        arch_fn = get_func_from_config(self.net_config)
        self.net = arch_fn(**self.net_config.args)
        self.local_lora_args = local_lora_args
        self.local_lora_weights_path = f'{self.config.model_directory}/{self.ckp.name}/{self.cid}.pt'

        # prompt formatting
        self.data_collator = default_data_collator
        prompt_func = get_func_from_config(
            self.config.models.prompt.formatting_prompt_func)
        self.preprocess_function = prompt_func(
            self.tokenizer, **self.config.models.prompt.formatting_prompt_func.args)

        # evaluation metric
        metric_func = get_func_from_config(self.config.models.metric)
        self.compute_metric = metric_func(**self.config.models.metric.args)

        # data class
        data_config = self.config.data
        data_class = get_func_from_config(data_config)
        self.data_class = data_class(self.ckp, **data_config.args)
        self.get_dataset = self.data_class.get_dataset

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
        state_dict = get_peft_model_state_dict(self.net)
        return [val.cpu().to(torch.float32).numpy() for _, val in state_dict.items()]

    def set_parameters(self, parameters):
        peft_state_dict_keys = get_peft_model_state_dict(self.net).keys()
        params_dict = zip(peft_state_dict_keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        set_peft_model_state_dict(self.net, state_dict)

    def finetune(self, net, trainset, num_train_epochs, max_steps, lr):
        net.train()

        # define training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(
                self.config.model_directory, self.ckp.name),
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

        optimizer = torch.optim.AdamW(
            [p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0)
        scheduler = get_constant_schedule(optimizer)

        # Construct trainer
        trainer = Trainer(
            model=net,
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
            num_of_samples_needed = min(
                self.batch_size * self.gradient_accumulation_steps * self.max_steps, len(trainset))
            trainset = trainset.shuffle(seed=round_config.current_round).select(
                range(num_of_samples_needed))

        trainset = trainset.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=False,
        )

        # train global adapter
        results = self.finetune(
            self.net, trainset, self.num_train_epochs, self.max_steps, round_config.lr)

        # load local adapter
        learner = deepcopy(self.net)
        learner = learner.merge_and_unload()
        local_adapter_config = LoraConfig(
            **self.local_lora_args
        )
        learner = get_peft_model(learner, local_adapter_config)

        # load local adapter lora_params (if any)
        if os.path.exists(self.local_lora_weights_path):
            learner_state_dict = torch.load(self.local_lora_weights_path)
            set_peft_model_state_dict(learner, learner_state_dict)

        # train local adapter: set lora_alpha to take into account of fedDPA's alpha
        results = self.finetune(
            learner, trainset, self.num_train_epochs, self.max_steps, round_config.lr)

        # save local adapter lora_params
        learner_state_dict = get_peft_model_state_dict(learner)
        torch.save(learner_state_dict, self.local_lora_weights_path)

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

        # 2. evaluation
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
            output_dir=os.path.join(
                self.config.model_directory, self.ckp.name),
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

        # merge global adapter
        local_net = self.net.merge_and_unload().to('cuda')

        # merge local adapter
        local_adapter_config = LoraConfig(
            **self.local_lora_args
        )
        local_net = get_peft_model(local_net, local_adapter_config)
        learner_state_dict = torch.load(self.local_lora_weights_path)
        set_peft_model_state_dict(local_net, learner_state_dict)
        local_net = local_net.merge_and_unload().to('cuda')
        local_net.eval()

        trainer = Trainer(
            model=local_net,
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

        return 0., 0, metrics  # server expects loss, num of samples, and dict
