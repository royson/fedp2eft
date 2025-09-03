import flwr as fl
import os
import torch
import numpy as np
import transformers
from collections import OrderedDict
from src.log import Checkpoint
from config import AttrDict
from src.utils import get_func_from_config
from peft import PeftModel
from typing import List
from src.apps.clients.fedp2eft_trainers import FedP2EFTSeqClsTrainer
from src.models.btlora.layer import fedbt_update_bts, fedbt_get_bts, fedbt_get_input_output_sizes

from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments, Trainer, get_constant_schedule, default_data_collator


class SeqClsClient(fl.client.NumPyClient):
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
        bt_args=None,
        load_local_layers=False,  # for feddpa/dept
        load_local_tokenizer=False,  # for dept
        **kwargs,
    ):
        transformers.logging.set_verbosity_error()
        self.cid = cid
        self.ckp = ckp
        self.config = ckp.config

        # tokenizer
        self.tokenizer_config = self.config.models.tokenizer
        token_fn = get_func_from_config(self.tokenizer_config)
        if load_local_tokenizer:
            self.tokenizer = token_fn(self.cid, **self.tokenizer_config.args)
        else:
            self.tokenizer = token_fn(**self.tokenizer_config.args)

        # model
        self.load_local_layers = load_local_layers
        self.net_config = self.config.models.net
        arch_fn = get_func_from_config(self.net_config)
        if self.load_local_layers:
            self.net = arch_fn(
                self.cid, **self.net_config.args, tokenizer=self.tokenizer)
        else:
            self.net = arch_fn(**self.net_config.args,
                               tokenizer=self.tokenizer)

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

        # additional adalora kwargs
        self.preprocess_model_func = None
        if hasattr(self.net_config, 'preprocess_model'):
            self.preprocess_model_func = get_func_from_config(
                self.net_config.preprocess_model)

        self.callbacks = []
        if hasattr(self.net_config, 'trainer_callback'):
            callback_func = get_func_from_config(
                self.net_config.trainer_callback)
            self.callbacks.append(callback_func(
                **self.net_config.trainer_callback.args))

        # additional bayestune kwargs
        self.bt_args = bt_args
        if self.bt_args:
            assert self.net_config.args.bt, 'bt arguments were passed but model is not set.'
            _, o_size = fedbt_get_input_output_sizes(self.net)
            self.max_r = self.net_config.args.adapter_args.r
            self.bts_o_size = o_size * self.max_r if self.bt_args.btmode == 'rankwise' else o_size
            self.bt_chunk_size = self.max_r if self.bt_args.btmode == 'rankwise' else 1

            assert self.bt_args.eval_rank, 'please provide an evaluation rank.'
            assert self.bt_args.eval_rank <= self.max_r

            if self.bt_args.load_bts_path:
                self.learned_bts = self.ckp.offline_load(
                    os.path.join(self.bt_args.load_bts_path, f'{self.cid}.pkl'))
                self.bt_train = False
                if self.learned_bts is None:
                    raise FileNotFoundError(
                        f"learned bts not found at {self.bt_args.load_bts_path} for cid {self.cid}")
            else:
                self.learned_bts = torch.ones(self.bts_o_size) * 1e-4
                self.bt_train = True

    def get_parameters(self, **kwargs):
        if isinstance(self.net, PeftModel):
            state_dict = get_peft_model_state_dict(
                self.net, save_embedding_layers=False)
        else:
            state_dict = self.net.state_dict()
        return [val.cpu().to(torch.float32).numpy() for _, val in state_dict.items()]

    def set_parameters(self, parameters):
        if isinstance(self.net, PeftModel):
            peft_state_dict_keys = get_peft_model_state_dict(
                self.net, save_embedding_layers=False).keys()
            params_dict = zip(peft_state_dict_keys, parameters)
            state_dict = OrderedDict({k: torch.Tensor(v)
                                     for k, v in params_dict})
            set_peft_model_state_dict(self.net, state_dict)
        else:
            params_dict = zip(self.net.state_dict().keys(), parameters)
            state_dict = OrderedDict(
                {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
            )
            self.net.load_state_dict(state_dict, strict=True)

    def finetune(self, trainset, num_train_epochs, max_steps, lr):
        self.net.train()

        if self.preprocess_model_func:
            self.preprocess_model_func(**self.net_config.preprocess_model.args, **{
                'model': self.net,
                'dataset': trainset,
                'batch_size': self.batch_size,
                'gradient_accumulation_steps': self.gradient_accumulation_steps,
                'num_train_epochs': self.num_train_epochs,
                'max_steps': self.max_steps,
            })

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

        if type(lr) == float:
            optimizer = torch.optim.AdamW(
                [p for p in self.net.parameters() if p.requires_grad], lr=lr, weight_decay=0)
        else:
            assert len([n for n, p in self.net.named_parameters()
                       if p.requires_grad]) == len(lr)
            params_lr = []
            for _p, _lr in zip([p for p in self.net.parameters() if p.requires_grad], lr):
                params_lr.append({"params": _p, "lr": _lr, "weight_decay": 0})
            optimizer = torch.optim.AdamW(params_lr, weight_decay=0)
        scheduler = get_constant_schedule(optimizer)

        # Construct trainer
        trainer = Trainer(
            model=self.net,
            args=training_args,
            optimizers=(optimizer, scheduler),
            train_dataset=trainset,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=self.callbacks,
        )

        # Do local training
        results = trainer.train()
        return results

    def bt_finetune(self, num_train_epochs, max_steps, lr, seed):
        pretrained_state_dict = get_peft_model_state_dict(
            self.net, save_embedding_layers=False)
        pretrained_weights = [val.cpu().to(torch.float32).numpy()
                              for val in pretrained_state_dict.values()]

        self.net.train()

        trainset = self.get_dataset(data_pool='client',
                                    partition='train',
                                    cid=self.cid)

        # get sub-dataset based on total number of samples needed for finetuning
        if max_steps > 0:
            num_of_samples_needed = min(
                self.batch_size * self.gradient_accumulation_steps * max_steps, len(trainset))
            trainset = trainset.shuffle(seed=seed).select(
                range(num_of_samples_needed))

        trainset = trainset.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=False,
        )

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
            [p for p in self.net.parameters() if p.requires_grad], lr=lr, weight_decay=0)
        scheduler = get_constant_schedule(optimizer)

        self.learned_bts = self.learned_bts.to('cuda')
        bts = self.learned_bts.split(self.bt_chunk_size)
        fedbt_update_bts(self.net, bts, to_params=True)
        bts = fedbt_get_bts(self.net)
        optimizer.add_param_group({'params': bts, 'lr': self.bt_args.bts_lr})

        # Construct trainer
        trainer = FedP2EFTSeqClsTrainer(
            loss_weights=self.bt_args.loss_weights,
            bts_masks=None,  # no masks
            bts_max_clamp=self.bt_args.bts_max_clamp,
            model=self.net,
            tokenizer=self.tokenizer,
            optimizers=(optimizer, scheduler),
            args=training_args,
            train_dataset=trainset,
            data_collator=self.data_collator,
        )

        # Do local training
        results = trainer.train()

        # mask bts for last optimizer.step()
        for idx, bt in enumerate(bts):
            bts[idx] = torch.clamp(
                bt, min=1e-4, max=self.bt_args.bts_max_clamp)

        self.learned_bts = torch.stack(bts).flatten().detach().cpu()
        # self.learned_bts.requires_grad = False
        self.bt_train = False

        # save client's learned_bts
        self.ckp.offline_save(f'{self.config.model_directory}/{self.ckp.name}/{self.cid}.pkl',
                              self.learned_bts)

        # load the base model weights
        pretrained_params_dict = zip(
            pretrained_state_dict.keys(), pretrained_weights)
        pretrained_state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in pretrained_params_dict})
        set_peft_model_state_dict(self.net, pretrained_state_dict)
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

        results = self.finetune(
            trainset, self.num_train_epochs, self.max_steps, round_config.lr)

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
        seed = round_config.seed if 'seed' in round_config else round_config.current_round

        # 0. if bayestune, check if learned bts is provided.
        if self.bt_args and self.bt_train:
            # learned bt not provided. train bt first
            _ = self.bt_finetune(
                num_train_epochs, max_steps, round_config.lr, seed)

        # 1. finetuning before evaluation
        train_results = None
        if max_steps > 0 or num_train_epochs > 0:
            trainset = self.get_dataset(data_pool='client',
                                        partition='train',
                                        cid=self.cid)

            # get sub-dataset based on total number of samples needed for finetuning
            if max_steps > 0:
                num_of_samples_needed = min(
                    self.batch_size * self.gradient_accumulation_steps * max_steps, len(trainset))
                trainset = trainset.shuffle(seed=seed).select(
                    range(num_of_samples_needed))

            trainset = trainset.map(
                self.preprocess_function,
                batched=True,
                load_from_cache_file=False,
            )

            if self.bt_args:
                assert not self.bt_train
                bts_threshold = self.bt_args.eval_rank / self.max_r
                if bts_threshold == 1:
                    mask_bts = self.learned_bts
                else:
                    top_indices = torch.topk(self.learned_bts, int(
                        bts_threshold * len(self.learned_bts)), sorted=False).indices
                    mask_bts = torch.zeros_like(self.learned_bts)
                    mask_bts[top_indices] = self.learned_bts[top_indices]

                mask_bts = mask_bts.to('cuda')
                # print(f'Client {self.cid} #########', torch.min(self.learned_bts), torch.max(self.learned_bts), torch.mean(self.learned_bts), torch.median(self.learned_bts))
                mask_bts = mask_bts.split(self.bt_chunk_size)
                fedbt_update_bts(self.net, mask_bts)

            train_results = self.finetune(
                trainset, num_train_epochs, max_steps, round_config.lr)

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

        if isinstance(self.net, PeftModel):
            self.net = self.net.merge_and_unload().to('cuda')
            self.net.eval()

        trainer = Trainer(
            model=self.net,
            args=training_args,
            eval_dataset=testset,
            compute_metrics=self.compute_metric,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            callbacks=self.callbacks,
        )

        test_results = trainer.evaluate(eval_dataset=testset)

        metrics = {
            'accuracy': test_results['eval_accuracy'],
            'count': len(testset),
            'language': testset[0]['language'],
            'train_loss': train_results.training_loss if train_results is not None else -1,
        }

        return 0., 0, metrics  # server expects loss, num of samples, and dict
