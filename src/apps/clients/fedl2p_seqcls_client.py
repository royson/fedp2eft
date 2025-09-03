import flwr as fl
import os
import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from config import AttrDict
from src.utils import get_func_from_config
from transformers import Trainer

from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments
from src.apps.clients.fedl2p_trainers import FedL2PTrainer
from src.apps.clients import SeqClsClient
from src.models.weight_net import ClampLRNet, weights_init
from src.models.model_utils import fedl2p_precompute_feats_stats

# import pdb

class FedL2PSeqClsClient(SeqClsClient):
    def __init__(
        self,
        *args,
        inner_loop_lr=1e-4,
        learnable_inner_loop_lr=True,
        outer_loop_steps=3,
        outer_loop_grad_accum=4,
        hypergrad=None,
        meta_optimizers=None,
        task_optimizer=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        ### Compute input and output sizes for LRNet
        adapter_name = self.net_config.args.adapter
        no_of_learnable_layers = 0
        i_size = 0
        for mod_name, m in self.net.named_modules():
            if adapter_name in mod_name and (isinstance(m, torch.nn.Linear) or (isinstance(m,torch.nn.ParameterDict) and len(m) > 0)):
                no_of_learnable_layers +=  1
                if isinstance(m, torch.nn.Linear):
                    m.weight.retain_grad()
            elif isinstance(m, nn.Linear):
                i_size += 1
        i_size *= 2 # mean & std

        assert no_of_learnable_layers > 0, f'{adapter_name} not found in model'

        self.outer_loop_steps = outer_loop_steps
        self.outer_loop_grad_accum = outer_loop_grad_accum
        self.learnable_inner_loop_lr = learnable_inner_loop_lr

        ### Define inner_loop_lrs
        if learnable_inner_loop_lr:
            self.inner_loop_lrs = nn.ParameterList([nn.Parameter(torch.ones(1, dtype=torch.bfloat16) * inner_loop_lr) for _ in range(no_of_learnable_layers)])
        else:
            self.inner_loop_lrs = inner_loop_lr

        ### Define LRNet
        self.lr_net = ClampLRNet(input_size=i_size, output_size=no_of_learnable_layers, max_limit=1000)
        self.lr_net.apply(lambda b: weights_init(b, 0.1, 1.0))

        ### Hypergrad
        if hypergrad is not None:
            self.hypergrad = get_func_from_config(hypergrad)
            self.hypergrad = self.hypergrad(**hypergrad.args)
        else:
            self.hypergrad = hypergrad

        self.meta_optimizers = meta_optimizers
        self.task_optimizer = task_optimizer

    def get_parameters(self, **kwargs):
        params = []
        params.extend([val.cpu().to(torch.float32).numpy() for val in self.lr_net.state_dict().values()])
        if self.learnable_inner_loop_lr:
            params.extend([p.detach().cpu().to(torch.float32).numpy() for p in self.inner_loop_lrs])
        return params

    def set_parameters(self, parameters):
        # load lr net
        params_dict = zip(self.lr_net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        self.lr_net.load_state_dict(state_dict, strict=True)

        # load inner loop lrs
        if self.learnable_inner_loop_lr:
            self.inner_loop_lrs = nn.ParameterList([nn.Parameter(torch.ones(1, dtype=torch.bfloat16) * float(inn_lr.item())) 
                for inn_lr in parameters[len(self.lr_net.state_dict().keys()):]])
        
    def meta_finetune(self, trainset, valset, num_train_epochs, max_steps):
        '''
        returns trainer and train results
        '''
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
            bf16=True,
        )
        
        fedl2p_optimizers = {}
        if self.learnable_inner_loop_lr:
            fedl2p_optimizers['inner_loop_lrs'] = get_func_from_config(self.meta_optimizers['inner_loop_lrs'])(
                list(self.inner_loop_lrs),
                **self.meta_optimizers['inner_loop_lrs'].args
            )
        fedl2p_optimizers['lr_net'] = get_func_from_config(self.meta_optimizers['lr_net'])(
            self.lr_net.parameters(),
            **self.meta_optimizers['lr_net'].args
        )

        fedl2p_inner_optimizer = get_func_from_config(self.task_optimizer)(
            **self.task_optimizer.args
        )

        lr_net_input = fedl2p_precompute_feats_stats(self.net, self.tokenizer, trainset, template=None)

        # Construct trainer
        trainer = FedL2PTrainer(
            args=training_args, 
            model=self.net,
            tokenizer=self.tokenizer,
            val_dataset=valset,
            lr_net=self.lr_net,
            lr_net_input=lr_net_input,
            inner_loop_lrs=self.inner_loop_lrs,
            fedl2p_inner_optimizer=fedl2p_inner_optimizer,
            fedl2p_meta_optimizers=fedl2p_optimizers,
            hypergrad=self.hypergrad,
            outer_loop_steps=self.outer_loop_steps,
            outer_loop_grad_accum=self.outer_loop_grad_accum,
            train_dataset=trainset, 
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
        valset = self.get_dataset(data_pool='client', 
                        partition='val',
                        cid=self.cid)

        # remove samples that exceed the model max length
        if self.max_steps > 0:
            num_of_samples_needed = min(self.batch_size * self.gradient_accumulation_steps * self.max_steps, len(trainset))
            trainset = trainset.shuffle(seed=round_config.current_round).select(range(num_of_samples_needed))
        
        trainset = trainset.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=False,
        )
        valset = valset.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=False,
        )
                
        results = self.meta_finetune(trainset, valset, self.num_train_epochs, self.max_steps)

        # torch.cuda.empty_cache()
        return (
            self.get_parameters(),
            len(trainset),
            results,
        )

    def evaluate(self, parameters, round_config):
        # print(f"evaluate() on client cid={self.cid}")
        round_config = AttrDict(round_config)
        self.set_parameters(parameters)

        num_train_epochs = round_config.num_train_epochs if 'num_train_epochs' in round_config else self.num_train_epochs
        max_steps = round_config.max_steps if 'max_steps' in round_config else self.max_steps
        seed = round_config.seed if 'seed' in round_config else round_config.current_round

        #### 0. Save pretrained model weights to load later
        pretrained_state_dict = get_peft_model_state_dict(self.net, save_embedding_layers=False)
        pretrained_weights = [val.cpu().numpy() for val in pretrained_state_dict.values()]

        #### 1. finetuning before evaluation
        if max_steps > 0 or num_train_epochs > 0: 
            trainset = self.get_dataset(data_pool='client', 
                            partition='train',
                            cid=self.cid)

            # get sub-dataset based on total number of samples needed for finetuning
            if max_steps > 0:
                num_of_samples_needed = min(self.batch_size * self.gradient_accumulation_steps * max_steps, len(trainset))
                trainset = trainset.shuffle(seed=seed).select(range(num_of_samples_needed))

            trainset = trainset.map(
                self.preprocess_function,
                batched=True,
                load_from_cache_file=False,
            )

            lr_net_input = fedl2p_precompute_feats_stats(self.net, self.tokenizer, trainset, template=None)
            with torch.no_grad():
                self.lr_net = self.lr_net.to('cuda', dtype=torch.bfloat16)
                lr_net_input = lr_net_input.to('cuda', dtype=torch.bfloat16)
                lrnet_lrs = self.lr_net(lr_net_input).squeeze().cpu()
            if self.learnable_inner_loop_lr:
                assert len(lrnet_lrs) == len(self.inner_loop_lrs.cpu())
                _lr = [(lr1 * lr2).item() for lr1, lr2 in zip(lrnet_lrs.squeeze(), self.inner_loop_lrs.cpu())]
            else:
                _lr = [lr1.item() * self.inner_loop_lrs for lr1 in lrnet_lrs.squeeze()]

            _ = self.finetune(trainset, num_train_epochs, max_steps, _lr)

        #### 2. evaluation    
        ## create model for evaluation
        net_config = self.config.models.net
        arch_fn = get_func_from_config(net_config)
        if self.load_local_layers:
            net = arch_fn(self.cid, **net_config.args, train=False, tokenizer=self.tokenizer) 
        else:
            net = arch_fn(**net_config.args, train=False, tokenizer=self.tokenizer) 
        
        ## getting ft weights and setting weights to eval model
        trained_state_dict = get_peft_model_state_dict(self.net, save_embedding_layers=False)
        weights = [val.cpu().numpy() for val in trained_state_dict.values()]

        peft_state_dict_keys = get_peft_model_state_dict(net, save_embedding_layers=False).keys()
        params_dict = zip(peft_state_dict_keys, weights)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        set_peft_model_state_dict(net, state_dict)
        
        # merge peft weights
        net = net.merge_and_unload().to('cuda')
        net.eval()

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

        trainer = Trainer(
            model=net,
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
        
        #### 3. load back pretrained weights
        pretrained_params_dict = zip(pretrained_state_dict.keys(), pretrained_weights)
        pretrained_state_dict = OrderedDict({k: torch.Tensor(v) for k, v in pretrained_params_dict})
        set_peft_model_state_dict(self.net, pretrained_state_dict)

        return 0., 0, metrics # server expects loss, num of samples, and dict

