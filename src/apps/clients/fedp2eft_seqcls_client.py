import flwr as fl
import os
import torch
import numpy as np
from collections import OrderedDict
from config import AttrDict
from src.utils import get_func_from_config

from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments, get_constant_schedule, Trainer
from src.apps.clients.fedp2eft_trainers import FedP2EFTSeqClsTrainer
from src.apps.clients import SeqClsClient
from src.models.weight_net import BTNet, weights_init
from src.models.model_utils import fedl2p_precompute_feats_stats
from src.models.btlora.layer import fedbt_update_bts, fedbt_get_bts, fedbt_get_input_output_sizes


class FedP2EFTSeqClsClient(SeqClsClient):
    def __init__(
        self,
        *args,
        btmode='rankwise',  # 'layerwise' or 'rankwise'
        bt_steps=5,
        btnet_optimizer=None,
        loss_weights=None,
        eval_rank=None,
        eval_train_bts=False,
        eval_rank_only=False,
        sample_bts=False,
        sample_bts_warmup=20,
        include_quantity=False,
        bt_hl_mul=None,
        task_lr=1e-4,
        bts_lr=1e-4,
        bts_max_clamp=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Compute input and output sizes for BTNet
        adapter_name = self.net_config.args.adapter
        i_size, o_size = fedbt_get_input_output_sizes(self.net)
        self.max_r = self.net_config.args.adapter_args.r
        btnet_o_size = o_size * self.max_r if btmode == 'rankwise' else o_size

        if include_quantity:
            i_size += 1

        assert o_size > 0, f'{adapter_name} not found in model'
        self.no_of_adapter_modules = o_size

        # Define BTNet
        self.bts_max_clamp = bts_max_clamp
        hl_size = i_size * \
            bt_hl_mul if bt_hl_mul else int(0.75*i_size + btnet_o_size)
        self.bt_net = BTNet(input_size=i_size, hl_size=hl_size,
                            output_size=btnet_o_size, max_limit=self.bts_max_clamp)
        self.bt_net.apply(lambda b: weights_init(b, 1e-6, 1e-4))

        self.bt_net_chunk_size = self.max_r if btmode == 'rankwise' else 1
        self.bt_steps = bt_steps
        self.loss_weights = loss_weights
        self.btnet_optimizer = btnet_optimizer
        self.task_lr = task_lr
        self.bts_lr = bts_lr
        self.eval_threshold = None
        self.eval_rank_only = eval_rank_only
        self.eval_train_bts = eval_train_bts
        if eval_rank:
            self.eval_threshold = eval_rank / self.max_r
            assert self.eval_threshold <= 1, 'evaluation rank >= max rank of model'
        if self.eval_train_bts:
            assert self.eval_threshold is not None

        # sampling top-k bts rank instead of using max rank
        self.sample_ranks = []
        if sample_bts:
            _max_r = self.max_r
            while _max_r > 0:
                self.sample_ranks.append(_max_r)
                _max_r = _max_r // 2
            self.sample_bts_warmup = sample_bts_warmup

        self.include_quantity = include_quantity

    def get_parameters(self, **kwargs):
        params = []
        params.extend([val.cpu().to(torch.float32).numpy()
                      for val in self.bt_net.state_dict().values()])
        return params

    def set_parameters(self, parameters):
        params_dict = zip(self.bt_net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        self.bt_net.load_state_dict(state_dict, strict=True)

    def two_stage_finetune(self, current_round, trainset, num_train_epochs, max_steps, seed):

        # 0. Save pretrained model weights to load later
        pretrained_state_dict = get_peft_model_state_dict(
            self.net, save_embedding_layers=False)
        pretrained_weights = [val.cpu().to(torch.float32).numpy()
                              for val in pretrained_state_dict.values()]

        self.net.train()
        # 1. Define training arguments, optimizers and scheduler
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

        btnet_optimizer = get_func_from_config(self.btnet_optimizer)(
            self.bt_net.parameters(),
            **self.btnet_optimizer.args
        )

        optimizer = torch.optim.AdamW([p for p in self.net.parameters(
        ) if p.requires_grad], lr=self.task_lr, weight_decay=0)
        scheduler = get_constant_schedule(optimizer)

        # 2. get initial BTS and update self.net
        fedbt_update_bts(self.net, torch.zeros(self.no_of_adapter_modules))
        bt_net_input = fedl2p_precompute_feats_stats(
            self.net, self.tokenizer, trainset, template=None)

        if self.include_quantity:
            quantity = torch.log(torch.tensor([len(trainset)])).unsqueeze(
                0).to(bt_net_input.device)
            bt_net_input = torch.cat([bt_net_input, quantity], dim=1)

        with torch.no_grad():
            self.bt_net = self.bt_net.to('cuda', dtype=torch.bfloat16)
            bt_net_input = bt_net_input.to('cuda', dtype=torch.bfloat16)
            bts = self.bt_net(bt_net_input).squeeze()

        bts_threshold = 1
        if self.sample_ranks and current_round > self.sample_bts_warmup:
            rng = np.random.default_rng(seed)
            r_c = rng.choice(self.sample_ranks)
            # print(self.sample_ranks)
            bts_threshold = r_c / self.max_r
            print(
                f'seed: {seed}, choice r: {r_c}, bts_threshold: {bts_threshold}')

        bts_grad_masks = None
        mask_grad_bts = None
        if bts_threshold == 1:
            mask_bts = bts
        else:
            top_indices = torch.topk(
                bts, int(bts_threshold * len(bts)), sorted=False).indices
            mask_bts = torch.zeros_like(bts)
            mask_bts[top_indices] = bts[top_indices]

            mask_grad_bts = torch.zeros_like(mask_bts)
            mask_grad_bts[top_indices] = 1
            bts_grad_masks = mask_grad_bts.split(self.bt_net_chunk_size)

        mask_bts = mask_bts.split(self.bt_net_chunk_size)

        fedbt_update_bts(self.net, mask_bts, to_params=True)
        bts = fedbt_get_bts(self.net)
        optimizer.add_param_group({'params': bts, 'lr': self.bts_lr})

        # 3. 2 stage training
        trainer = FedP2EFTSeqClsTrainer(
            loss_weights=self.loss_weights,
            bts_masks=bts_grad_masks,
            bts_max_clamp=self.bts_max_clamp,
            model=self.net,
            tokenizer=self.tokenizer,
            optimizers=(optimizer, scheduler),
            args=training_args,
            train_dataset=trainset,
            data_collator=self.data_collator,
        )

        # Stage 1: do local training
        results = trainer.train()

        # mask bts for last optimizer.step()
        for idx, bt in enumerate(bts):
            if bts_grad_masks is not None:
                bts[idx] = torch.clamp(
                    bt, min=1e-4, max=self.bts_max_clamp) * bts_grad_masks[idx]
            else:
                bts[idx] = torch.clamp(bt, min=1e-4, max=self.bts_max_clamp)

        # Stage 2: train btnet
        gt_bts = torch.stack(fedbt_get_bts(self.net)).detach().flatten()

        log_bt_losses = []
        for _ in range(self.bt_steps):
            btnet_optimizer.zero_grad()
            bts = self.bt_net(bt_net_input).squeeze()
            if mask_grad_bts is not None:
                bt_loss = torch.sum(torch.abs((bts - gt_bts) * mask_grad_bts))
            else:
                bt_loss = torch.sum(torch.abs((bts - gt_bts)))
            # bt_loss = torch.sum(torch.abs((bts - gt_bts) * mask_grad_bts)) / torch.sum(mask_grad_bts)

            log_bt_losses.append(bt_loss.item())
            bt_loss.backward()
            btnet_optimizer.step()

        results.metrics['mean_bt_loss'] = np.mean(log_bt_losses)

        # 3. load back pretrained weights
        pretrained_params_dict = zip(
            pretrained_state_dict.keys(), pretrained_weights)
        pretrained_state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in pretrained_params_dict})
        set_peft_model_state_dict(self.net, pretrained_state_dict)

        with torch.no_grad():
            bts = self.bt_net(bt_net_input).squeeze()

        results.metrics['mean_bts'] = torch.mean(bts).item()
        return {'mean_bt_loss': np.mean(log_bt_losses),
                'mean_bts': torch.mean(bts).item(),
                'training_loss': results.training_loss,
                # 'bts_threshold': bts_threshold,
                }

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

        results = self.two_stage_finetune(round_config.current_round, trainset, self.num_train_epochs,
                                          self.max_steps, seed=round_config.current_round+int(self.cid))
        # torch.cuda.empty_cache()
        return (
            self.get_parameters(),
            len(trainset),
            results,
        )

    def evaluate_finetune(self, trainset, num_train_epochs, max_steps, bts_grad_masks=None):
        # same as finetune() with learnable bts
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

        mask_bts = fedbt_get_bts(self.net)
        inner_optimizer = torch.optim.AdamW(
            [p for p in self.net.parameters()], lr=self.task_lr, weight_decay=0)
        inner_optimizer.add_param_group(
            {'params': mask_bts, 'lr': self.bts_lr})
        scheduler = get_constant_schedule(inner_optimizer)
        loss_weights = {'sparsity': 0, 'significance': 0, 'task': 1.}

        trainer = FedP2EFTSeqClsTrainer(
            loss_weights=loss_weights,
            bts_masks=bts_grad_masks,
            bts_max_clamp=self.bts_max_clamp,
            model=self.net,
            tokenizer=self.tokenizer,
            optimizers=(inner_optimizer, scheduler),
            args=training_args,
            train_dataset=trainset,
            data_collator=self.data_collator,
        )

        results = trainer.train()
        for idx, bt in enumerate(mask_bts):
            if bts_grad_masks is not None:
                mask_bts[idx] = torch.clamp(
                    bt, min=1e-4, max=self.bts_max_clamp) * bts_grad_masks[idx]
            else:
                mask_bts[idx] = torch.clamp(
                    bt, min=1e-4, max=self.bts_max_clamp)

        return results

    def evaluate(self, parameters, round_config):
        # print(f"evaluate() on client cid={self.cid}")
        round_config = AttrDict(round_config)
        self.set_parameters(parameters)

        num_train_epochs = round_config.num_train_epochs if 'num_train_epochs' in round_config else self.num_train_epochs
        max_steps = round_config.max_steps if 'max_steps' in round_config else self.max_steps
        seed = round_config.seed if 'seed' in round_config else round_config.current_round

        # 0. Save pretrained model weights to load later
        pretrained_state_dict = get_peft_model_state_dict(
            self.net, save_embedding_layers=False)
        pretrained_weights = [val.cpu().to(torch.float32).numpy()
                              for val in pretrained_state_dict.values()]

        # 1. Compute bts
        trainset = self.get_dataset(data_pool='client',
                                    partition='train',
                                    cid=self.cid)

        trainset = trainset.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file=False,
        )

        fedbt_update_bts(self.net, torch.zeros(self.no_of_adapter_modules))
        bt_net_input = fedl2p_precompute_feats_stats(
            self.net, self.tokenizer, trainset, template=None)

        if self.include_quantity:
            quantity = torch.log(torch.tensor([len(trainset)])).unsqueeze(
                0).to(bt_net_input.device)
            bt_net_input = torch.cat([bt_net_input, quantity], dim=1)

        with torch.no_grad():
            self.bt_net = self.bt_net.to('cuda', dtype=torch.bfloat16)
            bt_net_input = bt_net_input.to('cuda', dtype=torch.bfloat16)
            bts = self.bt_net(bt_net_input).squeeze()
            mask_bts = bts
            if self.eval_threshold:
                top_indices = torch.topk(
                    bts, int(self.eval_threshold * len(bts)), sorted=False).indices
                mask_bts = torch.zeros_like(bts)
                mask_bts[top_indices] = bts[top_indices] if not self.eval_rank_only else 1.
                if self.eval_train_bts:
                    # bts is learnable, hence, generate mask for bts gradients
                    mask_grad_bts = torch.zeros_like(mask_bts)
                    mask_grad_bts[top_indices] = 1
                    bts_grad_masks = mask_grad_bts.split(
                        self.bt_net_chunk_size)
            mask_bts = mask_bts.split(self.bt_net_chunk_size)

        # 2. finetuning before evaluation
        if max_steps > 0 or num_train_epochs > 0:
            # get sub-dataset based on total number of samples needed for finetuning
            if max_steps > 0:
                num_of_samples_needed = min(
                    self.batch_size * self.gradient_accumulation_steps * max_steps, len(trainset))
                trainset = trainset.shuffle(seed=seed).select(
                    range(num_of_samples_needed))

            if self.eval_train_bts:
                fedbt_update_bts(self.net, mask_bts, to_params=True)
                _ = self.evaluate_finetune(
                    trainset, num_train_epochs, max_steps, bts_grad_masks=bts_grad_masks)
            else:
                # fedbt_update_bts(self.net, mask_bts)
                fedbt_update_bts(self.net, mask_bts,
                                 to_params=False, trim_model=True)
                _ = self.finetune(trainset, num_train_epochs,
                                  max_steps, lr=self.task_lr)

        # 2. evaluation
        # create model for evaluation
        net_config = self.config.models.net
        arch_fn = get_func_from_config(net_config)
        if self.load_local_layers:
            net = arch_fn(self.cid, **net_config.args,
                          train=False, tokenizer=self.tokenizer)
        else:
            net = arch_fn(**net_config.args, train=False,
                          tokenizer=self.tokenizer)

        # getting ft weights and setting weights to eval model
        trained_state_dict = get_peft_model_state_dict(
            self.net, save_embedding_layers=False)
        weights = [val.cpu().numpy() for val in trained_state_dict.values()]

        fedbt_update_bts(net, mask_bts, to_params=False, trim_model=True)
        peft_state_dict_keys = get_peft_model_state_dict(
            net, save_embedding_layers=False).keys()
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

        # 3. load back pretrained weights
        arch_fn = get_func_from_config(self.net_config)
        if self.load_local_layers:
            self.net = arch_fn(
                self.cid, **self.net_config.args, tokenizer=self.tokenizer)
        else:
            self.net = arch_fn(**self.net_config.args,
                               tokenizer=self.tokenizer)
        pretrained_params_dict = zip(
            pretrained_state_dict.keys(), pretrained_weights)
        pretrained_state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in pretrained_params_dict})
        set_peft_model_state_dict(self.net, pretrained_state_dict)

        return 0., 0, metrics  # server expects loss, num of samples, and dict
