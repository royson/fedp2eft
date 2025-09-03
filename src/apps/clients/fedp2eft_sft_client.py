import flwr as fl
import os
import torch
import math
from torch import nn
from copy import deepcopy
import numpy as np
from collections import OrderedDict, defaultdict
from config import AttrDict
from src.utils import get_func_from_config
import evaluate as llm_evaluate
from src.data import Meteor
from tqdm import tqdm

from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments, get_constant_schedule
from src.apps.clients.fedp2eft_trainers import FedP2EFTSFTTrainer
from src.apps.clients import SFTClient
from src.models.weight_net import BTNet, weights_init
from src.models.model_utils import fedl2p_precompute_feats_stats
from src.models.btlora.layer import fedbt_update_bts, fedbt_get_bts, fedbt_get_input_output_sizes


class FedP2EFTSFTClient(SFTClient):
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
        pretrained_weights = [val.cpu().numpy()
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
            self.net, self.tokenizer, trainset, self.prompt_template)

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
            # print(f'seed: {seed}, choice r: {r_c}, bts_threshold: {bts_threshold}')

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
        trainer = FedP2EFTSFTTrainer(
            loss_weights=self.loss_weights,
            bts_masks=bts_grad_masks,
            bts_max_clamp=self.bts_max_clamp,
            model=self.net,
            tokenizer=self.tokenizer,
            optimizers=(optimizer, scheduler),
            args=training_args,
            max_seq_length=self.config.models.tokenizer.args.seq_length,
            train_dataset=trainset,
            formatting_func=self.formatting_prompts_func,
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
        # pdb.set_trace()

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

        # remove samples that exceed the model max length
        trainset = trainset.filter(lambda x: len(self.tokenizer(self.prompt_template.format(x['instruction'], '', ''))['input_ids'])
                                   <= self.tokenizer.model_max_length)
        assert len(trainset) > 1

        # get sub-dataset based on total number of samples needed for finetuning
        max_steps = round_config.max_steps if 'max_steps' in round_config else self.max_steps
        if max_steps > 0:
            num_of_samples_needed = min(
                self.batch_size * self.gradient_accumulation_steps * self.max_steps, len(trainset))
            trainset = trainset.shuffle(seed=round_config.current_round).select(
                range(num_of_samples_needed))

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
        )

        mask_bts = fedbt_get_bts(self.net)
        inner_optimizer = torch.optim.AdamW(
            [p for p in self.net.parameters()], lr=self.task_lr, weight_decay=0)
        inner_optimizer.add_param_group(
            {'params': mask_bts, 'lr': self.bts_lr})
        scheduler = get_constant_schedule(inner_optimizer)
        loss_weights = {'sparsity': 0, 'significance': 0, 'task': 1.}

        trainer = FedP2EFTSFTTrainer(
            loss_weights=loss_weights,
            bts_masks=bts_grad_masks,
            bts_max_clamp=self.bts_max_clamp,
            model=self.net,
            tokenizer=self.tokenizer,
            optimizers=(inner_optimizer, scheduler),
            args=training_args,
            max_seq_length=1024,
            train_dataset=trainset,
            formatting_func=self.formatting_prompts_func,
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
        pretrained_weights = [val.cpu().numpy()
                              for val in pretrained_state_dict.values()]

        # 1. Compute bts
        trainset = self.get_dataset(data_pool='client',
                                    partition='train',
                                    cid=self.cid)

        # remove samples that exceed the model max length
        trainset = trainset.filter(lambda x: len(self.tokenizer(self.prompt_template.format(x['instruction'], '', ''))['input_ids'])
                                   <= self.tokenizer.model_max_length)
        assert len(trainset) > 1
        fedbt_update_bts(self.net, torch.zeros(self.no_of_adapter_modules))
        bt_net_input = fedl2p_precompute_feats_stats(
            self.net, self.tokenizer, trainset, self.prompt_template)

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
        # metrics
        rouge_metric = llm_evaluate.load('rouge')
        meteor_metric = Meteor()

        # tokenizer
        tokenizer_config = self.config.models.tokenizer
        arch_fn = get_func_from_config(tokenizer_config)
        _eval_tokenizer_args = {
            k: v for k, v in tokenizer_config.args.items() if k != 'seq_length'}
        # padding set to left during generate
        _eval_tokenizer_args = {**_eval_tokenizer_args, 'padding_side': 'left'}
        tokenizer = arch_fn(**_eval_tokenizer_args)

        # create model for evaluation
        net_config = self.config.models.net
        arch_fn = get_func_from_config(net_config)
        net = arch_fn(**net_config.args, train=False,
                      tokenizer=tokenizer)  # no quantization

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

        # static kv cache
        if 'cache_implementation' in self.config.app.evaluate_fn.generate_args \
                and self.config.app.evaluate_fn.generate_args.cache_implementation == 'static':
            net.forward = torch.compile(
                net.forward, mode="reduce-overhead", fullgraph=True)

        # local test data
        testset = self.get_dataset(data_pool='client',
                                   partition='test',
                                   cid=self.cid)

        metrics = defaultdict(float)
        metrics['languages'] = set()

        def _prompt_format(example):
            example['instruction'] = self.prompt_template.format(
                example["instruction"], "", "")[:-1]
            example['response'] = example['response'].strip()
            return example

        iter_testset = testset.map(_prompt_format).iter(
            batch_size=self.ckp.config.app.evaluate_fn.batch_size)

        for data_point in tqdm(iter_testset, total=math.ceil(len(testset)/self.ckp.config.app.evaluate_fn.batch_size), desc=f"Client {self.cid} Eval"):
            input_ids = tokenizer(
                data_point["instruction"], return_tensors="pt", padding=True).to(net.device)
            output_ids = net.generate(
                **input_ids, **self.config.app.evaluate_fn.generate_args, pad_token_id=tokenizer.eos_token_id)
            output_ids = output_ids[:, len(input_ids['input_ids'][0]):]
            preds = tokenizer.batch_decode(
                output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]

            result_rouge = rouge_metric.compute(predictions=preds,
                                                references=data_point['response'],
                                                rouge_types=[
                                                    'rouge1', 'rouge2', 'rougeL'],
                                                use_aggregator=False)
            result_meteor = meteor_metric.compute(
                predictions=preds, references=data_point['response'])

            for language, rouge1, rouge2, rougeL, meteor in \
                zip(data_point['language'],
                    result_rouge['rouge1'],
                    result_rouge['rouge2'],
                    result_rouge['rougeL'],
                    result_meteor['meteor']):
                metrics['languages'].add(language)
                metrics[f'{language}_count'] += 1
                metrics[f'{language}_rouge1'] += rouge1
                metrics[f'{language}_rouge2'] += rouge2
                metrics[f'{language}_rougeL'] += rougeL
                metrics[f'{language}_meteor'] += meteor

        # 3. load back pretrained weights
        arch_fn = get_func_from_config(self.net_config)
        self.net = arch_fn(**self.net_config.args, tokenizer=self.tokenizer)
        pretrained_params_dict = zip(
            pretrained_state_dict.keys(), pretrained_weights)
        pretrained_state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in pretrained_params_dict})
        set_peft_model_state_dict(self.net, pretrained_state_dict)

        return 0., 0, metrics  # server expects loss, num of samples, and dict
