# Part of this code is taken and modified from https://github.com/huggingface/transformers
#
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import time
import sys
import os
import math
import shutil
import warnings
import numpy as np
from transformers.integrations import hp_params

import torch
import torch.nn as nn
import torch.distributed as dist
from trl import SFTTrainer
from transformers import Trainer
import datasets
from accelerate.state import PartialState
from accelerate import skip_first_batches
from torch.utils.data import DataLoader, RandomSampler
from transformers.trainer import TRAINER_STATE_NAME, _is_peft_model
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.utils import (
    is_datasets_available,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
    is_apex_available,
)
from transformers.trainer_utils import (
    seed_worker,
    has_length,
    HPSearchBackend,
    speed_metrics,
    TrainOutput,
    set_seed,
    find_executable_batch_size,
    get_last_checkpoint,
    enable_full_determinism,
)
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.trainer_callback import TrainerState, ExportableState
from transformers.trainer_pt_utils import get_model_param_count, LengthGroupedSampler
from typing import Optional, Union, Any, List, Dict

from src.apps.clients import FedL2PLearner
from src.data import cycle

import logging
logger = logging.getLogger(__name__)

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

if is_apex_available():
    from apex import amp

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class FedL2PTrainer(Trainer):
    '''
    Wrapper around hf Trainer, inheriting all its attributes and methods
    Includes a validation dataset for outer-loop optimization
    Includes FedL2P's LRNet and learnable LRs (inner_loop_lrs), along with the optimizers
    Overwrites training pipeline to include bi-level optimization using learn2learn
    '''

    def __init__(self,
                 args=None,
                 tokenizer=None,
                 val_dataset=None,
                 lr_net=None,
                 lr_net_input=None,
                 inner_loop_lrs=None,
                 fedl2p_inner_optimizer=None,
                 fedl2p_meta_optimizers=None,
                 hypergrad=None,
                 outer_loop_steps=3,
                 outer_loop_grad_accum=4,
                 **kwargs):
        super().__init__(args=args,
                         tokenizer=tokenizer,
                         **kwargs)

        assert val_dataset is not None, 'a validation set must be provided in FedL2PSFTTrainer'

        self.outer_loop_steps = outer_loop_steps
        self.lr_net = lr_net
        self.lr_net_input = lr_net_input
        self.inner_loop_lrs = inner_loop_lrs
        self.fedl2p_meta_optimizers = fedl2p_meta_optimizers
        self.fedl2p_inner_optimizer = fedl2p_inner_optimizer
        self.hypergrad = hypergrad
        self.outer_loop_grad_accum = outer_loop_grad_accum
        self.val_dataset = val_dataset

    def _get_val_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.val_dataset is None or not has_length(self.val_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.val_dataset, datasets.Dataset):
                lengths = (
                    self.val_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.val_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[
                0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.val_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(self.val_dataset)

    def get_train_dataloader(self, batch_size=None) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training")

        dataloader_params = {
            "batch_size": batch_size if batch_size is not None else self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_val_dataloader(self, batch_size=None) -> DataLoader:
        if self.val_dataset is None:
            raise ValueError(
                "FedL2PSFTTrainer: training requires a val_dataset.")

        val_dataset = self.val_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(val_dataset, datasets.Dataset):
            val_dataset = self._remove_unused_columns(
                val_dataset, description="validation")
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="validation")

        dataloader_params = {
            # use same batch size as train if batch size is not provided
            "batch_size": batch_size if batch_size is not None else self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(val_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_val_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(val_dataset, **dataloader_params))

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Adopted and modified from Trainer to enable meta-training.
        See https://github.com/huggingface/transformers/blob/ac5a0556f14dec503b064d5802da1092e0b558ea/src/transformers/trainer.py
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train and not self.is_model_parallel:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(
                f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(
                self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            state = TrainerState.load_from_json(os.path.join(
                resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None:
                self._train_batch_size = state.train_batch_size

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        if args.push_to_hub:
            raise NotImplementedError()

        ##

        fedl2p_model = FedL2PLearner(model=self.model,
                                     optimizer=self.fedl2p_inner_optimizer,
                                     gradient_accumulation_steps=self.args.gradient_accumulation_steps)

        train_loader = iter(cycle(self.get_train_dataloader(batch_size=1)))
        val_loader = iter(cycle(self.get_val_dataloader(batch_size=1)))

        self.lr_net = self.lr_net.to(device=args.device, dtype=torch.bfloat16)
        self.lr_net_input = self.lr_net_input.to(
            args.device, dtype=torch.bfloat16)
        if type(self.inner_loop_lrs) == nn.ParameterList:
            self.inner_loop_lrs = self.inner_loop_lrs.to(
                args.device, dtype=torch.bfloat16)
        train_losses = []
        val_losses = []
        for ous in range(self.outer_loop_steps):
            for optimizer in self.fedl2p_meta_optimizers.values():
                optimizer.zero_grad()

            learner = copy.deepcopy(fedl2p_model)
            learner.train()

            # forward pass LRNet
            lrnet_lrs = self.lr_net(self.lr_net_input).squeeze()

            # inner loop training
            res = inner_training_loop(
                learner,
                lrnet_lrs,
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
            train_losses.append(res.training_loss)

            # outer-loop: compute hypergrad and update meta params
            meta_params = [p for p in self.lr_net.parameters()]
            if type(self.inner_loop_lrs) == nn.ParameterList:
                meta_params += [p for p in self.inner_loop_lrs]

            step_val_losses = []
            if self.hypergrad is None:
                # FOMAML+
                for outer_loop_idx in range(self.outer_loop_grad_accum):
                    v_i = next(val_loader)
                    outer_loss = self.meta_training_step(learner, v_i)
                    hypergrads = torch.autograd.grad(outer_loss,
                                                     meta_params,
                                                     allow_unused=True,
                                                     retain_graph=(outer_loop_idx + 1 != self.outer_loop_grad_accum))
                    outer_loss = outer_loss / self.outer_loop_grad_accum
                    step_val_losses.append(outer_loss.item())

                    for p, g in zip(meta_params, hypergrads):
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad = p.grad + g
            else:
                # IFT w Neumann approx
                learner_params = [
                    p for p in learner.parameters() if p.requires_grad]
                for outer_loop_idx in range(self.outer_loop_grad_accum):
                    t_i = next(train_loader)
                    # print(t_i['input_ids'].size(), v_i['input_ids'].size())
                    v_i = next(val_loader)
                    inner_loss = self.meta_training_step(learner, t_i)
                    outer_loss = self.meta_training_step(learner, v_i)

                    outer_loss = outer_loss / self.outer_loop_grad_accum
                    inner_loss = inner_loss / self.outer_loop_grad_accum
                    step_val_losses.append(outer_loss.item())

                    hypergrads = self.hypergrad.grad(outer_loss,
                                                     inner_loss,
                                                     meta_params,
                                                     learner_params,
                                                     retain_graph=(outer_loop_idx + 1 != self.outer_loop_grad_accum))

                    for p, g in zip(meta_params, hypergrads):
                        if p.grad is None:
                            p.grad = g
                        else:
                            p.grad = p.grad + g

            val_losses.append(np.sum(step_val_losses))

            torch.nn.utils.clip_grad_norm_(
                self.lr_net.parameters(), self.args.max_grad_norm)
            if type(self.inner_loop_lrs) == nn.ParameterList:
                torch.nn.utils.clip_grad_norm_(
                    self.inner_loop_lrs, self.args.max_grad_norm)

            for optimizer in self.fedl2p_meta_optimizers.values():
                optimizer.step()

        # logging
        with torch.no_grad():
            lrnet_lrs = self.lr_net(self.lr_net_input).squeeze()
            if type(self.inner_loop_lrs) == nn.ParameterList:
                effective_lrs = [(lr1 * lr2).item()
                                 for lr1, lr2 in zip(lrnet_lrs, self.inner_loop_lrs)]
            else:
                effective_lrs = [
                    lr1.item() * self.inner_loop_lrs for lr1 in lrnet_lrs]
            sparsity = sum(x == 0 for x in effective_lrs) / len(effective_lrs)

        results = {
            'training_loss': np.mean(train_losses),
            'val_loss': np.mean(val_losses),
            'lr_sparsity': sparsity,
            'mean_lrnet_lr': torch.mean(lrnet_lrs).item(),
        }

        if type(self.inner_loop_lrs) == nn.ParameterList:
            results['mean_inner_loop_lr'] = np.mean(
                [p.item() for p in self.inner_loop_lrs])

        return results

    def _inner_training_loop(
        self, learner, lrnet_lrs, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        '''
        self.optimizer and self.scheduler is not used!
        '''
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        assert not self.args.auto_find_batch_size, 'auto_find_batch_size is not implemented in FedL2PSFTTrainer'
        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * \
            args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(
                            train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(
                    args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(
                    train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(
                        train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(
                    train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(learner)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled(
        ) or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(
                    max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            learner.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        learner = self._wrap_model(learner)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        # use_accelerator_prepare = True if learner is self.model else False
        use_accelerator_prepare = True

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                learner = self.accelerator.prepare(learner)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            learner.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    learner = self.accelerator.prepare(learner)
                else:
                    learner, self.optimizer = self.accelerator.prepare(
                        learner, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                learner, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    learner, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        # if self.is_fsdp_enabled:
        #     learner = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        # if model is not self.model:
        #     self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = learner

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    learner, resume_from_checkpoint, load_module_strict=not _is_peft_model(
                        learner)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, learner)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # learner is the Transformers Model

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Number of trainable parameters = {get_model_param_count(learner, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step //
                                 num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = learner
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        learner.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(
                    epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(
                        learner, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current learner is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the learner class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += (
                            torch.sum(
                                self.accelerator.gather(
                                    torch.tensor(
                                        inputs[main_input_name].numel(), device=self.args.device, dtype=torch.int64
                                    )
                                )
                            )
                            .cpu()
                            .item()
                        )
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(
                        args, self.state, self.control)

                # with self.accelerator.accumulate(learner):
                # grad accumulation is done in training_step instead
                tr_loss_step = self.training_step(learner,
                                                  inputs,
                                                  step=step,
                                                  lrnet_lrs=lrnet_lrs,
                                                  last_step=(step == len(epoch_iterator) - 1 and epoch == num_train_epochs - 1))

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / \
                        (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (
                        step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    #     # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    #     # in accelerate. So, explicitly enable sync gradients to True in that case.
                    #     if is_last_step_and_steps_less_than_grad_acc:
                    #         self.accelerator.gradient_state._set_sync_gradients(True)

                    #     # Gradient clipping
                    #     if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    #         # deepspeed does its own clipping

                    #         if is_sagemaker_mp_enabled() and args.fp16:
                    #             _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                    #         elif self.use_apex:
                    #             # Revert to normal clipping otherwise, handling Apex or full precision
                    #             _grad_norm = nn.utils.clip_grad_norm_(
                    #                 amp.master_params(self.optimizer),
                    #                 args.max_grad_norm,
                    #             )
                    #         else:
                    #             _grad_norm = self.accelerator.clip_grad_norm_(
                    #                 learner.parameters(),
                    #                 args.max_grad_norm,
                    #             )

                    #         if (
                    #             is_accelerate_available()
                    #             and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                    #         ):
                    #             grad_norm = learner.get_global_grad_norm()
                    #             # In some cases the grad norm may not return a float
                    #             if hasattr(grad_norm, "item"):
                    #                 grad_norm = grad_norm.item()
                    #         else:
                    #             grad_norm = _grad_norm

                    #     self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                    #     # self.optimizer.step()

                    #     # self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                    #     # optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    #     # if optimizer_was_run:
                    #     #     # Delay optimizer scheduling until metrics are generated
                    #     #     if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    #     #         self.lr_scheduler.step()

                    #     learner.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + \
                        (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(
                        args, self.state, self.control)

                #     self._maybe_log_save_evaluate(tr_loss, grad_norm, learner, trial, epoch, ignore_keys_for_eval)
                # else:
                self.control = self.callback_handler.on_substep_end(
                    args, self.state, self.control)

                # if self.control.should_epoch_stop or self.control.should_training_stop:
                #     # PyTorch/XLA relies on the data loader to insert the mark_step for
                #     # each step. Since we are breaking the loop early, we need to manually
                #     # insert the mark_step here.
                #     if is_torch_xla_available():
                #         xm.mark_step()
                #     break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control)
            self._maybe_log_save_evaluate(
                tr_loss, grad_norm, learner, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        # Avoid ZeroDivisionError
        effective_global_step = max(self.state.global_step, 0.001)
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(learner)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], step, lrnet_lrs, last_step=False) -> torch.Tensor:
        # print(f"TRAINING STEP: {step}")
        model.train()
        # if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
        #     self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            # with torch.autocast(device_type="cuda"):
            loss = self.compute_loss(model, inputs)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        # kwargs = {}

        # # For LOMO optimizers you need to explicitly use the learnign rate
        # if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
        #     kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # if self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
            # self.accelerator.backward(loss, **kwargs)
        # if last_step:
        #     print("LAST STEP!!!!!!")

        # we are handling the gradient accumulation now
        loss = loss / self.args.gradient_accumulation_steps

        model.adapt(loss,
                    step,
                    lrnet_lrs=lrnet_lrs,
                    learnable_lrs=self.inner_loop_lrs,
                    track_grads=last_step,
                    clip_grad=self.args.max_grad_norm)

        return loss.detach()

    def meta_training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        # if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
        #     self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            # with torch.autocast(device_type="cuda"):
            loss = self.compute_loss(model, inputs)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        return loss


class FedL2PSFTTrainer(FedL2PTrainer, SFTTrainer):
    def __init__(self, *args, formatting_func=None, **kwargs):
        super().__init__(*args, formatting_func=formatting_func, **kwargs)
        with PartialState().local_main_process_first():
            self.val_dataset = self._prepare_dataset(
                self.val_dataset,
                self.tokenizer,
                self.args.packing,
                self.args.dataset_text_field,
                self.args.max_seq_length,
                formatting_func,
                self.args.num_of_sequences,
                self.args.chars_per_token,
                remove_unused_columns=self.args.remove_unused_columns if args is not None else True,
                **self.args.dataset_kwargs,
            )
