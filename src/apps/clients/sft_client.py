import flwr as fl
import os
import torch
import math
from collections import OrderedDict, defaultdict
from src.log import Checkpoint
from config import AttrDict
from trl import DataCollatorForCompletionOnlyLM
from src.utils import get_func_from_config
import evaluate as llm_evaluate
from src.data import Meteor
from tqdm import tqdm

from peft import get_peft_model_state_dict, set_peft_model_state_dict
from peft.utils.peft_types import PeftType
from transformers import TrainingArguments, get_constant_schedule
from trl import SFTTrainer
from src.apps.clients.fedp2eft_trainers import FedP2EFTSFTTrainer
from src.models.btlora.layer import fedbt_update_bts, fedbt_get_bts, fedbt_get_input_output_sizes


class SFTClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        ckp: Checkpoint,
        num_train_epochs: float = 0.,  # overrides by max_step if max_step > 0
        max_steps: int = 10,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        logging_steps: int = 100,
        save_strategy: str = 'no',
        bt_args=None,
        **kwargs,
    ):
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
        self.net = arch_fn(**self.net_config.args, tokenizer=self.tokenizer)

        # prompt formatting for llama models
        prompt_func = get_func_from_config(
            self.config.models.prompt.formatting_prompt_func)
        formatting_prompts_func, response_template = prompt_func(
            self.tokenizer.eos_token)
        response_template_ids = self.tokenizer.encode(
            response_template, add_special_tokens=False)[2:]
        self.formatting_prompts_func = formatting_prompts_func
        self.data_collator = DataCollatorForCompletionOnlyLM(
            response_template_ids, tokenizer=self.tokenizer)
        self.prompt_template = get_func_from_config(
            self.config.models.prompt.prompt_template)

        # data class
        data_config = self.config.data
        data_class = get_func_from_config(data_config)
        self.data_class = data_class(self.ckp, **data_config.args)
        self.get_dataset = self.data_class.get_dataset

        # fixed hyperparameters (hyperparameters conditioned on the round, e.g. lr,
        #                        are implemented on the server and passed to client in `round_config`)
        self.num_train_epochs = num_train_epochs
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # additional SFT training kwargs
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
        state_dict = get_peft_model_state_dict(
            self.net, save_embedding_layers=False)
        return [val.cpu().numpy() for _, val in state_dict.items()]

    def set_parameters(self, parameters):
        peft_state_dict_keys = get_peft_model_state_dict(
            self.net, save_embedding_layers=False).keys()
        params_dict = zip(peft_state_dict_keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        set_peft_model_state_dict(self.net, state_dict)

    def bt_finetune(self, num_train_epochs, max_steps, lr, seed):
        pretrained_state_dict = get_peft_model_state_dict(
            self.net, save_embedding_layers=False)
        pretrained_weights = [val.cpu().to(torch.float32).numpy()
                              for val in pretrained_state_dict.values()]

        self.net.train()
        trainset = self.get_dataset(data_pool='client',
                                    partition='train',
                                    cid=self.cid)

        trainset = trainset.filter(lambda x: len(self.tokenizer(self.prompt_template.format(x['instruction'], '', ''))['input_ids'])
                                   <= self.tokenizer.model_max_length)

        # get sub-dataset based on total number of samples needed for finetuning
        if max_steps > 0:
            num_of_samples_needed = min(
                self.batch_size * self.gradient_accumulation_steps * max_steps, len(trainset))
            trainset = trainset.shuffle(seed=seed).select(
                range(num_of_samples_needed))

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
        trainer = FedP2EFTSFTTrainer(
            loss_weights=self.bt_args.loss_weights,
            bts_masks=None,  # no masks
            bts_max_clamp=self.bt_args.bts_max_clamp,
            model=self.net,
            tokenizer=self.tokenizer,
            optimizers=(optimizer, scheduler),
            args=training_args,
            max_seq_length=self.config.models.tokenizer.args.seq_length,
            train_dataset=trainset,
            formatting_func=self.formatting_prompts_func,
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
        trainer = SFTTrainer(
            model=self.net,
            tokenizer=self.tokenizer,
            optimizers=(optimizer, scheduler),
            args=training_args,
            max_seq_length=self.config.models.tokenizer.args.seq_length,
            train_dataset=trainset,
            formatting_func=self.formatting_prompts_func,
            data_collator=self.data_collator,
            callbacks=self.callbacks,
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

        # remove samples that exceed the model max length
        trainset = trainset.filter(lambda x: len(self.tokenizer(self.prompt_template.format(x['instruction'], '', ''))['input_ids'])
                                   <= self.tokenizer.model_max_length)
        assert len(trainset) > 1

        # get sub-dataset based on total number of samples needed for finetuning
        if self.max_steps > 0:
            num_of_samples_needed = min(
                self.batch_size * self.gradient_accumulation_steps * self.max_steps, len(trainset))
            trainset = trainset.shuffle(seed=round_config.current_round).select(
                range(num_of_samples_needed))

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
        if max_steps > 0 or num_train_epochs > 0:
            trainset = self.get_dataset(data_pool='client',
                                        partition='train',
                                        cid=self.cid)

            # remove samples that exceed the model max length
            trainset = trainset.filter(lambda x: len(self.tokenizer(self.prompt_template.format(x['instruction'], '', ''))['input_ids'])
                                       <= self.tokenizer.model_max_length)
            assert len(trainset) > 1

            # get sub-dataset based on total number of samples needed for finetuning
            if max_steps > 0:
                num_of_samples_needed = min(
                    self.batch_size * self.gradient_accumulation_steps * max_steps, len(trainset))
                trainset = trainset.shuffle(seed=seed).select(
                    range(num_of_samples_needed))

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
                fedbt_update_bts(self.net, mask_bts,
                                 to_params=False, trim_model=True)

            _ = self.finetune(trainset, num_train_epochs,
                              max_steps, round_config.lr)

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

        # create model for evaluation if quantization was used for training
        if self.net_config.args.quantization:
            arch_fn = get_func_from_config(self.net_config)
            net = arch_fn(**self.net_config.args,
                          train=False, tokenizer=tokenizer)

            if self.net.peft_config['default'].peft_type == PeftType.ADALORA:
                # get peft state dict manually without trimming ranks
                # singular values for ranks that are not picked are already 0, hence, safe to merge
                state_dict = self.net.state_dict()
                state_dict = {k: state_dict[k]
                              for k in state_dict if "lora_" in k}
                state_dict = {k: v for k, v in state_dict.items() if (
                    ("lora_" in k and 'default' in k) or ("bias" in k))}
                weights = [val.cpu().numpy() for _, val in state_dict.items()]
            else:
                # getting ft weights and setting weights to eval model
                weights = self.get_parameters()
            if self.bt_args:
                fedbt_update_bts(
                    net, mask_bts, to_params=False, trim_model=True)
            peft_state_dict_keys = get_peft_model_state_dict(
                net, save_embedding_layers=False).keys()
            params_dict = zip(peft_state_dict_keys, weights)
            state_dict = OrderedDict({k: torch.Tensor(v)
                                     for k, v in params_dict})
            set_peft_model_state_dict(net, state_dict)
        else:
            net = self.net

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

        return 0., 0, metrics  # server expects loss, num of samples, and dict
