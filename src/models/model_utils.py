import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from typing import List
from collections import OrderedDict, defaultdict
from transformers import TrainerCallback
import math

import logging
logger = logging.getLogger(__name__)


def torch_clamp(x):
    return torch.clamp(x, min=0, max=1)


def set_weights(model: torch.nn.ModuleList, weights: fl.common.NDArrays) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.Tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def set_weights_multiple_models(nets: list, parameters: fl.common.NDArrays) -> None:
    # a list of nets and a list of parameters to load in order
    for net in nets:
        assert len(net.state_dict().keys()) <= len(
            parameters), f'Insufficient parameters to load {type(net).__name__}.'
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=True)
        parameters = parameters[len(net.state_dict().keys()):]


def get_updates(original_weights: fl.common.NDArrays, updated_weights: fl.common.NDArrays) -> List[torch.Tensor]:
    # extract the updates given two weights
    return [torch.from_numpy(np.copy(up)) - torch.from_numpy(np.copy(op)) for up, op in zip(updated_weights, original_weights)]


def apply_updates(original_weights: fl.common.NDArrays, updates: List[torch.Tensor]) -> fl.common.NDArrays:
    # apply updates to original weights
    return [np.copy(op) + up.cpu().detach().numpy() for up, op in zip(updates, original_weights)]


class Step(torch.autograd.Function):
    def __init__(self):
        super(Step, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return (input > 0.).long().float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def clampSTE_max(input, min_limit=0., max_limit=1.):
    return ClampSTE.apply(input, min_limit, max_limit)


class ClampSTE(torch.autograd.Function):
    def __init__(self):
        super(ClampSTE, self).__init__()

    @staticmethod
    def forward(ctx, input, min_limit=0., max_limit=1.):
        return torch.clamp(input, min=min_limit, max=max_limit)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def update_mean_and_var(m_x, v_x, N, m_y, v_y, M):
    if M == 1:
        var = v_x
    else:
        var1 = ((N - 1) * v_x + (M - 1) * v_y) / (N + M - 1)
        var2 = (N * M * ((m_x - m_y) ** 2)) / ((N+M)*(N+M-1))
        var = var1 + var2
    mean = (N*m_x + M*m_y) / (N+M)

    return mean, var, N+M


def fedl2p_precompute_feats_stats(model, tokenizer, trainset, template, eps=1e-5):
    model = model.to('cuda')
    model.eval()
    feat_stats = {}

    def set_hook(name):
        if name not in feat_stats:
            feat_stats[name] = defaultdict(float)

        def hook(m, inp, outp):
            inp = inp[0]
            # print(inp.size(), inp.numel())

            mean = inp.mean()
            var = inp.var(unbiased=True)
            n = inp.numel()

            with torch.no_grad():
                feat_stats[name]['running_mean'], \
                    feat_stats[name]['running_var'], \
                    feat_stats[name]['total_size'] = update_mean_and_var(feat_stats[name]['running_mean'],
                                                                         feat_stats[name]['running_var'],
                                                                         feat_stats[name]['total_size'],
                                                                         mean,
                                                                         var,
                                                                         n)

        return hook

    hooks = {}
    i = 0
    for mod_name, m in model.named_modules():
        if isinstance(m, nn.Linear) and 'lora' not in mod_name and 'original_module' not in mod_name:
            hooks[i] = m.register_forward_hook(set_hook(i))
            i += 1

    def _prompt_format(example):
        example['instruction'] = template.format(
            example["instruction"], "", "")[:-1]
        return example

    if template is not None:
        iter_trainset = trainset.map(_prompt_format).iter(batch_size=4)
    else:
        iter_trainset = trainset.iter(batch_size=4)

    with torch.no_grad():
        for data_point in iter_trainset:
            if 'instruction' in data_point:
                input_ids = tokenizer(
                    data_point["instruction"], return_tensors="pt", padding=True).to('cuda')
            else:
                input_ids = {'input_ids': torch.tensor(data_point['input_ids']).to('cuda'),
                             'attention_mask': torch.tensor(data_point['attention_mask']).to('cuda')}
                if 'token_type_ids' in data_point:
                    input_ids['token_type_ids'] = torch.tensor(
                        data_point['token_type_ids']).to('cuda')
                if 'labels' in data_point and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(
                        labels=torch.tensor(data_point["labels"], dtype=torch.int64).to('cuda'))
                    input_ids['decoder_input_ids'] = decoder_input_ids

            _ = model(**input_ids)

    for h in hooks.values():
        h.remove()

    lr_net_input = None
    for stats in feat_stats.values():
        if 'running_mean' in stats:
            l_stats = torch.cat([stats['running_mean'].view(
                1), torch.sqrt(stats['running_var'].view(1) + eps)])
        else:
            continue

        if lr_net_input is None:
            lr_net_input = l_stats
        else:
            lr_net_input = torch.cat([lr_net_input, l_stats])
    return lr_net_input.view(1, -1)


class AdaLoRA_Callback(TrainerCallback):
    def on_optimizer_step(self, args, state, control, **kwargs):
        # print(state.global_step, kwargs['model'].base_model.peft_config['default'].total_step - kwargs['model'].base_model.peft_config['default'].tfinal)
        kwargs['model'].base_model.update_and_allocate(state.global_step)
        return super().on_optimizer_step(args, state, control, **kwargs)


def set_adalora_steps(tinit_percentage, tfinal_percentage, **client_kwargs):
    model = client_kwargs['model']
    ds = client_kwargs['dataset']
    batch_size = client_kwargs['batch_size']
    grad_accum_steps = client_kwargs['gradient_accumulation_steps']
    num_train_epochs = client_kwargs['num_train_epochs']
    max_steps = client_kwargs['max_steps']
    if max_steps > 0:
        total_step = max_steps
    else:
        total_step = math.ceil((len(ds) / batch_size) /
                               grad_accum_steps) * num_train_epochs
    model.base_model.peft_config["default"].total_step = total_step
    model.base_model.peft_config["default"].tinit = int(
        tinit_percentage * total_step)
    model.base_model.peft_config["default"].tfinal = int(
        tfinal_percentage * total_step)
    # print(total_step, int(tinit_percentage * total_step), int(tfinal_percentage * total_step))
