# Part of this code is taken and modified from https://github.com/learnables/learn2learn
#
# MIT License

# Copyright(c) 2019 Debajyoti Datta, Ian Bunner, Praateek Mahajan, Sebastien Arnold

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import grad
from learn2learn.algorithms.base_learner import BaseLearner

alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""

llama3_instruct_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}{}
"""


def get_llama3_instruct_formatting_prompts_func(eos_token):
    overall_temp, response_temp = (
        llama3_instruct_template, '<|start_header_id|>assistant<|end_header_id|>\n\n')

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = overall_temp.format(
                example['instruction'][i], example['response'][i], eos_token)
            output_texts.append(text)
        return output_texts

    return formatting_prompts_func, response_temp


def get_alpaca_formatting_prompts_func(eos_token):
    overall_temp, response_temp = (alpaca_template, '\n### Response:')

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = overall_temp.format(
                example['instruction'][i], example['response'][i], eos_token)
            output_texts.append(text)
        return output_texts

    return formatting_prompts_func, response_temp


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def epochs_to_batches(num_epochs, dataset_len, batch_size, drop_last=False):
    if drop_last:
        fb_per_epoch = np.floor(dataset_len / int(batch_size))
    else:
        fb_per_epoch = np.ceil(dataset_len / int(batch_size))
    return int(fb_per_epoch * num_epochs)


def update_module(module, updates=None, memo=None):
    """
    Taken from https://github.com/learnables/learn2learn/blob/master/learn2learn/utils.py
    """
    if memo is None:
        memo = {}
    if updates is not None:
        params = list(module.parameters())
        if not len(updates) == len(list(params)):
            msg = 'WARNING:update_module(): Parameters and updates have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(updates)) + ')'
            print(msg)
        for p, g in zip(params, updates):
            p.update = g

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        cond = False
        if p in memo:
            cond = True
            module._parameters[param_key] = memo[p]
        else:
            if p is not None and hasattr(p, 'update') and p.update is not None:
                cond = True
                updated = p + p.update
                p.update = None
                memo[p] = updated
                module._parameters[param_key] = updated

        if cond and not module._parameters[param_key].requires_grad:
            module._parameters[param_key].requires_grad = True

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff in memo:
            module._buffers[buffer_key] = memo[buff]
        else:
            if buff is not None and hasattr(buff, 'update') and buff.update is not None:
                updated = buff + buff.update
                buff.update = None
                memo[buff] = updated
                module._buffers[buffer_key] = updated

    # Then, recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(
            module._modules[module_key],
            updates=None,
            memo=memo,
        )

    if hasattr(module, 'flatten_parameters'):
        module._apply(lambda x: x)
    return module


class Identity(torch.autograd.Function):
    # just identity, this class is defined for compatibility/scalability reasons
    def __init__(self):
        super(Identity, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Step(torch.autograd.Function):
    def __init__(self):
        super(Step, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return (input > 0.).long().float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SoftArgmax(torch.autograd.Function):
    def __init__(self):
        super(SoftArgmax, self).__init__()

    @staticmethod
    def forward(ctx, input):
        t = torch.argmax(F.softmax(input, dim=0), dim=0, keepdims=True)
        return torch.zeros_like(input).scatter_(0, t, 1.)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def split_weight_decay_parameters(model):
    wd_params = []
    wd_params_names = []
    for n, m in model.named_modules():
        if allow_weight_decay(m):
            wd_params.append(m.weight)
            wd_params_names.append(f'{n}.weight')

    no_wd_params = [p for n, p in model.named_parameters()
                    if n not in wd_params_names]
    assert len(wd_params) + len(no_wd_params) == len(list(model.parameters())
                                                     ), "Sanity check failed."
    return wd_params_names, wd_params, no_wd_params


def allow_weight_decay(module):
    return isinstance(module,
                      (nn.Linear,
                       nn.Conv1d,
                       nn.Conv2d,
                       nn.Conv3d,
                       nn.ConvTranspose1d,
                       nn.ConvTranspose2d,
                       nn.ConvTranspose3d)
                      )


class FedL2PAdamW(torch.nn.Module):
    '''
    AdamW inner-loop with learnable learning rates
    '''

    def __init__(self, betas=(0.9, 0.999), weight_decay=0, eps=1e-08):
        super(FedL2PAdamW, self).__init__()
        self.betas = betas
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.eps = eps

    def forward(self, model,
                learnable_lrs=None,
                lrnet_lrs=None,
                grads=None,
                ):
        if grads is not None:
            assert len(grads) == len(lrnet_lrs)
            if type(learnable_lrs) == nn.ParameterList:
                assert len(grads) == len(learnable_lrs)
                iter_learnable_lrs = iter(learnable_lrs)

            iter_lrnet_lrs = iter(lrnet_lrs)
            iter_grads = iter(grads)

            for (n, p) in model.named_parameters():
                if p.requires_grad:
                    g = next(iter_grads)
                    if type(learnable_lrs) == nn.ParameterList:
                        lr = next(iter_learnable_lrs)
                    else:
                        lr = learnable_lrs
                    lr = lr * next(iter_lrnet_lrs)

                    if n not in self.m:
                        self.m[n] = torch.zeros_like(g)
                        self.v[n] = torch.zeros_like(g)
                    self.m[n] = self.betas[0] * \
                        self.m[n] + (1. - self.betas[0]) * g
                    self.v[n] = self.betas[1] * self.v[n] + \
                        (1. - self.betas[1]) * (g ** 2)
                    m_hat = self.m[n] / (1 - self.betas[0])
                    v_hat = self.v[n] / (1 - self.betas[1])

                    upd = lr * (m_hat / (torch.sqrt(v_hat) + self.eps))
                    p.update = -((lr * self.weight_decay * p) + upd)

        return update_module(model)


class FedL2PSSGD(torch.nn.Module):
    '''
    SGD inner-loop with learnable learning rates as proposed in FedL2P
    '''

    def __init__(self):
        super(FedL2PSSGD, self).__init__()

    def forward(self, model,
                learnable_lrs=None,
                lrnet_lrs=None,
                grads=None,
                ):
        if grads is not None:
            assert len(grads) == len(lrnet_lrs)
            if type(learnable_lrs) == nn.ParameterList:
                assert len(grads) == len(learnable_lrs)
                iter_learnable_lrs = iter(learnable_lrs)

            iter_lrnet_lrs = iter(lrnet_lrs)
            iter_grads = iter(grads)

            for (n, p) in model.named_parameters():
                # print(f'updating {n}')
                if p.requires_grad:
                    g = next(iter_grads)
                    if type(learnable_lrs) == nn.ParameterList:
                        lr = next(iter_learnable_lrs)
                    else:
                        lr = learnable_lrs
                    lr = lr * next(iter_lrnet_lrs)

                    p.update = -lr * g  # minimize

        return update_module(model)


class FedL2PLearner(BaseLearner):
    def __init__(self, model, optimizer, gradient_accumulation_steps, grad_store=[]):
        super(FedL2PLearner, self).__init__()
        self.module = model
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_store = grad_store
        self.inner_loop_update = optimizer

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def adapt(self, loss,
              step,
              lrnet_lrs=None,
              learnable_lrs=None,
              track_grads=False,
              clip_grad=None,
              allow_nograd=True):

        gradients = grad(loss,
                         [p for p in self.module.parameters() if p.requires_grad],
                         retain_graph=False,
                         create_graph=False,
                         allow_unused=allow_nograd)

        if not self.grad_store:
            self.grad_store = gradients
        else:
            self.grad_store = [gs + g for gs,
                               g in zip(self.grad_store, gradients)]

        if (step > 0 and (step + 1) % self.gradient_accumulation_steps == 0) or track_grads:
            if clip_grad is not None and clip_grad > 0:
                norms = []
                norms.extend([torch.linalg.vector_norm(g, 2)
                             for g in self.grad_store])
                total_norm = torch.linalg.vector_norm(
                    torch.stack(norms), 2
                )
                clip_coef = clip_grad / (total_norm + 1e-6)
                clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
                self.grad_store = [g.mul_(clip_coef_clamped)
                                   for g in self.grad_store]

            if not track_grads:
                with torch.no_grad():
                    self.module = self.inner_loop_update(self.module,
                                                         learnable_lrs=learnable_lrs,
                                                         lrnet_lrs=lrnet_lrs,
                                                         grads=self.grad_store,
                                                         )
                # print(f'updated w/o track grads!, mean grad: {torch.mean(torch.stack([torch.mean(g) for g in self.grad_store])).item()}')
            else:
                self.module = self.inner_loop_update(self.module,
                                                     learnable_lrs=learnable_lrs,
                                                     lrnet_lrs=lrnet_lrs,
                                                     grads=self.grad_store,
                                                     )
                # print(f'updated w track grads!, mean grad: {torch.mean(torch.stack([torch.mean(g) for g in self.grad_store])).item()}')
            self.grad_store = []


class Hypergrad:
    """
    Credit: "Optimizing Millions of Hyperparameters by Implicit Differentiation"
    (https://arxiv.org/pdf/1911.02590.pdf)
    """

    def __init__(self, learning_rate=.1, truncate_iter=3, allow_nograd=False):
        self.learning_rate = learning_rate
        self.truncate_iter = truncate_iter
        self.allow_nograd = allow_nograd

    def grad(self, loss_val, loss_train, meta_params, params, retain_graph=False):
        if self.allow_nograd:
            params = [p for p in params if p.requires_grad]

        dloss_val_dparams = torch.autograd.grad(
            loss_val,
            params,
            retain_graph=True,
            allow_unused=True
        )

        dloss_train_dparams = torch.autograd.grad(
            loss_train,
            params,
            allow_unused=True,
            create_graph=True,
        )

        v2 = self._approx_inverse_hvp(
            dloss_val_dparams, dloss_train_dparams, params)

        v3 = torch.autograd.grad(
            dloss_train_dparams,
            meta_params,
            grad_outputs=v2,
            retain_graph=retain_graph,
            allow_unused=True
        )

        return list(-g.detach() for g in v3)

    def _approx_inverse_hvp(self, dloss_val_dparams, dloss_train_dparams, params):
        p = v = dloss_val_dparams

        for _ in range(self.truncate_iter):
            grad = torch.autograd.grad(
                dloss_train_dparams,
                params,
                grad_outputs=v,
                retain_graph=True,
                allow_unused=True
            )

            # scale: this a is key for convergence
            grad = [g * self.learning_rate for g in grad]

            v = [curr_v - curr_g for (curr_v, curr_g) in zip(v, grad)]
            p = [curr_p + curr_v for (curr_p, curr_v) in zip(p, v)]

        return p
