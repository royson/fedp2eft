# Part of this code is taken and modified from https://github.com/huggingface/peft
#
# Copyright 2023-present the HuggingFace Inc. team.
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
import torch
from torch import nn
from typing import Any, Tuple, Literal
from peft.utils.other import transpose
from functools import partial

from peft.tuners.lora.layer import Linear as loraLinear
from peft.tuners.lora.layer import Embedding as loraEmbedding
from peft.tuners.lora.bnb import Linear8bitLt, Linear4bit


def sparsity_loss(bts: Tuple[torch.Tensor],
                  reduction: Literal['sum', 'mean'] = 'sum',
                  eps=0):
    total_loss = 0.
    for bt in bts:
        total_loss += torch.sum(100 * (bt+eps) + torch.log(bt+eps))
    if reduction == 'mean':
        total_loss = total_loss / len(bts)

    return total_loss


def significance_loss(personalized_model: nn.Module,
                      init_model: nn.Module,
                      bts: Tuple[torch.Tensor],
                      adapter: str = 'default',
                      reduction: Literal['sum', 'mean'] = 'sum',
                      eps=0,
                      ):
    iter_bts = iter(bts)
    total_loss = 0.
    for p_m, i_m in zip(personalized_model.modules(), init_model.modules()):
        if hasattr(p_m, 'update_bt'):
            bt = next(iter_bts).squeeze()
            if isinstance(p_m, loraLinear) or isinstance(p_m, Linear8bitLt) or isinstance(p_m, Linear4bit):
                total_loss += torch.sum(
                    torch.abs(p_m.lora_B[adapter].weight -
                              i_m.lora_B[adapter].weight) / (bt+eps)
                )
            elif isinstance(p_m, loraEmbedding):
                total_loss += torch.sum(
                    torch.abs(
                        p_m.lora_embedding_B[adapter] - i_m.lora_embedding_B[adapter]) / (bt+eps)
                )

    if reduction == 'mean':
        total_loss = total_loss / len(bts)

    return total_loss


def fedbt_convert_lora_btlora(model):
    for m in model.modules():
        if isinstance(m, loraLinear):
            lora_linear_to_btlora_linear(m)
        elif isinstance(m, Linear8bitLt):
            lora_8bitlinear_to_btlora_8bitlinear(m)
        elif isinstance(m, Linear4bit):
            lora_4bitlinear_to_btlora_4bitlinear(m)
        elif isinstance(m, loraEmbedding):
            lora_embedding_to_btlora_embedding(m)


def fedbt_get_input_output_sizes(model):
    i_size = 0
    o_size = 0
    for mod_name, m in model.named_modules():
        if isinstance(m, loraLinear) or \
                isinstance(m, loraEmbedding) or \
                isinstance(m, Linear8bitLt) or \
                isinstance(m, Linear4bit):
            o_size += 1
        elif 'lora' not in mod_name and 'original_module' not in mod_name and isinstance(m, nn.Linear):
            i_size += 1
    i_size *= 2  # mean & std

    return i_size, o_size


def fedbt_update_bts(model: nn.Module,
                     bts: Tuple[torch.Tensor],
                     to_params=False,
                     trim_model=False):
    iter_bts = iter(bts)
    for m in model.modules():
        if hasattr(m, 'update_bt') and callable(m.update_bt):
            bt = next(iter_bts).squeeze()
            if to_params:
                bt = nn.Parameter(bt).detach()
                bt.requires_grad = True
            m.update_bt(bt, trim_model=trim_model)


def fedbt_get_bts(model: nn.Module):
    bts = []
    for m in model.modules():
        if hasattr(m, 'update_bt') and callable(m.update_bt):
            assert m.bt is not None
            bts.append(m.bt)
    return bts


def lora_linear_to_btlora_linear(lora_module):
    '''
    Override functions in hf lora linear
    '''
    lora_module.bt = None

    def update_bt(self, bt, trim_model=False):
        if not trim_model:
            self.bt = bt
        else:
            non_zero_indices = (bt != 0).nonzero().squeeze(1)
            self.bt = bt[bt.nonzero()].squeeze(1)
            r = len(self.bt)
            for active_adapter in self.active_adapters:
                if r == 0:
                    del self.lora_A[active_adapter]
                    del self.lora_B[active_adapter]
                    del self.lora_dropout[active_adapter]
                    del self.scaling[active_adapter]
                elif r < self.lora_A[active_adapter].weight.size(0):
                    lora_A_weight = self.lora_A[active_adapter].weight[non_zero_indices]
                    lora_B_weight = self.lora_B[active_adapter].weight[:,
                                                                       non_zero_indices]
                    self.lora_A[active_adapter].weight = torch.nn.Parameter(
                        lora_A_weight)
                    self.lora_A[active_adapter].out_features = r
                    self.lora_B[active_adapter].weight = torch.nn.Parameter(
                        lora_B_weight)
                    self.lora_B[active_adapter].in_features = r

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise NotImplementedError()
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                if not self.use_dora or not self.use_dora[active_adapter]:
                    result = result + \
                        lora_B(self.bt * lora_A(dropout(x))) * scaling
                else:
                    raise NotImplementedError()
                    x = dropout(x)
                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                    )

            result = result.to(torch_result_dtype)

        return result

    def get_delta_weight(self, adapter) -> torch.Tensor:
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        cast_to_fp32 = device.type == "cpu" and (
            dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            self.bt = self.bt.float()
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(
            (self.bt * weight_B) @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)
            self.bt = self.bt.to(dtype)

        return output_tensor

    lora_module.update_bt = partial(update_bt, lora_module)
    lora_module.forward = partial(forward, lora_module)
    lora_module.get_delta_weight = partial(get_delta_weight, lora_module)


def lora_embedding_to_btlora_embedding(lora_module):
    '''
    Override functions in hf lora embedding
    '''
    lora_module.bt = None

    def update_bt(self, bt, trim_model=False):
        assert not trim_model, 'TODO: trimming is not supported for embedding atm'
        self.bt = bt

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise NotImplementedError()
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_embedding_A:
                    continue
                embedding_A = self.lora_embedding_A[active_adapter].T
                embedding_B = self.lora_embedding_B[active_adapter].T
                scaling = self.scaling[active_adapter]

                if not self.use_dora or not self.use_dora[active_adapter]:
                    after_A = self._embed(x, (self.bt * embedding_A))
                    result = result + (after_A @ embedding_B) * scaling
                else:
                    raise NotImplementedError()
                    mag_norm_scale, dora_result = self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=embedding_A,
                        lora_B=embedding_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        embed_fn=self._embed,
                    )
                    result = mag_norm_scale * result + dora_result
            result = result.to(torch_result_dtype)

        return result

    def get_delta_weight(self, adapter) -> torch.Tensor:
        device = self.lora_embedding_B[adapter].device
        dtype = self.lora_embedding_A[adapter].dtype

        cast_to_fp32 = device.type == "cpu" and (
            dtype == torch.float16 or dtype == torch.bfloat16)

        weight_A = self.lora_embedding_A[adapter]
        weight_B = self.lora_embedding_B[adapter]

        if cast_to_fp32:
            self.bt = self.bt.float()
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(
            (self.bt * weight_B) @ weight_A, True) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_embedding_A[adapter] = weight_A.to(dtype)
            self.lora_embedding_B[adapter] = weight_B.to(dtype)
            self.bt = self.bt.to(dtype)

        return output_tensor

    lora_module.update_bt = partial(update_bt, lora_module)
    lora_module.forward = partial(forward, lora_module)
    lora_module.get_delta_weight = partial(get_delta_weight, lora_module)


def lora_8bitlinear_to_btlora_8bitlinear(lora_module):
    '''
    Override functions in hf lora linear8bitlt
    '''
    lora_module.bt = None

    def update_bt(self, bt, trim_model=False):
        if not trim_model:
            self.bt = bt
        else:
            non_zero_indices = (bt != 0).nonzero().squeeze(1)
            self.bt = bt[bt.nonzero()].squeeze(1)
            r = len(self.bt)
            for active_adapter in self.active_adapters:
                if r == 0:
                    del self.lora_A[active_adapter]
                    del self.lora_B[active_adapter]
                    del self.lora_dropout[active_adapter]
                    del self.scaling[active_adapter]
                elif r < self.lora_A[active_adapter].weight.size(0):
                    lora_A_weight = self.lora_A[active_adapter].weight[non_zero_indices]
                    lora_B_weight = self.lora_B[active_adapter].weight[:,
                                                                       non_zero_indices]
                    self.lora_A[active_adapter].weight = torch.nn.Parameter(
                        lora_A_weight)
                    self.lora_A[active_adapter].out_features = r
                    self.lora_B[active_adapter].weight = torch.nn.Parameter(
                        lora_B_weight)
                    self.lora_B[active_adapter].in_features = r

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise NotImplementedError()
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    compute_dtype = lora_A.weight.dtype
                    if x.dtype != compute_dtype:
                        x = x.to(compute_dtype)

                if not self.use_dora[active_adapter]:
                    result = result + \
                        lora_B(self.bt * lora_A(dropout(x))) * scaling
                else:
                    raise NotImplementedError()
                    if isinstance(dropout, torch.nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None

                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )
                if requires_conversion:
                    result = result.to(expected_dtype)

        return result

    def get_delta_weight(self, adapter):
        return (
            transpose(
                (self.bt *
                 self.lora_B[adapter].weight) @ self.lora_A[adapter].weight,
                False,
            )
            * self.scaling[adapter]
        )

    lora_module.update_bt = partial(update_bt, lora_module)
    lora_module.forward = partial(forward, lora_module)
    lora_module.get_delta_weight = partial(get_delta_weight, lora_module)


def lora_4bitlinear_to_btlora_4bitlinear(lora_module):
    '''
    Override functions in hf lora linear4bit
    '''
    lora_module.bt = None

    def update_bt(self, bt, trim_model=False):
        if not trim_model:
            self.bt = bt
        else:
            non_zero_indices = (bt != 0).nonzero().squeeze(1)
            self.bt = bt[bt.nonzero()].squeeze(1)
            r = len(self.bt)

            for active_adapter in self.active_adapters:
                if r == 0:
                    del self.lora_A[active_adapter]
                    del self.lora_B[active_adapter]
                    del self.lora_dropout[active_adapter]
                    del self.scaling[active_adapter]
                elif r < self.lora_A[active_adapter].weight.size(0):
                    lora_A_weight = self.lora_A[active_adapter].weight[non_zero_indices]
                    lora_B_weight = self.lora_B[active_adapter].weight[:,
                                                                       non_zero_indices]
                    self.lora_A[active_adapter].weight = torch.nn.Parameter(
                        lora_A_weight)
                    self.lora_A[active_adapter].out_features = r
                    self.lora_B[active_adapter].weight = torch.nn.Parameter(
                        lora_B_weight)
                    self.lora_B[active_adapter].in_features = r

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise NotImplementedError()
            result = self._mixed_batch_forward(
                x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            # As per Tim Dettmers, for 4bit, we need to defensively clone here.
            # The reason is that in some cases, an error can occur that backprop
            # does not work on a manipulated view. This issue may be solved with
            # newer PyTorch versions but this would need extensive testing to be
            # sure.
            result = result.clone()

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]

                requires_conversion = not torch.is_autocast_enabled()
                if requires_conversion:
                    expected_dtype = result.dtype
                    x = x.to(lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + \
                        lora_B(self.bt * lora_A(dropout(x))) * scaling
                else:
                    raise NotImplementedError()
                    if isinstance(dropout, torch.nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None

                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )
                if requires_conversion:
                    result = result.to(expected_dtype)

        return result

    def get_delta_weight(self, adapter):
        return (
            transpose(
                (self.bt *
                 self.lora_B[adapter].weight) @ self.lora_A[adapter].weight,
                False,
            )
            * self.scaling[adapter]
        )

    lora_module.update_bt = partial(update_bt, lora_module)
    lora_module.forward = partial(forward, lora_module)
    lora_module.get_delta_weight = partial(get_delta_weight, lora_module)
