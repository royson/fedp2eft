import pickle
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from flwr.common import parameters_to_ndarrays
from peft.utils import prepare_model_for_kbit_training
from src.models.btlora.layer import fedbt_convert_lora_btlora
from trl import get_kbit_device_map
from collections import OrderedDict
import numpy as np

from peft import (
    LoraConfig,
    AdaLoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)


def get_llm_tokenizer(model_name_or_path, cache_dir, padding_side, seq_length=None, use_fast=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              use_fast=use_fast,
                                              padding_side=padding_side,
                                              model_max_length=seq_length,
                                              cache_dir=cache_dir)

    if tokenizer.pad_token is None and tokenizer.unk_token is None:
        # unk token not found. add special pad token
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def get_llm_local_tokenizer(cid, local_tokenizer_path, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
    return tokenizer


def get_peft_config(model, adapter, adapter_args):
    peft_config = None
    if 'target_modules' in adapter_args and adapter_args['target_modules'] == 'all':
        adapter_args['target_modules'] = [name for name, layer in model.named_modules()
                                          if isinstance(layer, nn.Linear) or isinstance(layer, nn.Embedding)]

    if adapter == 'lora':
        peft_config = LoraConfig(
            **adapter_args,
        )
    elif adapter == 'adalora':
        peft_config = AdaLoraConfig(
            **adapter_args
        )
    else:
        # TODO
        raise NotImplementedError()

    return peft_config


def get_causal_llm_model(model_name_or_path,
                         cache_dir,
                         train=True,
                         gradient_checkpointing=True,
                         quantization=None,
                         attn_implementation='sdpa',
                         device_map='cuda',):

    low_cpu_mem_usage = None
    quant_config = None
    if quantization is not None:
        if quantization == 8:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            low_cpu_mem_usage = True
        elif quantization == 4:
            quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                              bnb_4bit_compute_dtype=torch.bfloat16,
                                              bnb_4bit_quant_type="nf4")
            low_cpu_mem_usage = True
        else:
            raise NotImplementedError()

    if train:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quant_config if quant_config else None,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            use_cache=False,  # kv cache
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=get_kbit_device_map() if quant_config else device_map,
        )

        if quantization is not None:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=gradient_checkpointing
            )
    else:  # inference
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            use_cache=True,  # kv cache
            attn_implementation='sdpa',  # use default flashattention for inference
            device_map=device_map,
        )
    return model


def get_causal_peft_llm_model(model_name_or_path,
                              adapter,
                              adapter_args,
                              cache_dir,
                              train=True,
                              gradient_checkpointing=True,
                              quantization=None,
                              attn_implementation='sdpa',
                              device_map='cuda',
                              seed=42,
                              bt=False,
                              tokenizer=None):
    model = get_causal_llm_model(model_name_or_path,
                                 cache_dir,
                                 train=train,
                                 gradient_checkpointing=gradient_checkpointing,
                                 quantization=quantization,
                                 attn_implementation=attn_implementation,
                                 device_map=device_map)

    if tokenizer is not None and model.model.embed_tokens.weight.size(0) != len(tokenizer):
        # special tokens are added
        model.resize_token_embeddings(len(tokenizer))
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    peft_config = get_peft_config(model, adapter=adapter,
                                  adapter_args=adapter_args)

    torch.manual_seed(seed)  # fixed model initialization
    model = get_peft_model(model, peft_config)

    if bt:
        assert adapter == 'lora', 'bt is only supported with lora'
        # convert all lora to btlora
        fedbt_convert_lora_btlora(model)

    return model


def get_seq_class_llm_model(model_name_or_path,
                            config_args,
                            cache_dir,
                            train=True,
                            gradient_checkpointing=False,
                            quantization=None,
                            attn_implementation='sdpa',
                            device_map='cuda',
                            **kwargs):

    low_cpu_mem_usage = None
    quant_config = None
    if quantization is not None:
        if quantization == 8:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            low_cpu_mem_usage = True
        elif quantization == 4:
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            low_cpu_mem_usage = True
        else:
            raise NotImplementedError()

    label_list = config_args['label_list']
    finetuning_task = config_args['finetuning_task']
    hf_config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(label_list),
        id2label={str(i): label for i, label in enumerate(label_list)},
        label2id={label: i for i, label in enumerate(label_list)},
        finetuning_task=finetuning_task,
        cache_dir=cache_dir,
    )

    if train:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            quantization_config=quant_config if quant_config else None,
            from_tf=False,
            config=hf_config,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=get_kbit_device_map() if quant_config else device_map,
        )

        if quantization is not None:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=gradient_checkpointing
            )
    else:  # inference
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            from_tf=False,
            config=hf_config,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
            attn_implementation='sdpa',  # use default flashattention for inference
            device_map=device_map,
        )
    return model


def get_seq_class_peft_llm_model(model_name_or_path,
                                 adapter,
                                 adapter_args,
                                 config_args,
                                 cache_dir,
                                 train=True,
                                 gradient_checkpointing=False,
                                 quantization=None,
                                 attn_implementation='sdpa',
                                 device_map='cuda',
                                 seed=42,
                                 bt=False,
                                 **kwargs):

    model = get_seq_class_llm_model(model_name_or_path,
                                    config_args,
                                    cache_dir,
                                    train=train,
                                    gradient_checkpointing=gradient_checkpointing,
                                    quantization=quantization,
                                    attn_implementation=attn_implementation,
                                    device_map=device_map)

    peft_config = get_peft_config(model, adapter=adapter,
                                  adapter_args=adapter_args)

    torch.manual_seed(seed)  # fixed model initialization
    model = get_peft_model(model, peft_config)

    if bt:
        assert adapter == 'lora', 'bt is only supported with lora'
        # convert all lora to btlora
        fedbt_convert_lora_btlora(model)

    return model


def get_seq_class_dept_peft_llm_model(
    cid,
    local_embeds_path,
    global_body_path,
    model_name_or_path,
    adapter,
    adapter_args,
    config_args,
    cache_dir,
    train=True,
    gradient_checkpointing=False,
    quantization=None,
    attn_implementation='sdpa',
    device_map='cuda',
    seed=42,
    bt=False,
    tokenizer=None,
):

    local_embeds_path = local_embeds_path.format(cid=str(cid))
    with open(global_body_path, 'rb') as f:
        body_weights = parameters_to_ndarrays(pickle.load(f))

    model = get_seq_class_llm_model(model_name_or_path,
                                    config_args,
                                    cache_dir,
                                    train=train,
                                    gradient_checkpointing=gradient_checkpointing,
                                    quantization=quantization,
                                    attn_implementation=attn_implementation,
                                    device_map=device_map)

    if model.bert.embeddings.word_embeddings.weight.size(0) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    assert len([k for k in model.state_dict().keys()
               if '_embeddings' not in k]) == len(body_weights)
    params_dict = zip([k for k in model.state_dict().keys()
                      if '_embeddings' not in k], body_weights)
    state_dict = OrderedDict(
        {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=False)

    embed_sd = torch.load(local_embeds_path)
    model.load_state_dict(embed_sd, strict=False)

    peft_config = get_peft_config(model, adapter=adapter,
                                  adapter_args=adapter_args)

    torch.manual_seed(seed)  # fixed model initialization
    model = get_peft_model(model, peft_config)

    if bt:
        assert adapter == 'lora', 'bt is only supported with lora'
        # convert all lora to btlora
        fedbt_convert_lora_btlora(model)

    return model


def get_seq_class_feddpa_peft_llm_model(
        cid,
        local_adapter_path,
        local_lora_args,
        model_name_or_path,
        adapter,
        adapter_args,
        config_args,
        cache_dir,
        train=True,
        gradient_checkpointing=False,
        quantization=None,
        attn_implementation='sdpa',
        device_map='cuda',
        seed=42,
        bt=False,
        **kwargs):

    model = get_seq_class_llm_model(model_name_or_path,
                                    config_args,
                                    cache_dir,
                                    train=train,
                                    gradient_checkpointing=gradient_checkpointing,
                                    quantization=quantization,
                                    attn_implementation=attn_implementation,
                                    device_map=device_map)

    # load client's specific local lora weights
    local_adapter_path = local_adapter_path.format(cid=str(cid))
    local_peft_config = LoraConfig(**local_lora_args)
    model = get_peft_model(model, local_peft_config)

    local_state_dict = torch.load(local_adapter_path)
    set_peft_model_state_dict(model, local_state_dict)
    model = model.merge_and_unload()

    peft_config = get_peft_config(model, adapter=adapter,
                                  adapter_args=adapter_args)

    torch.manual_seed(seed)  # fixed model initialization
    model = get_peft_model(model, peft_config)

    if bt:
        assert adapter == 'lora', 'bt is only supported with lora'
        # convert all lora to btlora
        fedbt_convert_lora_btlora(model)

    return model
