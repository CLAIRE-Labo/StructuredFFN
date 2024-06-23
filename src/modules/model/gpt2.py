"""A fast version of gpt2 with flash attention and transformer engine"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import transformer_engine.pytorch as te

from transformer_engine.pytorch.jit import (
    set_jit_fusion_options,
)
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from ..layer import CustomLinear
from ..mlp import FusedMLP
from ..op import bias_dropout_add_impl, RotaryEmbedding, apply_rotary_pos_emb


layernorm_func = {
    "layernorm": te.LayerNorm,
    "rmsnorm": te.RMSNorm,
}


class InferenceParams:

    def __init__(self, max_batch_size, max_sequence_length):
        self.max_sequence_length = max_sequence_length
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict = {}


class TransformerLayer(nn.Module):

    def __init__(self, config, layer_number):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.attn_dim = config.attn_dim
        self.ffn_dim = config.ffn_dim
        self.num_q_heads = config.num_q_heads
        assert self.attn_dim % self.num_q_heads == 0
        self.head_dim = self.attn_dim // self.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.layer_number = layer_number
        self.hidden_dropout = config.hidden_drop

        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        set_jit_fusion_options()
        self.ln1 = layernorm_func[config.ln](self.hidden_dim)
        self.qkv_linear = nn.Linear(
            self.hidden_dim,
            (self.num_q_heads + 2 * self.num_kv_heads) * self.head_dim,
            bias=config.bias,
        )
        self.o_linear = CustomLinear(
            self.attn_dim,
            self.hidden_dim,
            bias=config.bias,
            return_bias=True,
        )
        self.ln2 = layernorm_func[config.ln](self.hidden_dim)
        self.mlp = FusedMLP(
            self.hidden_dim, self.ffn_dim, bias=config.bias, act=config.act
        )

    def _bias_dropout_add(self, hidden_state, bias, residual):
        bias_dropout_add_func = bias_dropout_add_impl(self.training)
        output = bias_dropout_add_func(
            (hidden_state, bias), residual, self.hidden_dropout
        )
        return output

    def _adjust_key_value_for_inference(
        self, inference_params, k_out, v_out, rotary_pos_emb
    ):
        if inference_params is None:
            return k_out, v_out, rotary_pos_emb
        bs = k_out.shape[0]
        seq_len = k_out.shape[1]

        inference_key_memory, inference_value_memory = (
            inference_params.key_value_memory_dict[self.layer_number]
        )
        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + bs
        assert batch_end <= inference_key_memory.size(0)
        sequence_start = inference_params.sequence_len_offset
        sequence_end = sequence_start + seq_len
        assert sequence_end <= inference_key_memory.size(1)
        inference_key_memory[
            batch_start:batch_end, sequence_start:sequence_end, ...
        ] = k_out
        inference_value_memory[
            batch_start:batch_end, sequence_start:sequence_end, ...
        ] = v_out
        key = inference_key_memory[batch_start:batch_end, :sequence_end, ...]
        value = inference_value_memory[batch_start:batch_end, :sequence_end, ...]

        # adjust the key rotary positional embedding
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            q_pos_emb = q_pos_emb[sequence_start:sequence_end, :, :, :]
            k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

        return key, value, rotary_pos_emb

    def forward(
        self,
        hidden_states,
        inference_params: Optional[InferenceParams] = None,
        use_cache=False,
        rotary_pos_emb: torch.Tensor = None,
    ):
        hidden_states = hidden_states.contiguous()
        bs, seq_len, _ = hidden_states.shape
        qkv_out = self.qkv_linear(self.ln1(hidden_states))
        q_out = qkv_out[..., : (self.num_q_heads * self.head_dim)]
        kv_out = qkv_out[..., (self.num_q_heads * self.head_dim) :]
        k_out, v_out = kv_out.chunk(2, -1)
        q_out = q_out.reshape(bs, seq_len, self.num_q_heads, self.head_dim)
        k_out = k_out.reshape(bs, seq_len, self.num_kv_heads, self.head_dim)
        v_out = v_out.reshape(bs, seq_len, self.num_kv_heads, self.head_dim)

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        k_out, v_out, rotary_pos_emb = self._adjust_key_value_for_inference(
            inference_params, k_out, v_out, rotary_pos_emb
        )
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            q_out = apply_rotary_pos_emb(
                q_out,
                q_pos_emb,
            )
            k_out = apply_rotary_pos_emb(
                k_out,
                k_pos_emb,
            )
        softmax_scale = q_out.shape[-1] ** (-0.5)
        if self.scale_attn_by_inverse_layer_idx:
            softmax_scale /= float(self.layer_number + 1)
        if not use_cache:
            attention_out = flash_attn_func(
                q_out, k_out, v_out, softmax_scale=softmax_scale, causal=True
            ).reshape(bs, seq_len, self.attn_dim)
        else:
            attention_out = flash_attn_with_kvcache(
                q_out,
                k_out,
                v_out,
                softmax_scale=softmax_scale,
                cache_seqlens=inference_params.sequence_len_offset,
                causal=True,
            ).reshape(bs, seq_len, self.attn_dim)

        attention_out, attention_bias = self.o_linear(attention_out)
        hidden_states = self._bias_dropout_add(
            attention_out, attention_bias, hidden_states
        )
        ln2_out = self.ln2(hidden_states)
        fc2_out, fc2_bias = self.mlp(ln2_out)
        hidden_states = self._bias_dropout_add(fc2_out, fc2_bias, hidden_states)
        return hidden_states


class BasicGPT2(nn.Module):

    def __init__(
        self,
    ):
        super().__init__()

    @torch.no_grad()
    def _init_weights(self, module, init_config):
        """initialize the weight"""
        if init_config.weight_init == "fixed":
            initializer_range = init_config.initializer_range
            if isinstance(module, (nn.Linear, CustomLinear)):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
            elif isinstance(module, (nn.LayerNorm, te.LayerNorm, te.RMSNorm)):
                if hasattr(module, "bias"):
                    module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        else:
            raise NotImplementedError


class GPT2Model(BasicGPT2):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_dim
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        if config.pos_emb.name == "absolute":
            self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        elif config.pos_emb.name == "rope":
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=config.attn_dim // config.num_q_heads,
                rotary_interleaved=config.pos_emb.rotary_interleaved,
                seq_len_interpolation_factor=config.pos_emb.seq_len_interpolation_factor,
                rotary_base=config.pos_emb.rotary_base,
            )
        else:
            raise NotImplementedError
        self.drop = nn.Dropout(config.embd_drop)
        self.layers = nn.ModuleList(
            [TransformerLayer(config, i) for i in range(config.num_layers)]
        )
        self.ln_f = layernorm_func[config.ln](self.embed_dim)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inference_params: Optional[InferenceParams] = None,
        use_cache=False,
    ):
        bs, seq = input_ids.shape
        seq_start = (
            inference_params.sequence_len_offset
            if use_cache and inference_params is not None
            else 0
        )
        seq_end = seq_start + seq

        position_ids = (
            torch.arange(seq_start, seq_end, dtype=torch.long, device=input_ids.device)
            .unsqueeze(0)
            .view(-1, seq)
        )
        inputs_embeds = self.wte(input_ids)
        if self.config.pos_emb.name == "absolute":
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds
        hidden_states = self.drop(hidden_states)
        if self.config.pos_emb.name == "rope":
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, hidden_states
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
        else:
            rotary_pos_emb = None
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, inference_params, use_cache, rotary_pos_emb
            )
        hidden_states = self.ln_f(hidden_states)
        return hidden_states


class GPT2LMHeadModel(BasicGPT2):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GPT2Model(config)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        # init weight
        self.apply(
            lambda module: self._init_weights(module=module, init_config=config.init)
        )
        # tie weight embedding
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.wte.weight

    @staticmethod
    def get_ckpt_name(config_model):
        return (
            "h"
            + f"{config_model.hidden_dim}"
            + "a"
            + f"{config_model.attn_dim}"
            + "f"
            + f"{config_model.ffn_dim}"
            + "nkv"
            + f"{config_model.num_kv_heads}"
            + f"{config_model.act}"
            + f"{config_model.pos_emb.name}"
            + f"{config_model.ln}"
        )

    def get_flops(self, bs, seq_len):
        attn_qo = 2 * bs * seq_len * self.config.attn_dim * self.config.hidden_dim
        attn_kv = (
            2
            * bs
            * seq_len
            * (self.config.attn_dim // self.config.num_q_heads)
            * self.config.num_kv_heads
            * self.config.hidden_dim
        )
        sdp = 2 * bs * seq_len * seq_len * self.config.attn_dim
        return (
            2 * self.config.num_layers * (attn_qo + attn_kv + sdp)
            + self.get_flops_mlp(bs, seq_len)
            + 2 * bs * seq_len * self.config.vocab_size * self.config.hidden_dim
        )

    def get_params(
        self,
    ):
        attn_qo = 2 * self.config.attn_dim * self.config.hidden_dim
        attn_kv = (
            2
            * (self.config.attn_dim // self.config.num_q_heads)
            * self.config.num_kv_heads
            * self.config.hidden_dim
        )
        return (
            self.config.num_layers * (attn_qo + attn_kv)
            + self.get_params_mlp()
            + self.config.vocab_size * self.config.hidden_dim
        )

    def get_params_woembedding(
        self,
    ):
        attn_qo = 2 * self.config.attn_dim * self.config.hidden_dim
        attn_kv = (
            2
            * (self.config.attn_dim // self.config.num_q_heads)
            * self.config.num_kv_heads
            * self.config.hidden_dim
        )
        return self.config.num_layers * (attn_qo + attn_kv) + self.get_params_mlp()

    def get_flops_mlp(self, bs, seq):
        # as they're all linear layers. The flops just scales with the parameters
        mlp = 0
        for layer in self.model.layers:
            for para in layer.mlp.parameters():
                if len(para.shape) != 1:
                    mlp += para.numel()
        return 2 * mlp * bs * seq

    def get_params_mlp(
        self,
    ):
        mlp = 0
        for layer in self.model.layers:
            for para in layer.mlp.parameters():
                if len(para.shape) != 1:
                    mlp += para.numel()
        return mlp

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        labels: Optional[torch.LongTensor] = None,
        inference_params: Optional[InferenceParams] = None,
        use_cache=False,
    ):
        out = self.model(input_ids, inference_params, use_cache)
        lm_logits = self.lm_head(out)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                lm_logits.view(-1, lm_logits.size(-1)).contiguous(),
                labels.view(-1).contiguous(),
            )
        if loss is not None:
            return loss
        else:
            return lm_logits

    def prepare_inference_params(
        self, batch_size, mx_seq, torch_dtype=torch.bfloat16, device="cuda"
    ):
        # mx_seq is composed of prefill and generation length
        inference_params = InferenceParams(batch_size, mx_seq)
        inf_max_seq_len = inference_params.max_sequence_length
        inf_max_batch_size = inference_params.max_batch_size
        for i in range(self.config.num_layers):
            inference_key_memory = torch.empty(
                inf_max_batch_size,
                inf_max_seq_len,
                self.config.num_kv_heads,
                (self.config.attn_dim // self.config.num_q_heads),
                dtype=torch_dtype,
                device=device,
            )
            inference_value_memory = torch.empty(
                inf_max_batch_size,
                inf_max_seq_len,
                self.config.num_kv_heads,
                (self.config.attn_dim // self.config.num_q_heads),
                dtype=torch_dtype,
                device=device,
            )
            inference_params.key_value_memory_dict[i] = (
                inference_key_memory,
                inference_value_memory,
            )
        return inference_params
