# coding=utf-8
# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Quasar Long model."""

import math
import os
import warnings
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
try:
    from transformers.modeling_attn_mask_utils import (
        _prepare_4d_attention_mask,
        _prepare_4d_causal_attention_mask,
        _prepare_4d_causal_attention_mask_for_sdpa,
    )
except ImportError:
    # transformers 5.x removed these helpers
    def _prepare_4d_attention_mask(mask, dtype, tgt_len=None):
        raise NotImplementedError("_prepare_4d_attention_mask removed in transformers 5.x")
    def _prepare_4d_causal_attention_mask(*args, **kwargs):
        raise NotImplementedError("_prepare_4d_causal_attention_mask removed in transformers 5.x")
    def _prepare_4d_causal_attention_mask_for_sdpa(*args, **kwargs):
        raise NotImplementedError("_prepare_4d_causal_attention_mask_for_sdpa removed in transformers 5.x")
from transformers.modeling_outputs import MoeModelOutputWithPast
try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
except ImportError:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    def dynamic_rope_update(fn):
        return fn
from transformers.modeling_utils import PreTrainedModel
try:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
except ImportError:
    ALL_LAYERNORM_LAYERS = []
    is_torch_greater_or_equal_than_1_13 = True  # torch >= 1.13 is guaranteed in any modern env
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
# is_torch_fx_available was removed in transformers 5.x; define a no-op stub
try:
    from transformers.utils.import_utils import is_torch_fx_available
except ImportError:
    def is_torch_fx_available():
        return False
from .configuration_quasar_long import QuasarLongConfig
from transformers.generation.utils import GenerationMixin
_CONFIG_FOR_DOC = "QuasarLongConfig"
from dataclasses import dataclass
from transformers.utils import ModelOutput
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except Exception:
    LigerFusedLinearCrossEntropyLoss = None

# ── Engram: conditional N-gram memory (DeepSeek-AI, arXiv:2601.07372) ─────────
try:
    import sys as _sys
    import os as _os
    _HERE = _os.path.dirname(_os.path.abspath(__file__))
    if _HERE not in _sys.path:
        _sys.path.insert(0, _HERE)
    _RAVEN_PATH = _os.path.join(_HERE, "raven")
    if _RAVEN_PATH not in _sys.path:
        _sys.path.insert(0, _RAVEN_PATH)
    from engram import EngramModule
    _ENGRAM_AVAILABLE = True
except Exception as _engram_import_err:  # pragma: no cover
    EngramModule = None  # type: ignore[assignment,misc]
    _ENGRAM_AVAILABLE = False
def _debug_assert_finite(name: str, tensor: torch.Tensor, layer_idx: Optional[int] = None):
    return


def _sanitize_hybrid_tensor(name: str, tensor: torch.Tensor, layer_idx: Optional[int] = None):
    return tensor


def roll_tensor(tensor, shifts=-1, dims=-1, fill_value=0):
    """Roll the tensor input along the given dimension(s).
    Inserted elements are set to be 0.0.
    """
    rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
    rolled_tensor.select(dims, shifts).fill_(fill_value)
    return rolled_tensor, rolled_tensor.sum()


@dataclass
class MoEV2CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs as well as Mixture of Expert's router hidden
    states terms, to train a MoE model.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        z_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            z_loss for the sparse modules.
        aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            aux_loss for the sparse modules.
        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

            Router logits of the encoder model, useful to compute the auxiliary loss and the z_loss for the sparse
            modules.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    z_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None
    mtp_loss: Optional[torch.FloatTensor] = None
    mtp_logits: Optional[tuple[torch.FloatTensor, ...]] = None
    branch_past_key_values: Optional["QGRBranchCache"] = None
    branch_mimic_loss: Optional[torch.FloatTensor] = None
    branch_mimic_stats: Optional[dict] = None


class MoeV2ModelOutputWithPast(MoeModelOutputWithPast):

    def __init__(
        self,
        mtp_hidden_states=None,
        branch_past_key_values=None,
        branch_mimic_loss=None,
        branch_mimic_stats=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mtp_hidden_states = mtp_hidden_states
        self.branch_past_key_values = branch_past_key_values
        self.branch_mimic_loss = branch_mimic_loss
        self.branch_mimic_stats = branch_mimic_stats


class QGRBranchCache:
    """Recurrent-state cache for chunked Quasar/GLA/Raven training.

    It intentionally carries only linear/recurrent branch state, not dense GQA
    KV tensors. That lets a multi-million-token logical sequence be processed
    as chunks without allocating a dense multi-million-token attention cache.
    """

    def __init__(self, seen_tokens: int = 0):
        self.seen_tokens = int(seen_tokens)
        self.layers: list[dict] = []
        self.recurrent_states: dict[int, torch.Tensor] = {}
        self.conv_states: dict[int, tuple] = {}

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> dict:
        return self.layers[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        return self.seen_tokens

    def update(self, layer_idx: int, recurrent_state=None, conv_state=None, offset: int = 0, **kwargs):
        layer_idx = int(layer_idx)
        while len(self.layers) <= layer_idx:
            self.layers.append({})
        state = self.layers[layer_idx]
        if recurrent_state is not None:
            state["recurrent_state"] = recurrent_state
            self.recurrent_states[layer_idx] = recurrent_state
        if conv_state is not None:
            state["conv_state"] = conv_state
            self.conv_states[layer_idx] = conv_state
        if offset:
            self.seen_tokens += int(offset)
        return self

    def detach_(self, clone: bool = False) -> "QGRBranchCache":
        def _detach(value):
            if torch.is_tensor(value):
                value = value.detach()
                return value.clone() if clone else value
            if isinstance(value, tuple):
                return tuple(_detach(v) for v in value)
            if isinstance(value, list):
                return [_detach(v) for v in value]
            if isinstance(value, dict):
                return {k: _detach(v) for k, v in value.items()}
            return value

        self.layers = [_detach(layer) for layer in self.layers]
        self.recurrent_states = {k: _detach(v) for k, v in self.recurrent_states.items()}
        self.conv_states = {k: _detach(v) for k, v in self.conv_states.items()}
        return self


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# ── Sample packing: segment isolation under SDPA. Packed rows carry position_ids
# that reset to 0 at each segment start; padding is trailing (attention_mask 1/0).
# These helpers derive the block-diagonal mask + per-segment cu_seqlens from that.
def _quasar_segment_ids(position_ids: torch.Tensor) -> torch.Tensor:
    # 0-based segment index per row, +1 at each position_ids reset.
    starts = (position_ids == 0).to(torch.int32)
    return torch.cumsum(starts, dim=1) - 1


def _quasar_is_packed(position_ids: Optional[torch.Tensor], attention_mask_2d: Optional[torch.Tensor]) -> bool:
    # True iff some row has a second segment (a reset past column 0).
    if position_ids is None:
        return False
    pos = position_ids
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)
    if pos.shape[1] < 2:
        return False
    internal = pos[:, 1:] == 0
    if attention_mask_2d is not None:
        internal = internal & attention_mask_2d[:, 1:].to(torch.bool)  # ignore pad
    return bool(internal.any())


def _quasar_packed_cu_seqlens(
    position_ids: torch.Tensor,
    attention_mask_2d: Optional[torch.Tensor],
) -> torch.Tensor:
    # int32 cu_seqlens over the unpadded row-major token stream, boundary at each
    # segment start. Same order as get_unpad_data so it lines up with branch unpadding.
    pos = position_ids
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)
    B, S = pos.shape
    if attention_mask_2d is not None:
        keep = attention_mask_2d.to(torch.bool)
    else:
        keep = torch.ones(B, S, dtype=torch.bool, device=pos.device)
    flat_keep = keep.reshape(-1)
    flat_pos = pos.reshape(-1)[flat_keep]
    n_tokens = int(flat_pos.numel())
    starts = (flat_pos == 0).nonzero(as_tuple=False).flatten().to(torch.int32)
    end = torch.tensor([n_tokens], dtype=torch.int32, device=pos.device)
    return torch.cat([starts, end])


def _quasar_build_block_diag_sdpa_mask(
    position_ids: torch.Tensor,
    attention_mask_2d: Optional[torch.Tensor],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    # Additive [B,1,S,S] mask: each query attends causally only within its own
    # segment, never to pad keys. Reduces to the plain causal mask for one segment.
    pos = position_ids
    if pos.dim() == 1:
        pos = pos.unsqueeze(0)
    B, S = pos.shape
    seg = _quasar_segment_ids(pos)                       # [B, S]
    q_seg = seg[:, :, None]                              # [B, S, 1]
    k_seg = seg[:, None, :]                              # [B, 1, S]
    same_seg = q_seg == k_seg                            # [B, S, S]
    q_idx = torch.arange(S, device=device)[None, :, None]
    k_idx = torch.arange(S, device=device)[None, None, :]
    causal = k_idx <= q_idx                              # [1, S, S]
    allow = same_seg & causal
    if attention_mask_2d is not None:
        key_keep = attention_mask_2d.to(torch.bool)[:, None, :]   # [B, 1, S]
        allow = allow & key_keep
    # NaN-safety: any query row with no visible key attends itself.
    no_visible = ~allow.any(dim=-1, keepdim=True)        # [B, S, 1]
    allow = allow | (no_visible & (q_idx == k_idx))
    min_val = torch.finfo(dtype).min
    mask = torch.zeros(B, 1, S, S, dtype=dtype, device=device)
    mask = mask.masked_fill(~allow[:, None], min_val)
    return mask


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.QuasarLong.modeling_QuasarLong._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    warnings.warn(
        "Calling `transformers.models.QuasarLong.modeling_QuasarLong._make_causal_mask` is deprecated and will be removed in v4.37. Use `transformers.models.QuasarLong.modeling_QuasarLong.AttentionMaskConverter._make_causal_mask"
    )
    return AttentionMaskConverter._make_causal_mask(
        input_ids_shape=input_ids_shape, dtype=dtype, device=device, past_key_values_length=past_key_values_length
    )


class QuasarLongRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        QuasarLongRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states.to(input_dtype)).to(input_dtype)


class QuasarLongGroupRMSNorm(nn.Module):
    def __init__(self, hidden_size, group_norm_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.group_norm_size = group_norm_size
        assert hidden_size % group_norm_size == 0, "hidden_size must be divisible by group_norm_size"
        self.variance_epsilon = eps

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        input_shape = hidden_states.size()
        group_shape = input_shape[:-1] + (self.group_norm_size, input_shape[-1] // self.group_norm_size)
        hidden_states = hidden_states.view(group_shape).to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states.to(input_dtype).view(input_shape)).to(input_dtype)


ALL_LAYERNORM_LAYERS.append(QuasarLongRMSNorm)


def _quasar_long_safe_nope_enabled(config) -> bool:
    return bool(getattr(config, "use_nope", False)) and getattr(config, "long_context_mode", "") == "rope_short_nope_long"


def _quasar_long_global_nope_enabled(config) -> bool:
    return bool(getattr(config, "use_nope", False)) and not _quasar_long_safe_nope_enabled(config)


class QuasarLongRotaryEmbedding(nn.Module):
    def __init__(self, config: QuasarLongConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        if self.rope_type in ROPE_INIT_FUNCTIONS:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        else:
            # 'default' was removed in transformers 5.x; compute standard RoPE inv_freq inline
            self.rope_init_fn = None
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            dim = int(head_dim * partial_rotary_factor)
            rope_theta = getattr(config, "rope_theta", 10000.0)
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
            self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        if _quasar_long_global_nope_enabled(self.config):
            batch, seq_len = position_ids.shape
            head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
            partial_rotary_factor = getattr(self.config, "partial_rotary_factor", 1.0)
            rotary_dim = int(head_dim * partial_rotary_factor)
            cos = torch.ones(batch, seq_len, rotary_dim, device=x.device, dtype=x.dtype)
            sin = torch.zeros(batch, seq_len, rotary_dim, device=x.device, dtype=x.dtype)
            return cos, sin

        # Auto-recover inv_freq if it contains meta-device or weight-loader garbage values
        if (self.inv_freq.device != x.device or 
            self.inv_freq.ndim == 0 or 
            self.inv_freq.shape[0] == 0 or 
            self.inv_freq[0].item() > 2.0 or 
            (self.inv_freq.shape[0] > 1 and self.inv_freq[1].item() == 0.0)):
            
            print(f"[ROPE DEBUG] Triggered auto-recovery! Current inv_freq device: {self.inv_freq.device}, values: {self.inv_freq[:4]}", flush=True)
            partial_rotary_factor = getattr(self.config, "partial_rotary_factor", 1.0)
            head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
            dim = int(head_dim * partial_rotary_factor)
            rope_theta = getattr(self.config, "rope_theta", 10000.0)
            self.inv_freq = (1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=x.device) / dim))).to(x.device)
            print(f"[ROPE DEBUG] Recovered inv_freq: {self.inv_freq[:4]}", flush=True)
            
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        if _quasar_long_safe_nope_enabled(self.config):
            cutoff = int(getattr(self.config, "nope_after_position", 100485))
            nope_mask = (position_ids >= cutoff).unsqueeze(-1)
            if bool(nope_mask.any()):
                cos = torch.where(nope_mask, torch.ones_like(cos), cos)
                sin = torch.where(nope_mask, torch.zeros_like(sin), sin)
        return cos, sin


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class QuasarLongMLP(nn.Module):
    def __init__(self, config: QuasarLongConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QuasarLongGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts

        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.routed_scaling_factor = config.routed_scaling_factor

        self.register_buffer("expert_bias", torch.zeros((self.num_experts)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def group_limited_topk(
        self,
        scores: torch.Tensor,
    ):
        num_tokens, _ = scores.size()
        # Organize the experts into groups
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Mask the experts based on selection groups
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )

        masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
        probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)

        return probs, top_indices

    def forward(self, hidden_states):
        # compute gating score
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))

        scores = torch.sigmoid(logits.float()).type_as(logits)

        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)

        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)

        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight.type_as(hidden_states), logits


class QuasarLongSparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: QuasarLongConfig, layer_idx: int = -1):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self._setup_experts()
        self.gate = QuasarLongGate(config)
        if config.num_shared_experts is not None:
            self.shared_experts = QuasarLongMLP(
                config=config, intermediate_size=config.moe_intermediate_size * config.num_shared_experts
            )

    def reset_parameters(self) -> None:
        for module in self.children():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()

    def _setup_experts(self):
        self.experts_w12 = nn.Parameter(torch.zeros(self.config.num_experts, self.config.hidden_size, 2 * self.config.moe_intermediate_size))
        self.experts_w3 = nn.Parameter(torch.zeros(self.config.num_experts, self.config.moe_intermediate_size, self.config.hidden_size))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        w12_key = prefix + 'experts_w12'
        w3_key = prefix + 'experts_w3'
        
        # Initialize progressive accumulation buffers on first shard arrival
        if not hasattr(self, '_temp_gate_weights'):
            self._temp_gate_weights = {}
            self._temp_up_weights = {}
            self._temp_down_weights = {}
            
        num_experts = self.config.num_experts
        
        # Intercept and pop any separate expert weights from the active state dict shard
        for k in list(state_dict.keys()):
            if k.startswith(prefix + 'experts.'):
                parts = k[len(prefix + 'experts.'):].split('.')
                expert_idx = int(parts[0])
                proj_name = parts[1]
                
                weight = state_dict.pop(k)
                
                if proj_name == 'gate_proj':
                    self._temp_gate_weights[expert_idx] = weight.t()
                elif proj_name == 'up_proj':
                    self._temp_up_weights[expert_idx] = weight.t()
                elif proj_name == 'down_proj':
                    self._temp_down_weights[expert_idx] = weight.t()
                    
        # Once all shards have contributed their parameters, perform in-place fusion!
        if (len(self._temp_gate_weights) == num_experts and 
            len(self._temp_up_weights) == num_experts and 
            len(self._temp_down_weights) == num_experts):
            
            gate_stacked = torch.stack([self._temp_gate_weights[i] for i in range(num_experts)])
            up_stacked = torch.stack([self._temp_up_weights[i] for i in range(num_experts)])
            down_stacked = torch.stack([self._temp_down_weights[i] for i in range(num_experts)])
            
            self.experts_w12.data.copy_(torch.cat([gate_stacked, up_stacked], dim=-1))
            self.experts_w3.data.copy_(down_stacked)
            
            # Deallocate temporary buffers to free CPU memory
            del self._temp_gate_weights
            del self._temp_up_weights
            del self._temp_down_weights
            
        # Satisfy strict loading checks by injecting the fused tensors if HF expects them
        if w12_key not in state_dict:
            state_dict[w12_key] = self.experts_w12.data
        if w3_key not in state_dict:
            state_dict[w3_key] = self.experts_w3.data
            
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, hidden_states):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        topk_idx, topk_weight, router_logits = self.gate(hidden_states)
        
        # The old inference loop scans every expert and issues many tiny GPU ops,
        # which makes one-token decode extremely slow. Keep it as an escape hatch
        # for debugging, but default inference to a batched expert path.
        infer_all_experts = os.environ.get("QUASAR_MOE_INFER_ALL_EXPERTS", "1") == "1"
        decode_only_all_experts = os.environ.get("QUASAR_MOE_INFER_ALL_EXPERTS_DECODE_ONLY", "0") == "1"
        if (not self.training) and os.environ.get("QUASAR_MOE_INFER_LOOP", "0") == "1":
            y = self.moe_loop(hidden_states, topk_idx, topk_weight)
        elif not self.training and infer_all_experts and (not decode_only_all_experts or seq_len == 1):
            y = self.moe_all_experts(hidden_states, topk_idx, topk_weight)
        else:
            y = self.moe_vectorized(hidden_states, topk_idx, topk_weight)
            
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y, (router_logits.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, -1))

    def moe_loop(self, x, topk_ids, topk_weight):
        bsz, seq_len, h_dim = x.shape
        k = topk_ids.shape[-1]
        flat_x = x.view(-1, h_dim)
        flat_topk_idx = topk_ids.view(-1)
        
        routed_out = torch.zeros_like(flat_x)
        
        flat_x_repeated = flat_x.repeat_interleave(k, dim=0)
        flat_topk_weight = topk_weight.view(-1, 1)
        
        for i in range(self.config.num_experts):
            assigned_mask = (flat_topk_idx == i)
            if not assigned_mask.any():
                continue
                
            expert_inputs = flat_x_repeated[assigned_mask]
            expert_weights = flat_topk_weight[assigned_mask]
            
            w12 = self.experts_w12[i]
            w3 = self.experts_w3[i]
            
            h12 = expert_inputs @ w12
            h1, h2 = h12.chunk(2, dim=-1)
            h = F.silu(h1) * h2
            expert_out = h @ w3
            
            weighted_out = expert_out * expert_weights
            
            items_indices = torch.arange(bsz * seq_len * k, device=x.device)[assigned_mask]
            token_indices = items_indices // k
            
            routed_out.index_add_(0, token_indices, weighted_out)
            
        return routed_out.view(bsz, seq_len, h_dim)

    def moe_all_experts(self, x, topk_ids, topk_weight):
        bsz, seq_len, h_dim = x.shape
        num_tokens = bsz * seq_len
        flat_x = x.reshape(num_tokens, h_dim)
        # GPT-OSS style inference: compute all experts as one batched GEMM and
        # gather/weight only the routed experts. This trades memory for much
        # fewer tiny launches and is especially faster for one-token decode.
        expert_x = flat_x.unsqueeze(0).expand(self.config.num_experts, -1, -1)
        h12 = torch.bmm(expert_x, self.experts_w12)
        h1, h2 = h12.chunk(2, dim=-1)
        h = F.silu(h1) * h2
        expert_out = torch.bmm(h, self.experts_w3).transpose(0, 1).contiguous()
        routed = expert_out.gather(
            1,
            topk_ids.reshape(num_tokens, -1, 1).expand(-1, -1, h_dim),
        )
        routed = routed * topk_weight.reshape(num_tokens, -1, 1).to(dtype=routed.dtype)
        return routed.sum(dim=1).view(bsz, seq_len, h_dim)

    def moe_vectorized(self, x, topk_ids, topk_weight):
        bsz, seq_len, h_dim = x.shape
        k = topk_ids.shape[-1]
        flat_x = x.view(-1, h_dim)
        
        w12_t = self.experts_w12
        down_w_t = self.experts_w3
        
        num_experts = self.config.num_experts
        flat_topk_idx = topk_ids.view(-1)
        tokens_per_expert = torch.bincount(flat_topk_idx, minlength=num_experts)
        
        # Capacity limit: max 2.0x average tokens per expert, minimum 128
        avg_tokens = (bsz * seq_len * k) // num_experts
        capacity = max(128, int(2.0 * avg_tokens))
        
        sorted_indices = torch.argsort(flat_topk_idx)
        token_indices = torch.arange(bsz * seq_len, device=x.device).repeat_interleave(k)[sorted_indices]
        
        expert_starts = torch.cat([torch.tensor([0], device=x.device), tokens_per_expert[:-1].cumsum(0)])
        intra_offsets = torch.arange(bsz * seq_len * k, device=x.device) - expert_starts.repeat_interleave(tokens_per_expert)
        expert_idx = flat_topk_idx[sorted_indices]
        
        # Apply capacity limit mask
        mask = intra_offsets < capacity
        sorted_indices = sorted_indices[mask]
        token_indices = token_indices[mask]
        expert_idx = expert_idx[mask]
        intra_offsets = intra_offsets[mask]

        kept_per_expert = torch.bincount(expert_idx, minlength=num_experts)
        active_experts = torch.nonzero(kept_per_expert, as_tuple=False).flatten()
        active_counts = kept_per_expert[active_experts]
        active_starts = torch.cat(
            [active_counts.new_zeros(1), active_counts.cumsum(0)[:-1]],
            dim=0,
        )

        grouped_x = flat_x[token_indices]
        gating_flat = topk_weight.view(-1)
        sorted_gating = gating_flat[sorted_indices].unsqueeze(1)
        routed_out = torch.zeros_like(flat_x)

        # Keep the batched-GEMM path, but tile experts to cap peak activation memory.
        # Two-H200 runs leave very little headroom after FSDP unshards the MoE weights.
        default_tile_size = "1" if self.training else "8"
        expert_tile_size = int(os.environ.get("QUASAR_MOE_TILE_SIZE", default_tile_size))
        for tile_start in range(0, active_experts.numel(), expert_tile_size):
            tile_end = min(tile_start + expert_tile_size, active_experts.numel())
            tile_experts = active_experts[tile_start:tile_end]
            tile_counts = active_counts[tile_start:tile_end]
            tile_capacity = int(tile_counts.max().item())
            tile_data_start = int(active_starts[tile_start].item())
            tile_data_end = int((active_starts[tile_end - 1] + active_counts[tile_end - 1]).item())

            tile_grouped_x = grouped_x[tile_data_start:tile_data_end]
            tile_token_indices = token_indices[tile_data_start:tile_data_end]
            tile_intra_offsets = intra_offsets[tile_data_start:tile_data_end]
            tile_gating = sorted_gating[tile_data_start:tile_data_end]

            if tile_experts.numel() == 1:
                # Python-int indexing returns a view. Tensor/list indexing copies the
                # expert weights, which can OOM when FSDP has already unsharded them.
                expert_id = int(tile_experts[0].item())
                h12 = tile_grouped_x.matmul(w12_t[expert_id])
                h1, h2 = h12.chunk(2, dim=-1)
                h = F.silu(h1) * h2
                expert_out = h.matmul(down_w_t[expert_id])
                routed_out.index_add_(0, tile_token_indices, expert_out * tile_gating)
                continue

            tile_w12 = w12_t[tile_experts]
            tile_w3 = down_w_t[tile_experts]

            tile_expert_positions = torch.repeat_interleave(
                torch.arange(tile_experts.numel(), device=x.device),
                tile_counts,
            )

            padded_x = torch.zeros(
                tile_experts.numel(),
                tile_capacity,
                h_dim,
                device=x.device,
                dtype=x.dtype,
            )
            padded_x_flat = padded_x.view(-1, h_dim)
            flat_dest_indices = tile_expert_positions * tile_capacity + tile_intra_offsets
            padded_x_flat.index_put_((flat_dest_indices,), tile_grouped_x)

            h12 = torch.bmm(padded_x, tile_w12)
            h1, h2 = h12.chunk(2, dim=-1)
            h = F.silu(h1) * h2
            expert_out_padded = torch.bmm(h, tile_w3)

            tile_expert_out = expert_out_padded.view(-1, h_dim)[flat_dest_indices]
            weighted_out = tile_expert_out * tile_gating
            routed_out.index_add_(0, tile_token_indices, weighted_out)

        return routed_out.view(bsz, seq_len, h_dim)

    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        # CRITICAL: Use .tolist() instead of .cpu().numpy() to reduce sync overhead if possible
        # but the real fix is the vectorized path above.
        tokens_per_expert_list = tokens_per_expert.tolist()
        outputs = []
        dummy_outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert_list):
            expert = self.experts[i]
            if num_tokens > 0:
                expert_out = expert(sorted_tokens[start_idx:start_idx+num_tokens])
                outputs.append(expert_out)
                start_idx += num_tokens
            else:
                # Force ZeRO-3 hooks to trigger by passing a 1-element dummy tensor
                # Multiply by 0.0 and sum to a scalar so it can be added to the graph safely.
                dummy_input = sorted_tokens[0:1]
                dummy_out = expert(dummy_input) * 0.0
                dummy_outputs.append(dummy_out.sum())

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        
        # Add the dummy outputs to the graph to prevent PyTorch from skipping the backward pass
        if len(dummy_outputs) > 0:
            final_out = final_out + sum(dummy_outputs)
            
        return final_out


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int, head_first: bool = True) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    """
    if n_rep == 1:
        return hidden_states
    if head_first:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->QuasarLong
class QuasarLongAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: QuasarLongConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or self.hidden_size // self.num_heads
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        self.rope_dim = int(self.head_dim * partial_rotary_factor)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.query_key_value = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )

        if self.config.use_qk_norm:
            self.query_layernorm = QuasarLongRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = QuasarLongRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)

    def reset_parameters(self) -> None:
        for module in self.children():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.config.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings
        if not _quasar_long_global_nope_enabled(self.config):
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            cache_kwargs = {"sin": sin, "cos": cos}
            if self.layer_idx < self.config.num_hidden_layers:
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        kv_seq_len = key_states.shape[-2]
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->QuasarLong
class QuasarLongFlashAttention2(QuasarLongAttention):
    """
    QuasarLong flash attention module. This module inherits from `QuasarLongAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # QuasarLongFlashAttention2 attention does not support output_attentions
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.config.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings
        if not _quasar_long_global_nope_enabled(self.config):
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None and self.layer_idx < self.config.num_hidden_layers:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently cast in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slow down training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (QuasarLongRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            # Handle the case where the model is quantized
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            elif torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            else:
                target_dtype = self.query_key_value.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            query_length (`int`):
                The length of the query sequence in terms of tokens. This represents the number of tokens in the
                `query_states` tensor along the sequence dimension. It is used to determine the effective sequence
                length for attention computations.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in QuasarLongFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


# Copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention with Llama->QuasarLong
class QuasarLongSdpaAttention(QuasarLongAttention):
    """
    QuasarLong attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `QuasarLongAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from QuasarLongAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "QuasarLongModel is using QuasarLongSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.config.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        cos, sin = position_embeddings
        if not _quasar_long_global_nope_enabled(self.config):
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None and self.layer_idx < self.config.num_hidden_layers:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            kv_seq_len = key_states.shape[-2]
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.dense(attn_output)

        return attn_output, None, past_key_value


class QuasarLongLinearAttention(nn.Module):
    """Quasar-shaped GLA branch used as the trainable replacement candidate.

    This intentionally mirrors the original attention projection path: one
    fused QKV projection, optional QK RMSNorm, RoPE on Q/K, GQA-style KV repeat,
    and a final dense projection back to hidden size.
    """

    def __init__(self, config: QuasarLongConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.mode = getattr(config, "hybrid_gla_mode", "chunk")

        self.query_key_value = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )
        if self.config.use_qk_norm:
            self.query_layernorm = QuasarLongRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.key_layernorm = QuasarLongRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)
        self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.g_norm = QuasarLongGroupRMSNorm(
            self.num_heads * self.head_dim,
            group_norm_size=getattr(config, "hybrid_gla_group_norm_size", self.num_heads),
            eps=config.rms_norm_eps,
        )
        slope = -self.build_slope_tensor(self.num_heads)
        if config.num_hidden_layers > 1 and layer_idx is not None:
            slope = slope * (1 - max(layer_idx - 1, 0) / (config.num_hidden_layers - 1) + 1e-5)
        self.register_buffer("slope", slope, persistent=True)

        from fla.ops.simple_gla.chunk import chunk_simple_gla
        from fla.ops.simple_gla.fused_recurrent import fused_recurrent_simple_gla
        from fla.ops.simple_gla.naive import naive_chunk_simple_gla, naive_recurrent_simple_gla

        self.lightning_attn_ops = {
            "chunk": chunk_simple_gla,
            "fused_recurrent": fused_recurrent_simple_gla,
            "naive_chunk": naive_chunk_simple_gla,
            "naive_recurrent": naive_recurrent_simple_gla,
        }

    def reset_parameters(self) -> None:
        pass

    @staticmethod
    def build_slope_tensor(n_attention_heads: int):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

        return torch.tensor(get_slopes(n_attention_heads), dtype=torch.float32)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if past_key_value is None:
            # The hybrid wrapper passes the shared QGR branch cache as
            # `past_key_values` to match Quasar/Raven. Accept that alias here so
            # GLA can use the recurrent one-token decode kernel instead of the
            # much slower chunk kernel.
            past_key_value = kwargs.get("past_key_values", None)
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding)."
            )
        assert not output_attentions, "GLA replacement branch does not support output_attentions=True"

        bsz, q_len, _ = hidden_states.size()
        mode = self.mode
        if (
            (not self.training)
            and q_len == 1
            and use_cache
            and past_key_value is not None
            and mode in {"chunk", "fused_chunk", "naive_chunk"}
            and "fused_recurrent" in self.lightning_attn_ops
        ):
            mode = "fused_recurrent"

        # ── Sample packing: simple-GLA treats each row as one sequence, so a packed
        # row leaks state across segments. With cu_seqlens, flatten to varlen (B=1),
        # run with cu_seqlens (resets the recurrence per segment) and scatter back;
        # RoPE cos/sin are unpadded with the same indices.
        cu_seqlens = kwargs.get("cu_seqlens", None)
        unpad_indices = None
        eff_bsz, eff_len = bsz, q_len
        if cu_seqlens is not None and attention_mask is not None:
            from einops import rearrange
            from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
            unpad_indices, _, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), unpad_indices
            ).unsqueeze(0)
            eff_bsz, eff_len = 1, hidden_states.shape[1]

        qkv = self.query_key_value(hidden_states)
        _debug_assert_finite("qkv_proj", qkv, self.layer_idx)
        qkv = qkv.view(eff_bsz, eff_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        _debug_assert_finite("qkv_split_q", query_states, self.layer_idx)
        _debug_assert_finite("qkv_split_k", key_states, self.layer_idx)
        _debug_assert_finite("qkv_split_v", value_states, self.layer_idx)

        if self.config.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)
            _debug_assert_finite("qk_norm_q", query_states, self.layer_idx)
            _debug_assert_finite("qk_norm_k", key_states, self.layer_idx)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            if not _quasar_long_global_nope_enabled(self.config):
                if unpad_indices is not None:
                    # [B|1, S, rot] -> expand to [B, S, rot] -> unpad -> [1, N, rot]
                    cos_e = cos.expand(bsz, -1, -1) if cos.shape[0] == 1 else cos
                    sin_e = sin.expand(bsz, -1, -1) if sin.shape[0] == 1 else sin
                    cos = index_first_axis(rearrange(cos_e, "b s d -> (b s) d"), unpad_indices).unsqueeze(0)
                    sin = index_first_axis(rearrange(sin_e, "b s d -> (b s) d"), unpad_indices).unsqueeze(0)
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)
            _debug_assert_finite("rope_q", query_states, self.layer_idx)
            _debug_assert_finite("rope_k", key_states, self.layer_idx)

        if self.num_key_value_groups > 1:
            key_states = repeat_kv(key_states, self.num_key_value_groups, head_first=False)
            value_states = repeat_kv(value_states, self.num_key_value_groups, head_first=False)
            _debug_assert_finite("repeat_k", key_states, self.layer_idx)
            _debug_assert_finite("repeat_v", value_states, self.layer_idx)

        # Zero padded values only in the (un-flattened) batched path; packing has
        # already removed padding tokens via the unpad above.
        if unpad_indices is None and attention_mask is not None and not bool(attention_mask.all()):
            value_states = value_states * attention_mask[:, -q_len:, None, None].to(dtype=value_states.dtype)

        recurrent_state = None
        if past_key_value is not None and self.layer_idx is not None:
            try:
                if len(past_key_value) > self.layer_idx:
                    last_state = past_key_value[self.layer_idx]
                    if isinstance(last_state, dict):
                        recurrent_state = last_state.get("recurrent_state", None)
            except TypeError:
                pass
        kernel_fp32 = bool(getattr(self.config, "hybrid_gla_kernel_fp32", False))
        kernel_dtype = torch.float32 if kernel_fp32 else query_states.dtype
        query_states = query_states.to(kernel_dtype)
        key_states = key_states.to(kernel_dtype)
        value_states = value_states.to(kernel_dtype)
        decay = self.slope.to(dtype=kernel_dtype, device=hidden_states.device)
        op_kwargs = dict(
            q=query_states,
            k=key_states,
            v=value_states,
            g=decay[None, None, :].expand(eff_bsz, eff_len, self.num_heads),
            initial_state=recurrent_state,
            output_final_state=use_cache,
        )
        if cu_seqlens is not None:
            op_kwargs["cu_seqlens"] = cu_seqlens
        o, recurrent_state = self.lightning_attn_ops[mode](**op_kwargs)
        if past_key_value is not None and use_cache:
            past_key_value.update(
                layer_idx=self.layer_idx,
                recurrent_state=recurrent_state,
                conv_state=None,
                offset=q_len,
            )
        _debug_assert_finite("simple_gla_output", o, self.layer_idx)

        o = o.reshape(eff_bsz, eff_len, -1)
        o = self.g_norm(o)
        _debug_assert_finite("g_norm", o, self.layer_idx)
        o = o * torch.sigmoid(self.g_proj(hidden_states))
        _debug_assert_finite("output_gate", o, self.layer_idx)
        o = self.dense(o.to(hidden_states.dtype))
        _debug_assert_finite("dense", o, self.layer_idx)
        if unpad_indices is not None:
            o = pad_input(o.squeeze(0), unpad_indices, bsz, q_len)
        return o, None, past_key_value


class QuasarLongHybridReplacementSdpaAttention(QuasarLongSdpaAttention):
    """SDPA attention with a gated Quasar+GLA replacement path.

    Original GQA parameters stay at the top level of this module, so pretrained
    `attention.query_key_value` and `attention.dense` weights load unchanged.
    """

    def __init__(self, config: QuasarLongConfig, layer_idx: Optional[int] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        hybrid_layers = set(getattr(config, "hybrid_attention_layers", []) or [])
        self.hybrid_enabled = layer_idx in hybrid_layers
        self.hybrid_replacement_mode = str(getattr(config, "hybrid_replacement_mode", "gated")).lower()
        self.last_gqa_output = None
        self.last_linear_output = None
        self.last_quasar_output = None
        self.last_raven_output = None
        self.last_gla_output = None
        self.last_local_window_output = None
        self.last_pre_channel_output = None
        self.last_global_pre_channel_output = None
        if not self.hybrid_enabled:
            return

        from fla.layers.quasar import QuasarAttention
        if not os.path.isdir(os.path.join(_HERE, "raven")):
            raise ModuleNotFoundError("Quasar requires the bundled repo-local raven/ folder for Raven hybrid layers")
        from raven.layers.raven import RavenAttention

        use_short_conv = bool(getattr(config, "hybrid_use_short_conv", False))
        self.hybrid_branch_layout = str(getattr(config, "hybrid_branch_layout", "mixed") or "mixed").strip().lower()
        self.hybrid_assigned_branch = "mixed"
        if self.hybrid_branch_layout == "layerwise":
            enabled_branches = {
                "quasar": bool(getattr(config, "hybrid_quasar_enabled", True)),
                "raven": bool(getattr(config, "hybrid_raven_enabled", False)),
                "gla": bool(getattr(config, "hybrid_gla_enabled", True)),
            }
            cycle = getattr(config, "hybrid_layerwise_cycle", ["quasar", "raven", "gla"]) or ["quasar"]
            cycle = [
                str(branch).strip().lower()
                for branch in cycle
                if str(branch).strip().lower() in enabled_branches
                and enabled_branches[str(branch).strip().lower()]
            ]
            if not cycle:
                cycle = [name for name, enabled in enabled_branches.items() if enabled] or ["quasar"]
            hybrid_order = sorted(hybrid_layers)
            branch_pos = hybrid_order.index(layer_idx) if layer_idx in hybrid_order else 0
            self.hybrid_assigned_branch = cycle[branch_pos % len(cycle)]
        self.replace_alpha_raw = nn.Parameter(
            torch.tensor([float(getattr(config, "hybrid_alpha_init", -15.0))], dtype=torch.float32)
        )
        self.branch_mix_logits = nn.Parameter(torch.zeros(3, dtype=torch.float32))
        self.branch_output_gain = nn.Parameter(
            torch.tensor([float(getattr(config, "hybrid_output_gain_init", 1.0))], dtype=torch.float32)
        )
        self.branch_global_output_gain = nn.Parameter(
            torch.tensor([float(getattr(config, "hybrid_global_output_gain_init", getattr(config, "hybrid_output_gain_init", 1.0)))], dtype=torch.float32)
        )
        self.branch_output_channel_gain = nn.Parameter(torch.ones(config.hidden_size, dtype=torch.float32))

        local_window_layers = set(getattr(config, "hybrid_local_window_layers", []) or [])
        self.local_window_size = int(getattr(config, "hybrid_local_window_size", 0) or 0)
        self.local_window_enabled = self.local_window_size > 0 and (
            not local_window_layers or layer_idx in local_window_layers
        )
        local_window_fraction = float(getattr(config, "hybrid_local_window_fraction", 0.0) or 0.0)
        local_window_fraction = min(max(local_window_fraction, 1e-6), 1.0 - 1e-6)
        self.branch_local_window_mix_logit = nn.Parameter(
            torch.tensor([math.log(local_window_fraction / (1.0 - local_window_fraction))], dtype=torch.float32)
        )
        local_meta_layers = set(getattr(config, "hybrid_local_meta_layers", []) or [])
        self.local_meta_enabled = self.local_window_enabled and (
            not local_meta_layers or layer_idx in local_meta_layers
        )
        self.local_meta_tokens = int(getattr(config, "hybrid_local_meta_tokens", 0) or 0)
        if not self.local_meta_enabled:
            self.local_meta_tokens = 0
        if self.local_window_enabled and self.local_meta_tokens > 0:
            self.local_meta_key = nn.Parameter(
                torch.empty(self.num_heads, self.local_meta_tokens, self.head_dim, dtype=torch.float32)
            )
            self.local_meta_value = nn.Parameter(
                torch.empty(self.num_heads, self.local_meta_tokens, self.head_dim, dtype=torch.float32)
            )
            self._reset_local_meta_tokens()
        else:
            self.local_meta_key = None
            self.local_meta_value = None
        self.branch_output_adapter_rank = int(getattr(config, 'hybrid_output_adapter_rank', 16) or 0)
        self.branch_output_adapter_scale = float(
            getattr(config, 'hybrid_output_adapter_alpha', max(self.branch_output_adapter_rank, 1))
        ) / max(self.branch_output_adapter_rank, 1)
        if self.branch_output_adapter_rank > 0:
            self.branch_output_adapter_down = nn.Linear(
                config.hidden_size, self.branch_output_adapter_rank, bias=False
            )
            self.branch_output_adapter_up = nn.Linear(
                self.branch_output_adapter_rank, config.hidden_size, bias=False
            )
            self.branch_output_adapter_down._skip_quasar_hf_init = True
            self.branch_output_adapter_up._skip_quasar_hf_init = True
            self._reset_branch_output_adapter()
        else:
            self.branch_output_adapter_down = None
            self.branch_output_adapter_up = None
        self.distill_sum = nn.Identity()
        gla_layers = set(getattr(config, "hybrid_gla_layers", []) or [])
        gla_enabled_here = bool(getattr(config, "hybrid_gla_enabled", True)) and (
            not gla_layers or layer_idx in gla_layers
        )
        layerwise = self.hybrid_branch_layout == "layerwise"
        want_quasar = bool(getattr(config, "hybrid_quasar_enabled", True)) and (
            not layerwise or self.hybrid_assigned_branch == "quasar"
        )
        want_raven = bool(getattr(config, "hybrid_raven_enabled", False)) and (
            not layerwise or self.hybrid_assigned_branch == "raven"
        )
        want_gla = gla_enabled_here and (
            not layerwise or self.hybrid_assigned_branch == "gla"
        )
        self.gla_attention = (
            QuasarLongLinearAttention(config=config, layer_idx=layer_idx)
            if want_gla
            else None
        )
        self.quasar_attention = (
            QuasarAttention(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_heads=config.num_attention_heads,
                mode=getattr(config, "hybrid_quasar_mode", "chunk"),
                use_short_conv=use_short_conv,
                conv_size=4,
                conv_bias=False,
                norm_eps=config.rms_norm_eps,
                layer_idx=layer_idx,
            )
            if want_quasar
            else None
        )
        self.raven_attention = (
            RavenAttention(
                mode=getattr(config, "hybrid_gla_mode", "fused_recurrent"),
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                num_slots=getattr(config, "hybrid_raven_slots", 64),
                topk=getattr(config, "hybrid_raven_topk", 32),
                decay_type=getattr(config, "hybrid_raven_decay_type", "Mamba2"),
                add_gumbel_noise=bool(getattr(config, "hybrid_raven_add_gumbel_noise", False)),
                norm_eps=config.rms_norm_eps,
                layer_idx=layer_idx,
            )
            if want_raven
            else None
        )
        for branch in (self.gla_attention, self.quasar_attention, self.raven_attention):
            if branch is not None:
                for module in branch.modules():
                    module._skip_quasar_hf_init = True

    def _reset_local_meta_tokens(self) -> None:
        if self.local_meta_key is None or self.local_meta_value is None:
            return
        std = float(getattr(self.config, "hybrid_local_meta_init_std", 0.02) or 0.02)
        nn.init.normal_(self.local_meta_key, mean=0.0, std=std)
        nn.init.normal_(self.local_meta_value, mean=0.0, std=std)

    def _reset_branch_output_adapter(self) -> None:
        if self.branch_output_adapter_down is None or self.branch_output_adapter_up is None:
            return
        nn.init.kaiming_uniform_(self.branch_output_adapter_down.weight, a=math.sqrt(5))
        self.branch_output_adapter_up.weight.data.zero_()

    def _apply_branch_output_adapter(self, linear_out: torch.Tensor) -> torch.Tensor:
        if self.branch_output_adapter_down is None or self.branch_output_adapter_up is None:
            return linear_out
        adapter_hidden = self.branch_output_adapter_down(linear_out)
        adapter_out = self.branch_output_adapter_up(adapter_hidden)
        return linear_out + self.branch_output_adapter_scale * adapter_out.to(dtype=linear_out.dtype)

    @staticmethod
    def _to_linear_attention_mask(
        attention_mask: Optional[torch.Tensor],
        *,
        bsz: int,
        q_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            mask = attention_mask[:, -q_len:]
            return None if bool(mask.all()) else mask.to(device=device, dtype=torch.int32)
        if attention_mask.dim() == 4 and attention_mask.shape[1] == 1:
            mask = attention_mask[:, 0, -1, -q_len:]
            mask = (mask > -1e4)
            return None if bool(mask.all()) else mask.to(device=device, dtype=torch.int32)
        raise ValueError(f"Unsupported linear attention mask shape: {attention_mask.shape}")

    def reset_hybrid_branch_parameters(self) -> None:
        if hasattr(self, "engram") and self.engram is not None:
            self.engram._init_weights()
        if not self.hybrid_enabled:
            return
        if self.gla_attention is not None and hasattr(self.gla_attention, "slope"):
            slope = -self.gla_attention.build_slope_tensor(self.gla_attention.num_heads)
            if self.config.num_hidden_layers > 1 and self.layer_idx is not None:
                slope = slope * (1 - max(self.layer_idx - 1, 0) / (self.config.num_hidden_layers - 1) + 1e-5)
            self.gla_attention.slope.data.copy_(slope.to(device=self.gla_attention.slope.device, dtype=self.gla_attention.slope.dtype))
        for branch in (self.gla_attention, self.quasar_attention, self.raven_attention):
            if branch is None:
                continue
            for module in branch.modules():
                if module is branch:
                    continue
                if isinstance(module, (QuasarLongRMSNorm, QuasarLongGroupRMSNorm)):
                    module.weight.data.fill_(1.0)
                    continue
                reset = getattr(module, "reset_parameters", None)
                if callable(reset):
                    reset()
            if hasattr(branch, "A_log"):
                branch.A_log.data.copy_(torch.log(torch.empty_like(branch.A_log).uniform_(1, 16)))
            if hasattr(branch, "dt_bias"):
                branch.dt_bias.data.zero_()
        self.replace_alpha_raw.data.fill_(float(getattr(self.config, "hybrid_alpha_init", -15.0)))
        self.branch_mix_logits.data.zero_()
        self.branch_output_gain.data.fill_(float(getattr(self.config, "hybrid_output_gain_init", 1.0)))
        self.branch_global_output_gain.data.fill_(
            float(getattr(self.config, "hybrid_global_output_gain_init", getattr(self.config, "hybrid_output_gain_init", 1.0)))
        )
        self.branch_output_channel_gain.data.fill_(1.0)

        local_window_fraction = float(getattr(self.config, "hybrid_local_window_fraction", 0.0) or 0.0)
        local_window_fraction = min(max(local_window_fraction, 1e-6), 1.0 - 1e-6)
        self.branch_local_window_mix_logit.data.fill_(math.log(local_window_fraction / (1.0 - local_window_fraction)))
        self._reset_branch_output_adapter()
        self._reset_local_meta_tokens()


    def _local_window_fraction(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        local_fraction = torch.sigmoid(self.branch_local_window_mix_logit).to(dtype=dtype, device=device)
        max_fraction = float(getattr(self.config, "hybrid_local_window_max_fraction", 0.3333333) or 0.3333333)
        return torch.clamp(local_fraction, min=0.0, max=max_fraction)

    def _local_window_attention_output(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # LoLCATs-style local softmax path. This keeps only a small causal window exact,
        # while the global branch remains Quasar+GLA.
        bsz, q_len, _ = hidden_states.shape
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.config.use_qk_norm:
            query_states = self.query_layernorm(query_states)
            key_states = self.key_layernorm(key_states)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            if not _quasar_long_global_nope_enabled(self.config):
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        scores = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        query_pos = torch.arange(q_len, device=hidden_states.device)[:, None]
        key_pos = torch.arange(q_len, device=hidden_states.device)[None, :]
        window = max(int(self.local_window_size), 1)
        local_mask = (key_pos <= query_pos) & (key_pos >= query_pos - window + 1)
        min_value = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~local_mask.view(1, 1, q_len, q_len), min_value)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                key_padding_mask = attention_mask[:, -q_len:].to(device=hidden_states.device).bool()
                scores = scores.masked_fill(~key_padding_mask.view(bsz, 1, 1, q_len), min_value)
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask[:, :, -q_len:, -q_len:].to(device=scores.device, dtype=scores.dtype)
            else:
                raise ValueError(f"Unsupported local attention mask shape: {attention_mask.shape}")

        if self.local_meta_key is not None and self.local_meta_value is not None:
            meta_key = self.local_meta_key.to(device=query_states.device, dtype=query_states.dtype)
            meta_value = self.local_meta_value.to(device=value_states.device, dtype=value_states.dtype)
            meta_scores = torch.einsum("bhqd,hmd->bhqm", query_states, meta_key) / math.sqrt(self.head_dim)
            scores = torch.cat([meta_scores, scores], dim=-1)
            meta_value = meta_value.unsqueeze(0).expand(bsz, -1, -1, -1)
            value_states = torch.cat([meta_value, value_states], dim=2)

        probs = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        probs = nn.functional.dropout(probs, p=self.attention_dropout, training=self.training)
        local_out = torch.matmul(probs, value_states)
        local_out = local_out.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        local_out = self.dense(local_out)
        _debug_assert_finite("local_window_output", local_out, self.layer_idx)
        return local_out

    def _linear_attention_output(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool,
        branch_past_key_values: Optional[QGRBranchCache] = None,
        branch_use_cache: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        packed_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            self.training
            and bool(getattr(self.config, "hybrid_attention_mimic_return_gqa", False))
            and not torch.is_grad_enabled()
        ):
            with torch.enable_grad():
                return self._linear_attention_output(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    output_attentions=output_attentions,
                    branch_past_key_values=branch_past_key_values,
                    branch_use_cache=branch_use_cache,
                    cu_seqlens=cu_seqlens,
                    packed_padding_mask=packed_padding_mask,
                )
        _debug_assert_finite("linear_input_hidden_states", hidden_states, self.layer_idx)
        bsz, q_len, _ = hidden_states.shape
        if cu_seqlens is not None:
            # Packed: give the branches the true 2D padding mask (-> varlen B=1) plus
            # segment cu_seqlens so their recurrent/GSA state resets per segment.
            linear_attention_mask = packed_padding_mask[:, -q_len:].to(
                device=hidden_states.device, dtype=torch.int32
            )
        else:
            linear_attention_mask = self._to_linear_attention_mask(
                attention_mask,
                bsz=bsz,
                q_len=q_len,
                device=hidden_states.device,
            )
        outputs = []
        self.last_quasar_output = None
        self.last_raven_output = None
        self.last_gla_output = None
        active_branches = None
        if self.training and (
            bool(getattr(self.config, "hybrid_attention_mimic_return_gqa", False))
            or bool(getattr(self.config, "hybrid_attention_collect_branch_loss", False))
        ):
            active_branches = set(getattr(self.config, "branch_mimic_branches", ["quasar", "raven", "gla", "mixed"]))
        eval_force_branch = None
        if not self.training:
            eval_force_branch = str(getattr(self.config, "hybrid_eval_force_branch", "") or "").strip().lower()
            if eval_force_branch in {"quasar", "raven", "gla", "mixed"}:
                active_branches = {eval_force_branch}
        needs_mixed = active_branches is None or "mixed" in active_branches
        # Branch-mimic distillation trains the replacement attention modules to
        # match the frozen GQA teacher on fixed hidden features. Detaching here
        # prevents backward from traversing the full frozen 20B base model.
        branch_hidden_states = hidden_states.detach() if active_branches is not None else hidden_states
        
        # 1. Quasar
        if self.quasar_attention is not None and (active_branches is None or "quasar" in active_branches or needs_mixed):
            use_quasar_rope = bool(getattr(self.config, "hybrid_quasar_use_rope", False)) and not _quasar_long_global_nope_enabled(self.config)
            cos, sin = position_embeddings if (use_quasar_rope and position_embeddings is not None) else (None, None)
            if cos is not None and sin is not None:
                q_head_dim = int(self.quasar_attention.head_dim)
                cos = cos[..., :q_head_dim]
                sin = sin[..., :q_head_dim]
                if cos.dim() == 3:
                    cos = cos.unsqueeze(1)
                    sin = sin.unsqueeze(1)
            q_out = self.quasar_attention(
                hidden_states=branch_hidden_states,
                attention_mask=linear_attention_mask,
                past_key_values=branch_past_key_values,
                use_cache=branch_use_cache,
                output_attentions=False,
                cos=cos,
                sin=sin,
                cu_seqlens=cu_seqlens,
            )[0]
            self.last_quasar_output = q_out
            _debug_assert_finite("quasar_output", q_out, self.layer_idx)
            q_out = _sanitize_hybrid_tensor("quasar_output", q_out, self.layer_idx)
            outputs.append(q_out)
        else:
            outputs.append(branch_hidden_states.new_zeros(branch_hidden_states.shape))

        # 2. Raven
        if self.raven_attention is not None and (active_branches is None or "raven" in active_branches or needs_mixed):
            r_out = self.raven_attention(
                hidden_states=branch_hidden_states,
                attention_mask=linear_attention_mask,
                past_key_values=branch_past_key_values,
                use_cache=branch_use_cache,
                output_attentions=output_attentions,
                cu_seqlens=cu_seqlens,
            )[0]
            self.last_raven_output = r_out
            _debug_assert_finite("raven_output", r_out, self.layer_idx)
            r_out = _sanitize_hybrid_tensor("raven_output", r_out, self.layer_idx)
            outputs.append(r_out)
        else:
            outputs.append(branch_hidden_states.new_zeros(branch_hidden_states.shape))

        # 3. GLA
        if self.gla_attention is not None and (active_branches is None or "gla" in active_branches or needs_mixed):
            g_out = self.gla_attention(
                hidden_states=branch_hidden_states,
                attention_mask=linear_attention_mask,
                past_key_values=branch_past_key_values,
                use_cache=branch_use_cache,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
            )[0]
            self.last_gla_output = g_out
            _debug_assert_finite("gla_output", g_out, self.layer_idx)
            g_out = _sanitize_hybrid_tensor("gla_output", g_out, self.layer_idx)
            outputs.append(g_out)
        else:
            outputs.append(branch_hidden_states.new_zeros(branch_hidden_states.shape))

        mix = torch.softmax(self.branch_mix_logits.float(), dim=0).to(dtype=hidden_states.dtype, device=hidden_states.device)
        available_mask = torch.tensor(
            [
                1.0 if self.quasar_attention is not None else 0.0,
                1.0 if self.raven_attention is not None else 0.0,
                1.0 if self.gla_attention is not None else 0.0,
            ],
            dtype=mix.dtype,
            device=mix.device,
        )
        mix = mix * available_mask
        if active_branches is not None and not needs_mixed:
            mask = torch.tensor(
                [
                    1.0 if "quasar" in active_branches else 0.0,
                    1.0 if "raven" in active_branches else 0.0,
                    1.0 if "gla" in active_branches else 0.0,
                ],
                dtype=mix.dtype,
                device=mix.device,
            )
            mix = mix * mask
        mix = mix / torch.clamp(mix.sum(), min=1e-6)
        global_out = (
            mix[0] * outputs[0].to(dtype=hidden_states.dtype)
            + mix[1] * outputs[1].to(dtype=hidden_states.dtype)
            + mix[2] * outputs[2].to(dtype=hidden_states.dtype)
        )
        global_out = _sanitize_hybrid_tensor("global_branch_mix", global_out, self.layer_idx)
        self._last_global_branch_output = global_out

        # The final forward applies branch_output_gain after local/global mixing.
        # Scale the global branch by global_gain / output_gain here so its final
        # effective gain is branch_global_output_gain while the local scaffold keeps
        # branch_output_gain. The shadow mimic path still consumes raw global_out.
        output_gain = self.branch_output_gain.to(dtype=hidden_states.dtype, device=hidden_states.device)
        global_gain = self.branch_global_output_gain.to(dtype=hidden_states.dtype, device=hidden_states.device)
        linear_out = (global_gain / torch.clamp(output_gain, min=1e-6)) * global_out
        if self.local_window_enabled:
            local_out = self._local_window_attention_output(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )
            self.last_local_window_output = local_out.detach()
            local_fraction = self._local_window_fraction(dtype=hidden_states.dtype, device=hidden_states.device)
            linear_out = (1.0 - local_fraction) * linear_out + local_fraction * local_out.to(dtype=hidden_states.dtype)
        _debug_assert_finite("linear_branch_mix", linear_out, self.layer_idx)
        linear_out = _sanitize_hybrid_tensor("linear_branch_mix", linear_out, self.layer_idx)
        return linear_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        branch_past_key_values: Optional[QGRBranchCache] = None,
        branch_use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if (
            self.training
            and bool(getattr(self.config, "hybrid_attention_mimic_return_gqa", False))
            and not torch.is_grad_enabled()
        ):
            with torch.enable_grad():
                return self.forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
        packed_cu_seqlens = kwargs.get("packed_cu_seqlens", None)
        packed_padding_mask = kwargs.get("packed_padding_mask", None)
        fast_full_replacement = bool(
            self.hybrid_enabled
            and self.hybrid_replacement_mode in {"full", "replace", "linear"}
            and bool(getattr(self.config, "hybrid_skip_gqa_in_full_replacement", False))
            and not (self.training and bool(getattr(self.config, "hybrid_attention_transfer_pass_gqa", False)))
        )
        if fast_full_replacement:
            linear_out = self._linear_attention_output(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
                branch_past_key_values=branch_past_key_values,
                branch_use_cache=branch_use_cache,
                cu_seqlens=packed_cu_seqlens,
                packed_padding_mask=packed_padding_mask,
            )
            global_branch_out = getattr(self, "_last_global_branch_output", None)
            linear_out = self.distill_sum(linear_out)
            _debug_assert_finite("linear_distill_sum", linear_out, self.layer_idx)
            linear_out = _sanitize_hybrid_tensor("linear_distill_sum", linear_out, self.layer_idx)
            gain = self.branch_output_gain.to(dtype=linear_out.dtype, device=linear_out.device)
            linear_out = gain * linear_out
            _debug_assert_finite("linear_output_gain", linear_out, self.layer_idx)
            linear_out = _sanitize_hybrid_tensor("linear_output_gain", linear_out, self.layer_idx)
            self.last_pre_channel_output = linear_out.detach()
            channel_gain = self.branch_output_channel_gain.to(dtype=linear_out.dtype, device=linear_out.device)
            linear_out = linear_out * channel_gain.view(1, 1, -1)
            _debug_assert_finite("linear_channel_gain", linear_out, self.layer_idx)
            linear_out = _sanitize_hybrid_tensor("linear_channel_gain", linear_out, self.layer_idx)
            linear_out = self._apply_branch_output_adapter(linear_out)
            _debug_assert_finite("linear_output_adapter", linear_out, self.layer_idx)
            linear_out = _sanitize_hybrid_tensor("linear_output_adapter", linear_out, self.layer_idx)
            self.last_replacement_output = linear_out.detach()
            self.last_linear_output = linear_out
            self.last_gqa_output = None
            self.last_global_linear_output = None
            if (
                global_branch_out is not None
                and self.local_window_enabled
                and bool(getattr(self.config, "hybrid_mimic_global_branch_when_local", False))
            ):
                self.last_global_linear_output = None
            return linear_out.to(dtype=hidden_states.dtype), None, None

        gqa_out, attn_weights, present_key_value = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if not self.hybrid_enabled:
            return gqa_out, attn_weights, present_key_value

        eval_mode = ""
        if not self.training:
            eval_mode = str(getattr(self.config, "hybrid_eval_mode", "") or "").strip().lower()
            if eval_mode == "gqa_only":
                return gqa_out, attn_weights, present_key_value

        mimic_return_gqa = self.training and bool(getattr(self.config, "hybrid_attention_mimic_return_gqa", False))
        if (
            self.training
            and bool(getattr(self.config, "hybrid_attention_transfer_pass_gqa", False))
            and not mimic_return_gqa
        ):
            self.last_gqa_output = gqa_out.detach()
            self.last_replacement_output = None
            self.last_linear_output = None
            self.last_global_linear_output = None
            self.last_quasar_output = None
            self.last_raven_output = None
            self.last_gla_output = None
            return gqa_out, attn_weights, present_key_value

        # Safeguard to completely bypass the hybrid branch when it is gated out
        # This prevents NaN propagation (0.0 * NaN = NaN) from uninitialized or unstable Triton kernels
        forced_eval = eval_mode in {"quasar_forced", "raven_forced", "gla_forced", "mixed_forced"}
        alpha_bypass_enabled = bool(getattr(self.config, "hybrid_alpha_zero_bypass", False))
        if alpha_bypass_enabled and float(self.replace_alpha_raw.detach().cpu()) < -13.8 and not forced_eval and not mimic_return_gqa:
            return gqa_out, attn_weights, present_key_value

        linear_out = self._linear_attention_output(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=output_attentions,
            branch_past_key_values=branch_past_key_values,
            branch_use_cache=branch_use_cache,
            cu_seqlens=packed_cu_seqlens,
            packed_padding_mask=packed_padding_mask,
        )
        global_branch_out = getattr(self, "_last_global_branch_output", None)
        if self.training:
            self.distill_sum._distill_teacher = gqa_out.detach()
        linear_out = self.distill_sum(linear_out)
        _debug_assert_finite("linear_distill_sum", linear_out, self.layer_idx)
        linear_out = _sanitize_hybrid_tensor("linear_distill_sum", linear_out, self.layer_idx)
        gain = self.branch_output_gain.to(dtype=linear_out.dtype, device=linear_out.device)
        linear_out = gain * linear_out
        _debug_assert_finite("linear_output_gain", linear_out, self.layer_idx)
        linear_out = _sanitize_hybrid_tensor("linear_output_gain", linear_out, self.layer_idx)
        self.last_pre_channel_output = linear_out.detach()
        channel_gain = self.branch_output_channel_gain.to(dtype=linear_out.dtype, device=linear_out.device)
        linear_out = linear_out * channel_gain.view(1, 1, -1)
        _debug_assert_finite("linear_channel_gain", linear_out, self.layer_idx)
        linear_out = _sanitize_hybrid_tensor("linear_channel_gain", linear_out, self.layer_idx)
        linear_out = self._apply_branch_output_adapter(linear_out)
        _debug_assert_finite("linear_output_adapter", linear_out, self.layer_idx)
        linear_out = _sanitize_hybrid_tensor("linear_output_adapter", linear_out, self.layer_idx)

        mimic_out = linear_out
        if (
            global_branch_out is not None
            and self.local_window_enabled
            and bool(getattr(self.config, "hybrid_mimic_global_branch_when_local", False))
        ):
            global_mimic_out = self.distill_sum(global_branch_out)
            _debug_assert_finite("global_mimic_distill_sum", global_mimic_out, self.layer_idx)
            global_mimic_out = _sanitize_hybrid_tensor("global_mimic_distill_sum", global_mimic_out, self.layer_idx)
            global_gain = self.branch_global_output_gain.to(dtype=global_mimic_out.dtype, device=global_mimic_out.device)
            global_mimic_out = global_gain * global_mimic_out
            _debug_assert_finite("global_mimic_output_gain", global_mimic_out, self.layer_idx)
            global_mimic_out = _sanitize_hybrid_tensor("global_mimic_output_gain", global_mimic_out, self.layer_idx)
            self.last_global_pre_channel_output = global_mimic_out.detach()
            global_mimic_out = global_mimic_out * channel_gain.view(1, 1, -1)
            _debug_assert_finite("global_mimic_channel_gain", global_mimic_out, self.layer_idx)
            global_mimic_out = _sanitize_hybrid_tensor("global_mimic_channel_gain", global_mimic_out, self.layer_idx)
            global_mimic_out = self._apply_branch_output_adapter(global_mimic_out)
            _debug_assert_finite("global_mimic_output_adapter", global_mimic_out, self.layer_idx)
            global_mimic_out = _sanitize_hybrid_tensor("global_mimic_output_adapter", global_mimic_out, self.layer_idx)
            mimic_out = global_mimic_out
            self.last_global_linear_output = global_mimic_out.detach()

        self.last_gqa_output = gqa_out.detach()
        self.last_replacement_output = linear_out.detach()
        self.last_linear_output = mimic_out

        if mimic_return_gqa:
            return gqa_out, attn_weights, present_key_value

        if forced_eval:
            return linear_out.to(dtype=gqa_out.dtype), attn_weights, present_key_value

        if self.hybrid_replacement_mode in {"full", "replace", "linear"}:
            return linear_out.to(dtype=gqa_out.dtype), attn_weights, present_key_value

        alpha = torch.sigmoid(self.replace_alpha_raw).to(dtype=gqa_out.dtype, device=gqa_out.device)
        linear_out = linear_out.to(dtype=gqa_out.dtype)
        attn_output = gqa_out + alpha * linear_out
        attn_output = _sanitize_hybrid_tensor("gated_hybrid_output", attn_output, self.layer_idx)
        return attn_output, attn_weights, present_key_value


ATTENTION_CLASSES = {
    "eager": QuasarLongAttention,
    "flash_attention_2": QuasarLongFlashAttention2,
    "sdpa": QuasarLongHybridReplacementSdpaAttention,
}


class QuasarLongMTPLayer(nn.Module):
    def __init__(self, config: QuasarLongConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.input_layernorm = QuasarLongRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.enorm = QuasarLongRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.post_attention_layernorm = QuasarLongRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)
        self.mlp = QuasarLongSparseMoeBlock(config, layer_idx)

        self.hnorm = QuasarLongRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_layernorm = QuasarLongRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_embeds,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        def custom_mtp_attention(input_embeds_t, hidden_states_t, past_key_value_t):
            input_embeds_norm = self.enorm(input_embeds_t)
            hidden_states_norm = self.hnorm(hidden_states_t)
            h = self.eh_proj(torch.cat([input_embeds_norm, hidden_states_norm], dim=-1))
            res = h
            h_normed = self.input_layernorm(h)

            h_attn, attn_w, pres_kv = self.attention(
                hidden_states=h_normed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value_t,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
            )
            h_out = res + h_attn
            return h_out, attn_w, pres_kv

        is_ckpt_enabled = self.training and bool(getattr(self.config, "gradient_checkpointing", False))
        if is_ckpt_enabled:
                hidden_states, self_attn_weights, present_key_value = torch.utils.checkpoint.checkpoint(
                    custom_mtp_attention,
                    input_embeds,
                    hidden_states,
                    past_key_value,
                    use_reentrant=False,
                    determinism_check="none",
                )
        else:
            hidden_states, self_attn_weights, present_key_value = custom_mtp_attention(
                input_embeds,
                hidden_states,
                past_key_value,
            )

        # Fully Connected (executed outside checkpoint to prevent CheckpointError in dynamic routing)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.to(residual.device)
        hidden_states = self.final_layernorm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class QuasarLongDecoderLayer(nn.Module):
    def __init__(self, config: QuasarLongConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.attention = ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = (
            QuasarLongSparseMoeBlock(config, layer_idx)
            if (config.num_experts is not None and layer_idx >= config.first_k_dense_replace)
            else QuasarLongMLP(config=config, intermediate_size=config.intermediate_size)
        )
        self.input_layernorm = QuasarLongRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QuasarLongRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ── Looped-Transformer input-injection gate ──────────────────────────
        # logit(-6.907) ≈ 0.001 gate at step-0, conservative while the
        # looped path adapts on top of a pretrained checkpoint.
        # Mirrors HybridBlock.injection_gate in quasar_rope.py.
        if getattr(config, "use_looped_injection", False):
            self.injection_gate = nn.Parameter(torch.tensor([-6.907]))
            num_loops = max(1, int(getattr(config, "num_loops", 1)))
            self.injection_gate.register_hook(lambda g: g / float(num_loops))
        else:
            self.register_parameter("injection_gate", None)

        # Parcae-style loop stabilizer. This is initialized as a near-identity
        # transition so pretrained checkpoints are not shocked when enabled.
        if getattr(config, "use_parcae_loop_stabilizer", False):
            self.parcae_decay_raw = nn.Parameter(torch.tensor([-6.907]))
            self.parcae_anchor_gate = nn.Parameter(torch.tensor([-6.907]))
            num_loops = max(1, int(getattr(config, "num_loops", 1)))
            self.parcae_decay_raw.register_hook(lambda g: g / float(num_loops))
            self.parcae_anchor_gate.register_hook(lambda g: g / float(num_loops))
        else:
            self.register_parameter("parcae_decay_raw", None)
            self.register_parameter("parcae_anchor_gate", None)

        # ── Engram: static N-gram conditional memory ─────────────────────────
        # Attach only to the layer indices listed in config.engram_layers.
        # Falls back gracefully when engram.py is unavailable.
        _engram_layers = list(getattr(config, "engram_layers", []))
        if _ENGRAM_AVAILABLE and EngramModule is not None and layer_idx in _engram_layers:
            self.engram: Optional[nn.Module] = EngramModule(
                vocab_size=config.vocab_size,
                d_model=config.hidden_size,
                d_mem=getattr(config, "engram_dim", config.hidden_size // 4),
                num_heads=getattr(config, "engram_num_heads", 8),
                ngram_orders=list(getattr(config, "engram_ngram_orders", [2, 3])),
                target_slots=getattr(config, "engram_slots", 2_000_000),
                n_layers=config.num_hidden_layers,
            )
            self.engram.triton_training = bool(getattr(config, "engram_triton_training", False))
            # Mark so _init_weights skips re-initializing internal Engram params
            for m in self.engram.modules():
                m._skip_quasar_hf_init = True
        else:
            self.engram = None

        self._engram_residual_scale = float(getattr(config, "engram_residual_scale", 0.01))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        input_ids: Optional[torch.LongTensor] = None,  # for Engram N-gram lookup
        injection_P: Optional[torch.Tensor] = None,     # looped-injection anchor
        branch_past_key_values: Optional[QGRBranchCache] = None,
        branch_use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states: (batch, seq_len, embed_dim)
            input_ids: (batch, seq_len) – raw token IDs, optional; required only when
                an EngramModule is attached to this layer.
            injection_P: optional anchor embedding for looped-injection mixing.
            (All other args identical to the standard QuasarLongDecoderLayer.)
        """
        packed_cu_seqlens = kwargs.get("packed_cu_seqlens", None)
        packed_padding_mask = kwargs.get("packed_padding_mask", None)

        def custom_attention(h, injection_P_t, input_ids_t, past_key_value_t):
            # ── Parcae-style stable recurrence: h' = decay * h + gate * P ──
            if (
                injection_P_t is not None
                and self.parcae_decay_raw is not None
                and self.parcae_anchor_gate is not None
            ):
                decay = torch.exp(-F.softplus(self.parcae_decay_raw)).to(dtype=h.dtype, device=h.device)
                anchor_gate = torch.sigmoid(self.parcae_anchor_gate).to(dtype=h.dtype, device=h.device)
                h = decay * h + anchor_gate * injection_P_t

            # ── Looped-injection: blend residual stream with initial embeddings ──
            if injection_P_t is not None and self.injection_gate is not None:
                h = h + torch.sigmoid(self.injection_gate) * injection_P_t

            # ── Engram: add static N-gram memory signal before attention ─────────
            if self.engram is not None and input_ids_t is not None:
                engram_out, _alpha = self.engram(input_ids_t, h)
                h = h + self._engram_residual_scale * engram_out

            residual_attn = h
            h_normed = self.input_layernorm(h)

            # Self Attention
            h_attn, attn_w, pres_kv = self.attention(
                hidden_states=h_normed,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value_t,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
                branch_past_key_values=branch_past_key_values,
                branch_use_cache=branch_use_cache,
                packed_cu_seqlens=packed_cu_seqlens,
                packed_padding_mask=packed_padding_mask,
            )
            h_out = residual_attn + h_attn
            return h_out, attn_w, pres_kv

        base_no_grad = self.training and bool(getattr(self.config, "hybrid_attention_mimic_return_gqa", False))
        with torch.no_grad() if base_no_grad else nullcontext():
            is_ckpt_enabled = self.training and bool(getattr(self.config, "gradient_checkpointing", False))
            if is_ckpt_enabled:
                hidden_states, self_attn_weights, present_key_value = torch.utils.checkpoint.checkpoint(
                    custom_attention,
                    hidden_states,
                    injection_P,
                    input_ids,
                    past_key_value,
                    use_reentrant=False,
                    determinism_check="none",
                )
            else:
                hidden_states, self_attn_weights, present_key_value = custom_attention(
                    hidden_states,
                    injection_P,
                    input_ids,
                    past_key_value,
                )

            # Fully Connected (executed outside checkpoint to prevent CheckpointError in dynamic routing)
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            if isinstance(hidden_states, tuple):
                hidden_states, router_logits = hidden_states
            else:
                router_logits = None
            hidden_states = residual + hidden_states.to(residual.device)

        outputs = (hidden_states,)

        if self.training and (
            bool(getattr(self.config, "hybrid_attention_mimic_return_gqa", False))
            or bool(getattr(self.config, "hybrid_attention_collect_branch_loss", False))
        ):
            distill_clip = float(getattr(self.config, "branch_mimic_clip", 80.0))
            all_branch_names = ("quasar", "raven", "gla", "mixed")
            active_branch_set = set(getattr(self.config, "branch_mimic_branches", all_branch_names))
            branch_names = tuple(name for name in all_branch_names if name in active_branch_set)
            branch_attrs = tuple(
                item for item in (
                    ("quasar", "last_quasar_output"),
                    ("raven", "last_raven_output"),
                    ("gla", "last_gla_output"),
                    ("mixed", "last_linear_output"),
                )
                if item[0] in active_branch_set
            )
            branch_loss = hidden_states.new_zeros((), dtype=torch.float32)
            branch_loss_sums = {name: 0.0 for name in all_branch_names}
            branch_cos_sums = {name: 0.0 for name in all_branch_names}
            branch_rel_mse_sums = {name: 0.0 for name in all_branch_names}
            branch_loss_counts = {name: 0 for name in all_branch_names}
            skipped_distill = {name: 0 for name in all_branch_names}
            distill_count = 0
            detailed_branch_stats = bool(getattr(self.config, "branch_mimic_detailed_stats", False))
            sanitize_checks = False
            gqa_t = getattr(self.attention, "last_gqa_output", None)
            if gqa_t is not None:
                target = gqa_t.float().detach().clamp(-distill_clip, distill_clip)
                if detailed_branch_stats:
                    target_flat = target.reshape(-1)
                    target_energy = torch.mean(target_flat * target_flat).clamp_min(1e-8)
                for branch_name, attr_name in branch_attrs:
                    branch_s = getattr(self.attention, attr_name, None)
                    if branch_s is None:
                        continue
                    if sanitize_checks and not torch.isfinite(branch_s).all():
                        skipped_distill[branch_name] += 1
                        continue
                    pred = branch_s.float().clamp(-distill_clip, distill_clip)
                    loss_i = F.smooth_l1_loss(pred, target)
                    if sanitize_checks and not torch.isfinite(loss_i):
                        skipped_distill[branch_name] += 1
                        continue
                    branch_loss = branch_loss + loss_i
                    if detailed_branch_stats:
                        pred_flat = pred.reshape(-1)
                        mse_i = torch.mean((pred_flat - target_flat) ** 2)
                        cos_i = F.cosine_similarity(pred_flat, target_flat, dim=0)
                        branch_loss_sums[branch_name] += float(loss_i.detach().item())
                        branch_cos_sums[branch_name] += float(cos_i.detach().item()) if torch.isfinite(cos_i) else 0.0
                        branch_rel_mse_sums[branch_name] += float((mse_i / target_energy).detach().item()) if torch.isfinite(mse_i) else 0.0
                    branch_loss_counts[branch_name] += 1
                    distill_count += 1
            if distill_count > 0:
                branch_loss = branch_loss / distill_count
            outputs += (
                branch_loss,
                {
                    "branch_loss_sums": branch_loss_sums,
                    "branch_cos_sums": branch_cos_sums,
                    "branch_rel_mse_sums": branch_rel_mse_sums,
                    "branch_loss_counts": branch_loss_counts,
                    "skipped_distill": skipped_distill,
                    "distill_count": distill_count,
                },
            )

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


QUASAR_LONG_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`QuasarLongConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare QuasarLong Model outputting raw hidden-states without any specific head on top.",
    QUASAR_LONG_START_DOCSTRING,
)
class QuasarLongPreTrainedModel(PreTrainedModel):
    config_class = QuasarLongConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["QuasarLongDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # 1. Let super().from_pretrained load and instantiate the model normally
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        # 2. Check if we need to fuse MoE experts from separate parameters
        import os
        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download
        
        print(f"[FUSION LOADER] Post-loading MoE expert check/fusion for {pretrained_model_name_or_path}...", flush=True)
        try:
            repo_path = snapshot_download(pretrained_model_name_or_path, allow_patterns=["*.safetensors", "*.json"])
        except Exception as e:
            repo_path = str(pretrained_model_name_or_path)
            
        files = sorted([os.path.join(repo_path, f) for f in os.listdir(repo_path) if f.endswith(".safetensors")])
        if files:
            print(f"[FUSION LOADER] Analyzing safetensors for separate MoE weights in {repo_path}...", flush=True)
            expert_weights = {}
            has_unfused_experts = False
            
            for f in files:
                sd = load_file(f)
                for k, weight in sd.items():
                    if "mlp.experts." in k:
                        has_unfused_experts = True
                        parts = k.split(".")
                        if "layers" in parts and "experts" in parts:
                            layer_idx = int(parts[parts.index("layers") + 1])
                            expert_idx = int(parts[parts.index("experts") + 1])
                            proj_name = parts[parts.index("experts") + 2]
                            expert_weights[(layer_idx, expert_idx, proj_name)] = weight
                            
            if has_unfused_experts:
                print("[FUSION LOADER] Separate experts detected! Fusing in-flight...", flush=True)
                fused_sd = {}
                layer_indexes = sorted(list(set(k[0] for k in expert_weights.keys())))
                for l_idx in layer_indexes:
                    exp_indexes = sorted(list(set(k[1] for k in expert_weights.keys() if k[0] == l_idx)))
                    num_exp = len(exp_indexes)
                    if num_exp == 0:
                        continue
                    
                    print(f"  [FUSION LOADER] Fusing {num_exp} experts in layer {l_idx}...", flush=True)
                    gate_list = []
                    up_list = []
                    down_list = []
                    for e_idx in range(num_exp):
                        gate_list.append(expert_weights[(l_idx, e_idx, "gate_proj")].t())
                        up_list.append(expert_weights[(l_idx, e_idx, "up_proj")].t())
                        down_list.append(expert_weights[(l_idx, e_idx, "down_proj")].t())
                        
                    gate_stacked = torch.stack(gate_list)
                    up_stacked = torch.stack(up_list)
                    down_stacked = torch.stack(down_list)
                    
                    # Convert to the model's active dtype. During HF low-memory
                    # loading, parameters may still live on the meta device; in
                    # that case creating fused tensors on meta and calling a
                    # normal load_state_dict is a no-op, leaving MoE experts
                    # randomly materialized later. Keep real CPU tensors and
                    # assign them into the module below.
                    target_dtype = model.dtype
                    target_device = next(model.parameters()).device
                    if target_device.type == "meta":
                        target_device = torch.device("cpu")
                    
                    fused_sd[f"model.layers.{l_idx}.mlp.experts_w12"] = torch.cat([gate_stacked, up_stacked], dim=-1).to(device=target_device, dtype=target_dtype)
                    fused_sd[f"model.layers.{l_idx}.mlp.experts_w3"] = down_stacked.to(device=target_device, dtype=target_dtype)
                    
                print("[FUSION LOADER] Applying fused weights to the initialized model...", flush=True)
                info = model.load_state_dict(fused_sd, strict=False, assign=True)
                print(f"[FUSION LOADER] Post-load fusion complete! Missing: {len(info.missing_keys)}, Unexpected: {len(info.unexpected_keys)}", flush=True)
            else:
                print("[FUSION LOADER] Checkpoint already contains fused weights, skipping post-load fusion.", flush=True)
        else:
            print("[FUSION LOADER] No safetensors files found, skipping post-load fusion.", flush=True)
            
        return model

    def _init_weights(self, module):
        if getattr(module, "_skip_quasar_hf_init", False):
            return
        direct_params = list(module.parameters(recurse=False))
        direct_buffers = [buffer for buffer in module.buffers(recurse=False) if buffer is not None]
        if direct_params or direct_buffers:
            if all(getattr(param, "_is_hf_initialized", False) for param in direct_params) and all(
                getattr(buffer, "_is_hf_initialized", False) for buffer in direct_buffers
            ):
                module._is_hf_initialized = True
                return
        if not hasattr(self, "_init_count"):
            self._init_count = 0
        self._init_count += 1
        if self._init_count % 1000 == 0:
            print(f"  [MODEL INIT] Initializing module weights... ({self._init_count} modules processed)")
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, QuasarLongHybridReplacementSdpaAttention) and module.hybrid_enabled:
            module.replace_alpha_raw.data.fill_(float(getattr(self.config, "hybrid_alpha_init", -15.0)))
            module.branch_mix_logits.data.zero_()


QUASAR_LONG_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare QuasarLong Model outputting raw hidden-states without any specific head on top.",
    QUASAR_LONG_START_DOCSTRING,
)
class QuasarLongModel(QuasarLongPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`QuasarLongDecoderLayer`]

    Args:
        config: QuasarLongConfig
    """

    def __init__(self, config: QuasarLongConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_nextn_predict_layers = config.num_nextn_predict_layers

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = []
        for layer_idx in range(config.num_hidden_layers + config.num_nextn_predict_layers):
            if layer_idx % 1 == 0: # Print every layer for visibility
                print(f"[MODEL INIT] Building layer {layer_idx}/{config.num_hidden_layers + config.num_nextn_predict_layers-1}...")
            layer_cls = QuasarLongDecoderLayer if layer_idx < config.num_hidden_layers else QuasarLongMTPLayer
            self.layers.append(layer_cls(config, layer_idx))

        self.layers = nn.ModuleList(self.layers)

        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = QuasarLongRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = QuasarLongRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        print("[MODEL INIT] Finished building layers. Starting weight initialization (post_init)... this can take a few minutes for 20B models.")
        self.post_init()
        print("[MODEL INIT] Weight initialization complete.")

    def reset_hybrid_branch_parameters(self) -> None:
        for layer in self.layers:
            injection_gate = getattr(layer, "injection_gate", None)
            if injection_gate is not None:
                with torch.no_grad():
                    injection_gate.fill_(-6.907)
            attention = getattr(layer, "attention", None)
            reset = getattr(attention, "reset_hybrid_branch_parameters", None)
            if callable(reset):
                reset()
            if hasattr(layer, "engram") and layer.engram is not None:
                layer.engram._init_weights()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value

    @add_start_docstrings_to_model_forward(QUASAR_LONG_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        branch_past_key_values: Optional[QGRBranchCache] = None,
        branch_use_cache: bool = False,
        **kwargs,
    ) -> Union[Tuple, MoeV2ModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if branch_use_cache and branch_past_key_values is None:
            branch_past_key_values = QGRBranchCache(seen_tokens=past_seen_tokens)
        if branch_use_cache and past_seen_tokens == 0 and branch_past_key_values is not None:
            past_seen_tokens = int(branch_past_key_values.get_seq_length())

        if position_ids is None:
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
            position_ids = position_ids.unsqueeze(0)

        # ── Sample packing: position_ids resets -> block-diagonal mask + cu_seqlens.
        # When nothing is packed these stay None and the paths below are unchanged.
        packed_padding_mask = None
        packed_cu_seqlens = None
        attention_mask_2d = attention_mask if (attention_mask is not None and attention_mask.dim() == 2) else None
        is_packed = past_seen_tokens == 0 and _quasar_is_packed(position_ids, attention_mask_2d)
        if is_packed:
            seg_pos = position_ids
            if seg_pos.dim() == 1:
                seg_pos = seg_pos.unsqueeze(0)
            if seg_pos.shape[0] == 1 and batch_size > 1:
                seg_pos = seg_pos.expand(batch_size, -1)
            packed_cu_seqlens = _quasar_packed_cu_seqlens(seg_pos, attention_mask_2d)
            # A non-None mask (all ones when unpadded) keeps the fla branches on the
            # varlen B=1 path that cu_seqlens needs.
            if attention_mask_2d is None:
                packed_padding_mask = torch.ones(
                    batch_size, seq_length, dtype=torch.long, device=inputs_embeds.device
                )
            else:
                packed_padding_mask = attention_mask_2d

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            if is_packed:
                # Block-diagonal causal mask: each segment attends only within itself.
                attention_mask = _quasar_build_block_diag_sdpa_mask(
                    seg_pos, attention_mask_2d, inputs_embeds.dtype, inputs_embeds.device
                )
            else:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_seen_tokens,
                )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_seen_tokens
            )

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None
        layers = self.layers[: -self.num_nextn_predict_layers] if self.num_nextn_predict_layers > 0 else self.layers
        mtp_layers = self.layers[-self.num_nextn_predict_layers :] if self.num_nextn_predict_layers > 0 else None

        if os.environ.get("LOCAL_RANK", "0") == "0" and getattr(self, "_model_forward_debug", 0) < 1:
            self._model_forward_debug = 1
            print(f"[DEBUG RANK 0] QuasarLongModel.forward started: seq_len={seq_length}", flush=True)

        # ── Looped-Transformer: anchor embedding for injection mixing ─────────
        num_loops = max(1, int(getattr(self.config, "num_loops", 1)))
        use_looped_injection = bool(getattr(self.config, "use_looped_injection", False))
        use_parcae_loop_stabilizer = bool(getattr(self.config, "use_parcae_loop_stabilizer", False))
        collect_branch_mimic = self.training and (
            bool(getattr(self.config, "hybrid_attention_mimic_return_gqa", False))
            or bool(getattr(self.config, "hybrid_attention_collect_branch_loss", False))
        )
        branch_mimic_loss_accum = hidden_states.new_zeros((), dtype=torch.float32)
        branch_mimic_stats = None
        branch_mimic_count = 0
        if collect_branch_mimic:
            branch_mimic_stats = {
                "branch_loss_sums": {"quasar": 0.0, "raven": 0.0, "gla": 0.0, "mixed": 0.0},
                "branch_cos_sums": {"quasar": 0.0, "raven": 0.0, "gla": 0.0, "mixed": 0.0},
                "branch_rel_mse_sums": {"quasar": 0.0, "raven": 0.0, "gla": 0.0, "mixed": 0.0},
                "branch_loss_counts": {"quasar": 0, "raven": 0, "gla": 0, "mixed": 0},
                "skipped_distill": {"quasar": 0, "raven": 0, "gla": 0, "mixed": 0},
                "distill_count": 0,
            }
        # P is kept as the initial embedding; each layer can blend it back in.
        injection_anchor = hidden_states if (use_looped_injection or use_parcae_loop_stabilizer) else None

        for _loop_idx in range(num_loops):
            for decoder_layer in layers:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    # Bypassed full layer checkpointing to use layer-level selective checkpointing
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                        input_ids=input_ids,
                        injection_P=injection_anchor,
                        branch_past_key_values=branch_past_key_values,
                        branch_use_cache=branch_use_cache,
                        packed_cu_seqlens=packed_cu_seqlens,
                        packed_padding_mask=packed_padding_mask,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                        input_ids=input_ids,
                        injection_P=injection_anchor,
                        branch_past_key_values=branch_past_key_values,
                        branch_use_cache=branch_use_cache,
                        packed_cu_seqlens=packed_cu_seqlens,
                        packed_padding_mask=packed_padding_mask,
                    )
                hidden_states = layer_outputs[0]

                if collect_branch_mimic:
                    layer_branch_loss = layer_outputs[1]
                    layer_stats = layer_outputs[2]
                    layer_count = int(layer_stats.get("distill_count", 0))
                    if layer_count > 0:
                        branch_mimic_loss_accum = branch_mimic_loss_accum + layer_branch_loss
                        branch_mimic_count += 1
                    branch_mimic_stats["distill_count"] += layer_count
                    for stat_name in (
                        "branch_loss_sums",
                        "branch_cos_sums",
                        "branch_rel_mse_sums",
                        "branch_loss_counts",
                        "skipped_distill",
                    ):
                        for branch_name, value in layer_stats.get(stat_name, {}).items():
                            branch_mimic_stats[stat_name][branch_name] += value

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                if output_router_logits and layer_outputs[-1] is not None:
                    all_router_logits += (layer_outputs[-1],)



        hidden_states = self.norm(hidden_states)
        main_hidden_states = hidden_states

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (main_hidden_states,)

        mtp_hidden_states = None

        if mtp_layers:
            for decoder_layer in mtp_layers:
                input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1)
                inputs_embeds = self.word_embeddings(input_ids)

                if self.gradient_checkpointing and self.training:
                    # Bypassed full layer checkpointing to use layer-level selective checkpointing
                    layer_outputs = decoder_layer(
                        inputs_embeds,
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        inputs_embeds,
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        output_router_logits=output_router_logits,
                        use_cache=use_cache,
                        position_embeddings=position_embeddings,
                    )
                if mtp_hidden_states is None:
                    mtp_hidden_states = []
                hidden_states = layer_outputs[0]
                mtp_hidden_states.append(hidden_states)

                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                if output_router_logits and layer_outputs[-1] is not None:
                    all_router_logits += (layer_outputs[-1],)

        branch_mimic_loss = None
        if collect_branch_mimic:
            branch_mimic_loss = (
                branch_mimic_loss_accum / branch_mimic_count
                if branch_mimic_count > 0
                else branch_mimic_loss_accum
            )

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(
                v
                for v in [
                    main_hidden_states,
                    next_cache,
                    branch_past_key_values if branch_use_cache else None,
                    all_hidden_states,
                    all_self_attns,
                    all_router_logits,
                ]
                if v is not None
            )
        return MoeV2ModelOutputWithPast(
            last_hidden_state=main_hidden_states,
            past_key_values=next_cache,
            branch_past_key_values=branch_past_key_values if branch_use_cache else None,
            hidden_states=all_hidden_states,
            mtp_hidden_states=mtp_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
            branch_mimic_loss=branch_mimic_loss,
            branch_mimic_stats=branch_mimic_stats,
        )


class QuasarLongForCausalLM(QuasarLongPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: QuasarLongConfig):
        super().__init__(config)
        self.model = QuasarLongModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_nextn_predict_layers = config.num_nextn_predict_layers
        self.mtp_loss_scaling_factor = config.mtp_loss_scaling_factor

        # Initialize weights and apply final processing
        self.post_init()

    def reset_hybrid_branch_parameters(self) -> None:
        self.model.reset_hybrid_branch_parameters()

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(QUASAR_LONG_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoEV2CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logit_indices: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        branch_past_key_values: Optional[QGRBranchCache] = None,
        branch_use_cache: bool = False,
        **kwargs,
    ) -> Union[Tuple, MoEV2CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer

        >>> model = QuasarLongForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        fast_ce_labels = kwargs.pop("fast_ce_labels", None)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            branch_past_key_values=branch_past_key_values,
            branch_use_cache=branch_use_cache,
            **kwargs,
        )

        loss = None
        all_mtp_loss = None
        aux_loss = None
        hidden_states = outputs[0]
        skip_logits = (
            self.training
            and labels is None
            and bool(getattr(self.config, "branch_mimic_skip_logits", False))
            and not bool(getattr(self.config, "branch_mimic_compute_logits", True))
        )
        if fast_ce_labels is not None:
            if LigerFusedLinearCrossEntropyLoss is None:
                raise RuntimeError("fast_ce_labels requested but liger_kernel is not available")
            if not hasattr(self, "_quasar_liger_ce"):
                self._quasar_liger_ce = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)
            ce_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
            ce_target = fast_ce_labels.to(device=ce_hidden.device, dtype=torch.long).reshape(-1)
            loss = self._quasar_liger_ce(self.lm_head.weight, ce_hidden, ce_target)
            logits = hidden_states.new_empty((hidden_states.shape[0], hidden_states.shape[1], 0), dtype=torch.float32)
        elif skip_logits:
            logits = hidden_states.new_empty((hidden_states.shape[0], hidden_states.shape[1], 0), dtype=torch.float32)
        elif logit_indices is not None:
            if labels is not None:
                raise ValueError("labels are not supported with logit_indices")
            if logit_indices.shape[1] > hidden_states.shape[1]:
                raise ValueError(
                    f"logit_indices sequence length {logit_indices.shape[1]} exceeds hidden length {hidden_states.shape[1]}"
                )
            selected_hidden = hidden_states[:, : logit_indices.shape[1], :]
            flat_indices = logit_indices.to(device=self.lm_head.weight.device, dtype=torch.long).reshape(-1)
            selected_weight = self.lm_head.weight.index_select(0, flat_indices)
            selected_weight = selected_weight.view(*logit_indices.shape, selected_hidden.shape[-1])
            logits = torch.einsum("bsh,bskh->bsk", selected_hidden, selected_weight)
        else:
            if isinstance(logits_to_keep, int):
                hidden_for_logits = hidden_states[:, -logits_to_keep:, :] if logits_to_keep > 0 else hidden_states
            else:
                hidden_for_logits = hidden_states[:, logits_to_keep, :]
            logits = self.lm_head(hidden_for_logits)
        if labels is not None:
            logits = logits.float()
        if logits.numel() > 0 and labels is not None and not torch.isfinite(logits).all():
            rank = os.environ.get("LOCAL_RANK", "0")
            if rank == "0":
                finite_mask = torch.isfinite(logits)
                nonfinite_count = (~finite_mask).sum().item()
                if finite_mask.any():
                    # Safely extract min/max of finite elements without indexing
                    # Replace non-finite elements with large positive/negative values for min/max calculation
                    logits_for_min = torch.where(finite_mask, logits, torch.tensor(float('inf'), device=logits.device, dtype=logits.dtype))
                    logits_for_max = torch.where(finite_mask, logits, torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype))
                    print(
                        "[DEBUG RANK 0] Non-finite logits before loss: "
                        f"finite_min={logits_for_min.min().item():.4e} "
                        f"finite_max={logits_for_max.max().item():.4e} "
                        f"nonfinite={nonfinite_count}",
                        flush=True,
                    )
                else:
                    print("[DEBUG RANK 0] Non-finite logits before loss: all logits non-finite", flush=True)

        if labels is not None:
            # --- LOSS DEBUG ---
            if os.environ.get("LOCAL_RANK", "0") == "0" and getattr(self, "_loss_debug_count", 0) < 5:
                self._loss_debug_count = getattr(self, "_loss_debug_count", 0) + 1
                print(f"[DEBUG RANK 0] Step {self._loss_debug_count}: labels[:5]={labels.reshape(-1)[:5].tolist()}, vocab={self.config.vocab_size}", flush=True)
                # Check if labels are all -100
                if (labels == -100).all():
                    print("[DEBUG RANK 0] WARNING: All labels are -100! Loss will be 0.", flush=True)
            
            loss = self.loss_function(logits, labels, self.config.vocab_size, **kwargs)
            
            if os.environ.get("LOCAL_RANK", "0") == "0" and getattr(self, "_loss_debug_count", 0) <= 5:
                print(f"[DEBUG RANK 0] Calculated loss: {loss.item() if loss is not None else 'None'}", flush=True)

        all_mtp_logits = None
        if self.num_nextn_predict_layers > 0:
            mtp_hidden_states = outputs.mtp_hidden_states
            shift_labels_mtp = None
            keep_mtp_logits = (not self.training) or (labels is None and fast_ce_labels is None)
            for i in range(self.num_nextn_predict_layers):
                mtp_hidden_states = mtp_hidden_states[i]
                mtp_logits = self.lm_head(mtp_hidden_states)
                if keep_mtp_logits:
                    if all_mtp_logits is None:
                        all_mtp_logits = []
                    all_mtp_logits.append(mtp_logits)
                if labels is not None:
                    if shift_labels_mtp is None:
                        shift_labels_mtp = labels.clone()
                    shift_labels_mtp, _ = roll_tensor(shift_labels_mtp, shifts=-1, dims=-1, fill_value=-100)
                    mtp_logits_ = mtp_logits.view(-1, self.config.vocab_size)
                    mtp_loss = self.loss_function(mtp_logits_, shift_labels_mtp.to(mtp_logits_.device).view(-1), self.config.vocab_size, **kwargs)
                    if loss is not None:
                        loss += self.mtp_loss_scaling_factor * mtp_loss
                    else:
                        loss = self.mtp_loss_scaling_factor * mtp_loss

                    if all_mtp_loss is None:
                        all_mtp_loss = []
                    all_mtp_loss.append(mtp_loss)
                del mtp_logits

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoEV2CausalLMOutputWithPast(
            loss=loss,
            mtp_loss=all_mtp_loss,
            aux_loss=aux_loss,
            branch_mimic_loss=getattr(outputs, "branch_mimic_loss", None),
            branch_mimic_stats=getattr(outputs, "branch_mimic_stats", None),
            logits=logits,
            mtp_logits=all_mtp_logits,
            past_key_values=outputs.past_key_values,
            branch_past_key_values=getattr(outputs, "branch_past_key_values", None),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
