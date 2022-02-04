import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES']="6"

import numpy as np
from datasets import load_dataset, load_metric
import torch
import transformers

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
print('this came to custome model pick nice***************************************************************************')


import sys
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

import copy
import math
import random
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutput2,
    BaseModelOutput3,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_bart import BartConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"


BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-large",
    # See all BART models at https://huggingface.co/models?filter=bart
]





def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


class TransformerSequenceClassifier(torch.nn.Module):

    def __init__(self, num_labels, pretrained_name, dropout=0.1):
        super().__init__()
        self.num_labels = num_labels
        # load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(pretrained_name)
        # initialize other layers (head after the transformer body)
        self.pre_classifier = torch.nn.Linear(self.transformer.config.dim, self.transformer.config.dim)
        self.classifier = torch.nn.Linear(self.transformer.config.dim, num_labels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids=None, **kwargs):
        # get text representation from transformer
        transformer_output = self.transformer(
            input_ids=input_ids,
            **kwargs,
        )
        hidden_state = transformer_output[0]                # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]                  # (bs, dim)
        # apply classification layers
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = F.relu(pooled_output)               # (bs, dim)
        pooled_output = self.dropout(pooled_output)         # (bs, dim)
        output = self.classifier(pooled_output)             # (bs, num_labels)
        return output




class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        assert padding_idx is not None, "`padding_idx` should not be None, but of type int"
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)




class BartPretrainedModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        print('inside dummay')
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class PretrainedBartModel(BartPretrainedModel):
    def __init_subclass__(self):
        warnings.warn(
            "The class `PretrainedBartModel` has been depreciated, please use `BartPretrainedModel` instead.",
            FutureWarning,
        )




class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
        self.scaling = self.head_dim ** -0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        # print(hidden_states.shape,'hidden_states')
        # print(self.scaling,'self/scaling',embed_dim,'embed_dim',self.q_proj)
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        
        assert attn_weights.size() == (
            bsz * self.num_heads,
            tgt_len,
            src_len,
        ), f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"

        if attention_mask is not None:
            assert attention_mask.size() == (
                bsz,
                1,
                tgt_len,
                src_len,
            ), f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        # print(attn_weights.size(),'attn_weights.size()')

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_heads,
            ), f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        assert attn_output.size() == (
            bsz * self.num_heads,
            tgt_len,
            self.head_dim,
        ), f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"

        attn_output = (
            attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(bsz, tgt_len, embed_dim)
        )

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value





class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        
        # print(hidden_states.shape,'hidden_states')
        # print(attention_mask.shape,'attention_mask')
        # # print(layer_head_mask.size,'layer_head_mask')
        # print(output_attentions,output_attentions)
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class BartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    :class:`BartEncoderLayer`.

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        enc=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # print(input_ids.shape,'input_ids')
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            # print(inputs_embeds.shape,'inputs_embeds')
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        # print(input_ids.shape,'input_ids shape shape shape shape')
        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False) and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]
            # print(output_attentions,'output_attentions') False
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        # print(output_hidden_states,'output_hidden_states') False
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        # print(encoder_states,'len encoder_states')
        # print(all_attentions,'len all_attentions')
        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        if(enc=='one'):
            return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            )
        if(enc=='two'):
            return BaseModelOutput2(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions2=all_attentions
            )
        if(enc=='three'):
            return BaseModelOutput3(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions3=all_attentions
            )


class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn1 = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn2 = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn3 = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )

        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decsion_from_encoder: Optional[torch.Tensor] = None,
        encoder_hidden_states1: Optional[torch.Tensor] = None,
        encoder_hidden_states2: Optional[torch.Tensor] = None,
        encoder_hidden_states3: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask2: Optional[torch.Tensor] = None,
        encoder_attention_mask3: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        encoder_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`torch.FloatTensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        # print(hidden_states.shape,'decoder_hidden_states')
        # print(residual.shape,'residual')

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states1 is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple

            cross_attn_past_key_value = past_key_value[2:4] if past_key_value is not None else None
            hidden_states1, cross_attn_weights1, cross_attn_present_key_value1 = self.encoder_attn1(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states1,
                attention_mask=encoder_attention_mask,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states1 = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states1 = residual + hidden_states1

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value1#((0,1)self_sttention,(2,3)cross_attention)

        if encoder_hidden_states2 is not None:
            # residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple


            cross_attn_past_key_value = past_key_value[4:6] if past_key_value is not None else None
            hidden_states2, cross_attn_weights2, cross_attn_present_key_value2 = self.encoder_attn2(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states2,
                attention_mask=encoder_attention_mask2,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states2 = F.dropout(hidden_states2, p=self.dropout, training=self.training)
            hidden_states2 = residual + hidden_states2

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value2#((0,1)self_sttention,(2,3)cross_attention)
        if encoder_hidden_states3 is not None:
            # residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[6:8] if past_key_value is not None else None

            hidden_states3, cross_attn_weights3, cross_attn_present_key_value3 = self.encoder_attn3(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states3,
                attention_mask=encoder_attention_mask3,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )

            hidden_states3 = F.dropout(hidden_states3, p=self.dropout, training=self.training)
            hidden_states3 = residual + hidden_states3
            hidden_states = self.encoder_attn_layer_norm(hidden_states1+hidden_states2+hidden_states3)
            

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value3#((0,1)self_sttention,(2,3)cross_attention)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states 
        hidden_states = self.final_layer_norm(hidden_states)
        # print(hidden_states.shape,'hidden_states shape')

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs



class BartDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`BartDecoderLayer`

    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            self.padding_idx,
        )
        self.layers = nn.ModuleList([BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids=None,
        input_ids2=None,
        input_ids3=None,
        attention_mask=None,
        encoder_hidden_states1=None,#last_hidden_states
        encoder_hidden_states2=None,#last_hidden_states
        encoder_hidden_states3=None,#last_hidden_states
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                for details.

                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            encoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # print(output_attentions,'output_attentions_decoder')
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # print(output_hidden_states,'output_hidden_states_deocder') False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")


        # if input_ids2 is not None and inputs_embeds2 is not None:
        #     raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        # elif input_ids2 is not None:
        #     input_shape2 = input_ids2.size()
        #     input_ids2 = input_ids2.view(-1, input_shape2[-1])
        # elif inputs_embeds2 is not None:
        #     input_shape2 = inputs_embeds2.size()[:-1]
        # else:
        #     raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")


        # if input_ids3 is not None and inputs_embeds3 is not None:
        #     raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        # elif input_ids3 is not None:
        #     input_shape3 = input_ids3.size()
        #     input_ids3 = input_ids3.view(-1, input_shape3[-1])
        # elif inputs_embeds3 is not None:
        #     input_shape3 = inputs_embeds.size()[:-1]
        # else:
        #     raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")




        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # print(past_key_values,'past_key_values')

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )
        # attention_mask2 = self._prepare_decoder_attention_mask(
        #     attention_mask2, input_shape2, inputs_embeds, past_key_values_length
        # )
        # attention_mask3 = self._prepare_decoder_attention_mask(
        #     attention_mask3, input_shape3, inputs_embeds, past_key_values_length
        # )


        # expand encoder attention mask
        if encoder_hidden_states1 is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        # hidden_states=torch.add(hidden_states,decsion_from_encoder)
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None #None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states1 is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            # print(past_key_value,'past_key_value')

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_head_mask[idx] if encoder_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    # decsion_from_encoder=decsion_from_encoder,
                    encoder_hidden_states1=encoder_hidden_states1,
                    encoder_hidden_states2=encoder_hidden_states2,
                    encoder_hidden_states3=encoder_hidden_states3,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]
            # print(use_cache,'use_cache_decoder') True
            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            # print(output_attentions,'output_attentions_decoder') False
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states1 is not None:  #None
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        # print(hidden_states.shape,'hidden_states sahpe')
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )




class BartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder1 = BartEncoder(config, self.shared)
        self.encoder2 = BartEncoder(config, self.shared)
        self.encoder3 = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
        # self.m = nn.AdaptiveAvgPool2d((1,768))



        # self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        # self.pre_classifier = torch.nn.Linear(config.d_model, config.d_model)
        # self.classifier = torch.nn.Linear(config.d_model, 1)
        # self.dropout = torch.nn.Dropout(0.6)


        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder1.embed_tokens = self.shared
        self.encoder2.embed_tokens = self.shared
        self.encoder3.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return [self.encoder1,self.encoder2,self.encoder3]
    # def get_encoder2(self):
    #     return self.encoder2
    # def get_encoder3(self):
    #     return self.encoder3

    def get_decoder(self):
        return self.decoder

    # @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint="facebook/bart-large",
    #     output_type=Seq2SeqModelOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_ids2=None,
        attention_mask2=None,
        input_ids3=None,
        attention_mask3=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs1=None,
        encoder_outputs2=None,
        encoder_outputs3=None,
        past_key_values=None,
        inputs_embeds1=None,
        inputs_embeds2=None,
        inputs_embeds3=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print(attention_mask.shape,attention_mask2.shape,attention_mask3.shape)
        if encoder_outputs1 is None:
            # print('gone_once')
            encoder_outputs1 = self.encoder1(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds1,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                enc='one',
            )

        # If the user passed a tuple for encoder_1outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs1, BaseModelOutput):
            encoder_outputs1 = BaseModelOutput(
                last_hidden_state=encoder_outputs1[0],
                hidden_states=encoder_outputs1[1] if len(encoder_outputs1) > 1 else None,
                attentions=encoder_outputs1[2] if len(encoder_outputs1) > 2 else None,
            )
        # print(type(encoder_outputs1[0]),'encoder_outputs1')
        if encoder_outputs2 is None:
            encoder_outputs2 = self.encoder2(
                input_ids=input_ids2,
                attention_mask=attention_mask2,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds2,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                enc='two',
            )

        # If the user passed a tuple for encoder_1outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs2, BaseModelOutput):
            encoder_outputs2 = BaseModelOutput(
                last_hidden_state=encoder_outputs1[0],
                hidden_states=encoder_outputs2[1] if len(encoder_outputs2) > 1 else None,
                attentions=encoder_outputs2[2] if len(encoder_outputs2) > 2 else None,
            )
        if encoder_outputs3 is None:
            encoder_outputs3 = self.encoder3(
                input_ids=input_ids3,
                attention_mask=attention_mask3,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds3,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                enc='three',
            )

        # If the user passed a tuple for encoder_1outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs3, BaseModelOutput):
            encoder_outputs3 = BaseModelOutput(
                last_hidden_state=encoder_outputs3[0],
                hidden_states=encoder_outputs3[1] if len(encoder_outputs3) > 1 else None,
                attentions=encoder_outputs3[2] if len(encoder_outputs3) > 2 else None,
            )

        # pooled_encoder_outputs1=self.m(encoder_outputs1[0])
        # pooled_encoder_outputs2=self.m(encoder_outputs2[0])
        # pooled_encoder_outputs3=self.m(encoder_outputs3[0])

        # print(encoder_outputs3[0].shape,encoder_outputs2[0].shape,encoder_outputs1[0].shape,'encoder_outputs')
        # print(pooled_encoder_outputs1.shape,pooled_encoder_outputs2.shape,pooled_encoder_outputs3.shape,'pooled_output')
        # sys.exit(0)
        encoder_outputs=BaseModelOutput(
            # last_hidden_state=pooled_encoder_outputs1+pooled_encoder_outputs2+pooled_encoder_outputs3,
            last_hidden_state1=encoder_outputs1[0],
            last_hidden_state2=encoder_outputs2[0],
            last_hidden_state3=encoder_outputs3[0],
            hidden_states=encoder_outputs1[0]+encoder_outputs2[0]+encoder_outputs3[0] if len(encoder_outputs3) > 1 else None,
            attentions=encoder_outputs1[0]+encoder_outputs2[0]+encoder_outputs3[0] if len(encoder_outputs3) > 2 else None,
        )
        # one=len(encoder_outputs[0])
        # two=len(encoder_outputs[1]) if encoder_outputs[1] is not None else None
        # three=len(encoder_outputs[2]) if encoder_outputs[2] is not None else None
        # print(one ,'one_two_three')
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            # decsion_from_encoder=encoder_outputs[0],
            encoder_hidden_states1=encoder_outputs1[0],
            encoder_hidden_states2=encoder_outputs2[0],
            encoder_hidden_states3=encoder_outputs3[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state1=encoder_outputs1.last_hidden_state,
            encoder_last_hidden_state2=encoder_outputs2.last_hidden_state,
            encoder_last_hidden_state3=encoder_outputs3.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class BartForConditionalGeneration(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: BartConfig):
        print('--------check_bart--------pre_repo')
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.pre_classifier = torch.nn.Linear(config.d_model, config.d_model)
        self.classifier = torch.nn.Linear(config.d_model, 1)
        self.dropout = torch.nn.Dropout(0.6)

        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_ids2=None,
        attention_mask2=None,
        input_ids3=None,
        attention_mask3=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs1=None,
        encoder_outputs2=None,
        encoder_outputs3=None,
        past_key_values=None,
        inputs_embeds1=None,
        inputs_embeds2=None,
        inputs_embeds3=None,
        decoder_inputs_embeds=None,
        labels=None,
        decision_label=None,
        indexs=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )


        # print(input_ids.shape)
        # print(input_ids2.shape)
        # print(input_ids3.shape)
        # sys.exit(0)


        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_ids2=input_ids2,
            attention_mask2=attention_mask2,
            input_ids3=input_ids3,
            attention_mask3=attention_mask3,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs1=encoder_outputs1,
            encoder_outputs2=encoder_outputs2,
            encoder_outputs3=encoder_outputs3,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds1=inputs_embeds1,
            inputs_embeds2=inputs_embeds2,
            inputs_embeds3=inputs_embeds3,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        

        # hidden_state = outputs[0]                # (bs, seq_len, dim)
        # pooled_output = hidden_state[:, 0]                  # (bs, dim)
        # # apply classification layers
        # pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        # pooled_output = F.relu(pooled_output)               # (bs, dim)
        # pooled_output = self.dropout(pooled_output)         # (bs, dim)
        # decision_output = self.classifier(pooled_output)             # (bs, num_labels)
        # # decision_output=torch.sigmoid(decision_output.view(1))
        
        
        decision_output=torch.ones(1,1)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # decision_loss= torch.nn.BCEWithLogitsLoss()#torch.nn.BCELoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
            # decision_label = decision_label.type_as(decision_output)
            # print(decision_output,'decision_output')
            # print(decision_label,'decision_label')
            # masked_lm_loss+=decision_loss(decision_output.reshape(-1,1),decision_label.reshape(-1,1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # return ([masked_lm_loss,
        #     lm_logits,
        #     outputs.past_key_values,
        #     outputs.decoder_hidden_states,
        #     outputs.decoder_attentions,
        #     outputs.cross_attentions,
        #     outputs.encoder_last_hidden_state,
        #     outputs.encoder_hidden_states,
        #     outputs.encoder_attentions,
        #     ])
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            decision_label=decision_output,
            indexs=indexs,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state1=outputs.encoder_last_hidden_state1,
            encoder_last_hidden_state2=outputs.encoder_last_hidden_state2,
            encoder_last_hidden_state3=outputs.encoder_last_hidden_state3,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        input_ids=None,
        input_ids2=None,
        input_ids3=None,
        past=None,
        attention_mask=None,
        attention_mask2=None,
        attention_mask3=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs1=None,
        encoder_outputs2=None,
        encoder_outputs3=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        # print(len(encoder_outputs),'encoder_outputs')

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "input_ids2": None,  # encoder_outputs is defined. input_ids not needed
            "input_ids3": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs1": encoder_outputs1,
            "encoder_outputs2": encoder_outputs2,
            "encoder_outputs3": encoder_outputs3,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "attention_mask2": attention_mask2,
            "attention_mask3": attention_mask3,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1 and self.config.force_bos_token_to_be_generated:
            self._force_token_id_to_be_generated(logits, self.config.bos_token_id)
        elif cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past




class TransformerSequenceClassifier(torch.nn.Module):

    def __init__(self, num_labels, pretrained_name, dropout=0.1):
        super().__init__()
        self.num_labels = num_labels
        # load pre-trained transformer
        self.transformer = AutoModel.from_pretrained(pretrained_name)
        # initialize other layers (head after the transformer body)
        self.pre_classifier = torch.nn.Linear(self.transformer.config.dim, self.transformer.config.dim)
        self.classifier = torch.nn.Linear(self.transformer.config.dim, num_labels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids=None, **kwargs):
        # get text representation from transformer
        transformer_output = self.transformer(
            input_ids=input_ids,
            **kwargs,
        )
        hidden_state = transformer_output[0]                # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]                  # (bs, dim)
        # apply classification layers
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = F.relu(pooled_output)               # (bs, dim)
        pooled_output = self.dropout(pooled_output)         # (bs, dim)
        output = self.classifier(pooled_output)             # (bs, num_labels)
        return output





if __name__ == '__main__':
    # initialize model and corresponding tokenizer
    pretrained_name = "distilbert-base-uncased"
    model = TransformerSequenceClassifier(2, pretrained_name)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    # apply model to some example sentences
    batch = tokenizer(
        ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    y_pred = model(**batch)