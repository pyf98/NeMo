# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List, Optional, Tuple

import torch
import librosa
from lhotse.dataset.collation import collate_vectors as collate_vectors_lhotse

import nemo.collections.nlp.modules.common.text_generation_strategy as text_generation_strategy
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import shift_tokens_by_multi_audios
from nemo.collections.nlp.modules.common.megatron.utils import build_position_ids
from nemo.collections.nlp.modules.common.lm_utils import pad_batch

# the text representation of eos_id, it applies for all tokenizers
END_OF_SEQ = '<|endoftext|>'


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    boolean = boolean.unsqueeze(0).unsqueeze(-1)
    return (1 - boolean) * val1 + boolean * val2


class AudioToTextGenerationStrategy(text_generation_strategy.GPTModelTextGenerationStrategy):
    def init_batch(
        self,
        context_tokens: torch.Tensor,
        context_lengths: torch.Tensor,
        audio_signal: torch.Tensor,
        audio_length: torch.Tensor,
        compute_attention_mask: bool,
        audio_locator_ids: torch.Tensor,
        tokens_to_generate: int,
    ):
        """initialize the batch data before the inference steps."""

        audio_feats, audio_feat_lens = self.model.perception(
            input_signal=audio_signal,
            input_signal_length=audio_length,
            processed_signal=None,
            processed_signal_length=None,
        )

        encoder_input, encoder_length, labels, loss_mask, attention_mask, position_ids = self.model.inject_perception_input(
            encoded=audio_feats,
            encoded_len=audio_feat_lens,
            input_ids=context_tokens,
            input_length=context_lengths,
            loss_mask=torch.empty(context_tokens.shape, dtype=torch.bool, device=context_tokens.device).fill_(1),     # dummy
            audio_locator_ids=audio_locator_ids,
            remove_bos_or_eos=False,
        )

        # pad to max len including tokens_to_generate
        encoder_input = torch.nn.functional.pad(encoder_input, (0, 0, 0, 0, 0, tokens_to_generate), value=0.)  # (T, B, D)
        labels = torch.nn.functional.pad(labels, (0, tokens_to_generate), value=self.model.tokenizer.pad_id)    # (B, T)

        self.attention_mask = self.model._create_attention_mask(encoder_input.transpose(0, 1))
        self.position_ids = build_position_ids(encoder_input[:, :, 0].transpose(0, 1))

        return labels, encoder_input, encoder_length

    def clip_max_len(self, maxlen: int) -> int:
        """clip the max len based on the LM model max sequence length"""
        # for positional embedding types that allow length extrapolation, don't clip the max length
        if self.model.cfg.get("position_embedding_type", "learned_absolute") == "learned_absolute":
            if maxlen > self.model.cfg.encoder_seq_length + 1:
                maxlen = self.model.cfg.encoder_seq_length + 1
        return maxlen

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        input_embeddings: torch.Tensor,
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_lengths: torch.Tensor,
        curr_context_length: int,
        compute_attention_mask: bool,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        # types2use = None
        if step == 0:
            # Allocate memory for the entire context.
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :curr_context_length]
            positions2use = self.position_ids[:, :curr_context_length]
            embeddings2use = input_embeddings[:curr_context_length]
        else:
            # Set this to false so the memory is not reallocated.
            set_inference_key_value_memory = False
            tokens2use = tokens[:, curr_context_length - 1].view(micro_batch_size, -1)
            positions2use = self.position_ids[:, curr_context_length - 1].view(micro_batch_size, -1)
            embeddings2use = self.model._get_text_embeddings(tokens2use, positions2use)
            started = context_lengths <= curr_context_length
            embeddings2use = switch(input_embeddings[curr_context_length - 1].unsqueeze(0), embeddings2use, started)

        """Prepare batch for each of the inference steps"""
        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

        batch = [tokens2use, embeddings2use, self.attention_mask, positions2use, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape

    def post_process(self, tokens: torch.Tensor, new_tokens: torch.Tensor, context_length: int):
        """
        At the end of the inference, post process the inference results
        """
        pass

    def end_of_generation_condition(
        self, tokens: torch.Tensor, prev: torch.Tensor, eod_id: int, end_strings: List[str]
    ) -> torch.Tensor:
        """
        return whether the generation should stop based on the previous token
        Args:
            tokens (torch.Tensor): the generated tokens so far
            prev  (torch.Tensor): the previous token
            eod_id (int): the end of document token id
            end_strings (List[str]): the list of end of generation strings
        returns:
            a boolean tensor indicating whether the generation should stop
        """
        if len(end_strings) == 1 and end_strings[0] == END_OF_SEQ:
            return prev == eod_id
        else:
            tokenizer = self.model.tokenizer
            conditions = []
            end_tokens = set()
            end_tokens.add(eod_id)
            for end_string in end_strings:
                if len(end_string) > 1:
                    continue
                ids_1 = tokenizer.text_to_ids(f'<extra_id_1>{end_string}')
                ids_2 = tokenizer.text_to_ids('<extra_id_1>')
                if len(ids_1) <= len(ids_2):
                    continue
                token_id = ids_1[len(ids_2) :][0]

                end_tokens.add(token_id)

            for p, token_item in zip(prev, tokens):
                text = tokenizer.ids_to_text(token_item.tolist())
                conditions.append(
                    any([text.endswith(end_string) for end_string in end_strings] + [p.item() in end_tokens])
                )
            return torch.tensor(conditions, dtype=torch.bool, device=tokens.device)

    def tokenize_batch(self, inputs, add_BOS):
        """Tokenize input texts, load input audios, pad them to the same length, and add bos to text if needed.

        Args:
            inputs (List): input_texts, input_audios, audio_locator
            add_BOS (bool): whether to add the BOS token at the beginning

        Returns:
            Tuple[torch.Tensor], context_tokens_tensor, context_length_tensor, audio_signal, audio_signal_length, audio_locator_ids
        """

        input_texts, input_audios, audio_locator = inputs
        tokenizer = self.model.tokenizer
        
        # tokenize and pad text
        if add_BOS:
            context_tokens = [[tokenizer.bos_id] + tokenizer.text_to_ids(s) for s in input_texts]
        elif hasattr(tokenizer.tokenizer, "get_prefix_tokens"):
            # chatglm: add tokenizer.gmask_id, tokenizer.sop_id
            context_tokens = [tokenizer.tokenizer.get_prefix_tokens() + tokenizer.text_to_ids(s) for s in input_texts]
        else:
            context_tokens = [tokenizer.text_to_ids(s) for s in input_texts]
        context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.pad_id, 0)
        context_tokens_tensor = torch.tensor(context_tokens, dtype=torch.long, device="cuda")
        context_length_tensor = torch.tensor(context_lengths, dtype=torch.long, device="cuda")

        # load and pad audio
        sr = self.model.cfg.perception.preprocessor.sample_rate
        audio_signal = [librosa.load(p, sr=sr)[0] for p in input_audios]
        audio_signal_length = torch.tensor(
            [len(audio) for audio in audio_signal], dtype=torch.long, device="cuda"
        )
        audio_signal = collate_vectors_lhotse(audio_signal, padding_value=0.).cuda()

        audio_locator_ids = torch.tensor(
            tokenizer.text_to_ids(audio_locator), dtype=torch.long, device="cuda"
        )

        return context_tokens_tensor, context_length_tensor, audio_signal, audio_signal_length, audio_locator_ids


class CrossAttendAudioToTextGenerationStrategy(AudioToTextGenerationStrategy):
    def init_batch(
        self,
        context_tokens: torch.Tensor,
        context_lengths: torch.Tensor,
        audio_signal: torch.Tensor,
        audio_length: torch.Tensor,
        compute_attention_mask: bool,
        num_audios: Optional[torch.Tensor] = None,
        context_start_idx: Optional[List[List[int]]] = None,
    ):
        """initialize the batch data before the inference steps."""
        # Move to GPU.
        batch = {
            'audio_signal': audio_signal,
            'audio_signal_length': audio_length,
            'tokens': context_tokens,
            'tokens_length': context_lengths,
            'labels': context_tokens,
            'loss_mask': None,
        }
        if self.model.perception.cfg.get('combine_return', True):
            (
                encoder_input,
                self.attention_mask,
                context_tokens,
                _,
                (speech_encoded, speech_encoded_len, extra_outputs),
            ) = self.model.prepare_llm_input(batch)
            self.position_ids = build_position_ids(encoder_input[:, :, 0].transpose(0, 1))
            self.extra_outputs = extra_outputs
            return (
                context_tokens,
                (encoder_input, speech_encoded, speech_encoded_len),
                torch.zeros_like(context_lengths),
            )
        else:
            (
                encoder_input,
                self.attention_mask,
                context_tokens,
                _,
                (speech_encoded, speech_encoded_len, llm_encoded_len, extra_outputs),
            ) = self.model.prepare_llm_input(batch)
            self.position_ids = build_position_ids(encoder_input[:, :, 0].transpose(0, 1))
            self.extra_outputs = extra_outputs
            return context_tokens, (encoder_input, speech_encoded, speech_encoded_len), llm_encoded_len

    def prepare_batch_at_step(
        self,
        tokens: torch.Tensor,
        input_embeddings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        maxlen: int,
        micro_batch_size: int,
        step: int,
        context_lengths: torch.Tensor,
        curr_context_length: int,
        compute_attention_mask: bool,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        # types2use = None
        self.input_embeds_hidden = self.extra_outputs.get('input_embeds_hidden', None)
        input_embeddings, speech_encoded, speech_encoded_len = input_embeddings
        if step == 0:
            # Allocate memory for the entire context.
            set_inference_key_value_memory = True
            tokens2use = tokens[:, :curr_context_length]
            positions2use = self.position_ids[:, :curr_context_length]
            embeddings2use = input_embeddings[:curr_context_length]
        else:
            # Set this to false so the memory is not reallocated.
            set_inference_key_value_memory = False
            tokens2use = tokens[:, curr_context_length - 1].view(micro_batch_size, -1)
            positions2use = self.position_ids[:, curr_context_length - 1].view(micro_batch_size, -1)
            embeddings2use = self.model._get_text_embeddings(tokens2use, positions2use).transpose(0, 1)
            started = context_lengths <= curr_context_length
            # for seq started, first get embeddings2use, and then run cross attend, after that replace embeddings2use with the cross attended embed
            # use speech_encoded; rerun cross attend
            # [1, b, d]
            decoder_mems_list = self.extra_outputs.get('decoder_mems_list', None)
            if decoder_mems_list is not None:
                decoder_mems_list = decoder_mems_list[:, :, : curr_context_length - 1]
            # need to use audio_ratio field if to support text-only decoding
            embeddings2use, self.extra_outputs = self.model.perception_cross_attn(
                speech_encoded,
                speech_encoded_len,
                embeddings2use,
                input_lengths=tokens2use.squeeze(-1) != self.model.tokenizer.eos_id,
                decoder_mems_list=decoder_mems_list,
                return_mems=True,
            )
            self.input_embeds_hidden = self.extra_outputs.get('input_embeds_hidden', None)
            embeddings2use = switch(
                input_embeddings[curr_context_length - 1].unsqueeze(0), embeddings2use.transpose(0, 1), started
            )

        """Prepare batch for each of the inference steps"""
        setkey_value_array = torch.tensor(
            [set_inference_key_value_memory] * micro_batch_size, device=torch.cuda.current_device()
        )
        len_array = torch.tensor([maxlen] * micro_batch_size, device=torch.cuda.current_device())

        batch = [tokens2use, embeddings2use, self.attention_mask, positions2use, setkey_value_array, len_array]
        tensor_shape = [tokens2use.shape[1], micro_batch_size, self.model.cfg.hidden_size]
        return batch, tensor_shape


def model_inference_strategy_dispatcher(model, **args):
    from nemo.collections.multimodal.speech_llm.models.modular_models import (
        CrossAttendModularAudioGPTModel,
        ModularAudioGPTModel,
    )

    if isinstance(model, CrossAttendModularAudioGPTModel):
        return CrossAttendAudioToTextGenerationStrategy(model, **args)
    elif isinstance(model, ModularAudioGPTModel):
        return AudioToTextGenerationStrategy(model, **args)
    else:
        return text_generation_strategy.model_inference_strategy_dispatcher(model, **args)
