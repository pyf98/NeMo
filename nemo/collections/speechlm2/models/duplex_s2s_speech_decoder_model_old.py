# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os
import random
import tempfile

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from peft import PeftModel
from torch import Tensor, nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import DynamicCache

from nemo.collections.audio.parts.utils.resampling import resample
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.models.duplex_s2s_model import replace_control_speech_codes, tokens_to_str
from nemo.collections.speechlm2.modules import EOUDecoder, EOUDecoderFromWav, TransformerARSpeechDecoder
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.metrics.asr_bleu import ASRBLEU
from nemo.collections.speechlm2.parts.metrics.bleu import BLEU
from nemo.collections.speechlm2.parts.metrics.results_logger import ResultsLogger
from nemo.collections.speechlm2.parts.metrics.token_accuracy import TokenAccuracy
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.speechlm2.parts.pretrained import (
    load_pretrained_hf,
    set_model_dict_for_partial_init,
    setup_audio_codec,
    setup_speech_encoder,
)
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType
from nemo.utils import logging


def delay_eos(tokens, eos_token_id, pad_token_id, shift=10):
    """
    Delays each EOS token by `shift` steps forward. Replaces original EOS with PAD.
    Skips move if it would go out of bounds or overwrite another EOS/PAD.
    Safe for GPU execution.
    """
    B, T = tokens.shape
    tokens = tokens.clone()
    device = tokens.device

    # Find all EOS positions
    eos_mask = tokens == eos_token_id
    if not eos_mask.any():
        return tokens

    # Flattened indices of EOS tokens
    eos_indices = eos_mask.nonzero(as_tuple=False)  # [N, 2]
    b_idx = eos_indices[:, 0]  # [N]
    eos_pos = eos_indices[:, 1]  # [N]
    new_pos = eos_pos + shift  # [N]

    # Filter: new position must be in bounds and not overwrite EOS or PAD
    valid = (new_pos < T)
    if valid.any():
        b_idx = b_idx[valid]
        old_pos = eos_pos[valid]
        new_pos = new_pos[valid]

        # Now, check overwrite safety in new positions
        target_vals = tokens[b_idx, new_pos]
        safe = (target_vals != eos_token_id)

        if safe.any():
            b_idx = b_idx[safe]
            old_pos = old_pos[safe]
            new_pos = new_pos[safe]
            # Move EOS token: clear original, set new
            tokens[b_idx, old_pos] = pad_token_id
            tokens[b_idx, new_pos] = eos_token_id
    return tokens


def generate_multiturn_speaking_mask(input_ids: torch.Tensor, bos_token_id: int = 0, eos_token_id: int = 1):
    """
    Efficient, batched speaking mask generator that marks 1 between <bos> and <eos> pairs.
    If <eos> is missing after a <bos>, mask continues to end. Handles multiple turns.

    Args:
        input_ids (torch.Tensor): LongTensor of shape (B, T)
        bos_token_id (int): Token ID for <bos>
        eos_token_id (int): Token ID for <eos>

    Returns:
        torch.Tensor: FloatTensor of shape (B, T), with 1.0 for speaking, 0.0 for silence.

    Note BOS is considered as speaking (1) and EOS as non speaking 0
    """
    B, T = input_ids.shape
    device = input_ids.device
    bos_mask = (input_ids == bos_token_id).to(torch.int32).to(device)
    eos_mask = (input_ids == eos_token_id).to(torch.int32).to(device)
    bos_cumsum = torch.cumsum(bos_mask, dim=1)
    eos_cumsum = torch.cumsum(eos_mask, dim=1)
    speaking_mask = (bos_cumsum > eos_cumsum).to(torch.float32)
    return speaking_mask.long()


def add_structured_noise_preserve_tail(
    mask: torch.Tensor,
    span_prob: float = 0.05,
    min_len: int = 2,
    max_len: int = 3,
    min_preserve: int = 4,
):
    """
    Adds structured noise to a binary mask by flipping random spans (2–3 tokens at a time),
    while preserving the last `min_preserve` tokens of each speaking region (1s).

    Args:
        mask (torch.Tensor): Binary mask of shape (B, T), values in {0, 1}
        span_prob (float): Probability of inserting a noisy span per token
        min_len (int): Minimum span length to flip
        max_len (int): Maximum span length to flip
        min_preserve (int): Number of 1s at the end of each span to protect from flipping

    Returns:
        torch.Tensor: Noised mask (same shape)
    """
    B, T = mask.shape
    noised_mask = mask.clone()

    for b in range(B):
        i = 0
        while i < T:
            if mask[b, i] == 1:
                # Start of a speaking region
                start = i
                while i < T and mask[b, i] == 1:
                    i += 1
                end = i  # exclusive
                span_len = end - start

                if span_len > min_preserve:
                    allowed_start = start
                    allowed_end = end - min_preserve
                    j = allowed_start
                    while j < allowed_end:
                        if random.random() < span_prob:
                            flip_len = random.randint(min_len, max_len)
                            flip_end = min(j + flip_len, allowed_end)
                            noised_mask[b, j:flip_end] = (noised_mask[b, j:flip_end] + 1) % 2
                            j = flip_end
                        else:
                            j += 1
            else:
                i += 1
    return noised_mask


import torch


class EfficientBatchStreamingSpeakingMaskGenerator:
    def __init__(
        self,
        max_length: int,
        batch_size: int,
        device,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        eou_window: int = 2,
        eos_lookback: int = 6,
        force_bos_from_eou: bool = False,  # NEW
        bos_lookback: int = 6,  # NEW
    ):
        self.max_length = max_length
        self.batch_size = batch_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.eou_window = eou_window
        self.eos_lookback = eos_lookback
        self.bos_lookback = bos_lookback
        self.force_bos_from_eou = force_bos_from_eou
        self.device = device

        self.bos_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.eos_counts = torch.zeros(batch_size, dtype=torch.int32, device=device)
        self.target_mask = torch.zeros(batch_size, max_length, dtype=torch.float32, device=device)
        self.eou_cache = torch.full((batch_size, max_length), float('nan'), device=device)

        self.eou_buffer = torch.zeros(batch_size, eou_window, dtype=torch.float32, device=device)
        self.eou_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)

        self.recent_is_eos = torch.zeros(batch_size, eos_lookback, dtype=torch.bool, device=device)
        self.recent_eos_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)

        self.recent_is_bos = torch.zeros(batch_size, bos_lookback, dtype=torch.bool, device=device)  # NEW
        self.recent_bos_ptr = torch.zeros(batch_size, dtype=torch.long, device=device)  # NEW

        self.t = 0  # timestep

    def step(self, tokens: torch.Tensor, eou_probs: torch.Tensor = None) -> torch.Tensor:
        B = self.batch_size
        device = self.device

        batch_indices = torch.arange(B, device=device)
        is_bos = (tokens == self.bos_token_id).view(-1)
        is_eos = (tokens == self.eos_token_id).view(-1)

        if eou_probs is not None:
            assert eou_probs.shape[0] == B, f"eou_probs must be shape [B], got {eou_probs.shape}"
            self.eou_cache[:, self.t] = eou_probs

        # Update recent BOS/EOS history
        self.recent_is_eos[batch_indices, self.recent_eos_ptr] = is_eos
        self.recent_eos_ptr = (self.recent_eos_ptr + 1) % self.eos_lookback

        self.recent_is_bos[batch_indices, self.recent_bos_ptr] = is_bos
        self.recent_bos_ptr = (self.recent_bos_ptr + 1) % self.bos_lookback

        self.bos_counts += is_bos.int()
        self.eos_counts += is_eos.int()

        forced_eos = torch.zeros(B, dtype=torch.bool, device=device)
        forced_bos = torch.zeros(B, dtype=torch.bool, device=device)

        if eou_probs is not None:
            # Update circular EOU buffer
            self.eou_buffer[:, self.eou_ptr[0]] = eou_probs
            self.eou_ptr = (self.eou_ptr + 1) % self.eou_window

            prev_idx = int((self.eou_ptr[0].item() - 2) % self.eou_window)
            last_idx = int((self.eou_ptr[0].item() - 1) % self.eou_window)

            prev_vals = self.eou_buffer[:, prev_idx]
            last_vals = self.eou_buffer[:, last_idx]

            just_ended = (prev_vals == 1.0) & (last_vals == 0.0)
            just_started = (prev_vals == 0.0) & (last_vals == 1.0)  # NEW

            no_recent_eos = ~self.recent_is_eos.any(dim=1)
            no_recent_bos = ~self.recent_is_bos.any(dim=1)  # NEW

            # Force EOS
            forced_eos = (~is_eos) & (self.bos_counts > self.eos_counts) & just_ended & no_recent_eos

            self.eos_counts += forced_eos.int()

            # Force BOS (optional)
            if self.force_bos_from_eou:
                forced_bos = (~is_bos) & (self.bos_counts <= self.eos_counts) & just_started & no_recent_bos
                self.bos_counts += forced_bos.int()

        # Compute speaking mask
        is_speaking = (self.bos_counts > self.eos_counts).float()
        is_speaking = torch.where(is_eos | forced_eos, torch.tensor(1.0, device=device), is_speaking)

        if self.t < self.max_length:
            self.target_mask[:, self.t] = is_speaking

        self.t += 1
        return self.target_mask[:, : self.t]

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.target_mask[:, : self.t], self.eou_cache[:, : self.t]


class DuplexS2SSpeechDecoderModel(LightningModule, HFHubMixin):
    def __init__(self, cfg: dict) -> None:
        assert isinstance(cfg, dict), (
            "You must pass the config to DuplexS2SModel as a Python dict to support hyperparameter serialization "
            f"in PTL checkpoints (we got: '{type(cfg)=}')."
        )
        super().__init__()
        self.save_hyperparameters()
        # convert dict to config
        cfg = DictConfig(cfg)
        self.cfg = cfg.model
        self.target_sample_rate = cfg.data.target_sample_rate
        self.source_sample_rate = cfg.data.source_sample_rate
        self.validation_save_path = os.path.join(cfg.exp_manager.explicit_log_dir, "validation_logs")

        # move back text channel by x, in inference it advance the text channel prediction by x frames
        self.advance_text_channel_by = self.cfg.get("advance_text_channel_by", None)

        # compute source fps
        self.source_fps = self.source_sample_rate / (
            self.source_sample_rate * cfg.data.frame_length
        )  # conver frame rate in fps

        setup_audio_codec(self)
        self._codebook_size = self.audio_codec.vector_quantizer.codebook_size_per_group
        self._num_codebooks = self.audio_codec.vector_quantizer.num_groups

        # to be able to load older model
        if self.cfg.get("custom_codebook_size", None):
            self._codebook_size = self.cfg.get("custom_codebook_size")

        # compute target fps
        self.target_fps = self.target_sample_rate / self.audio_codec.samples_per_frame
        # compute interpolation factor to interpolate
        self.interpolation_factor = self.target_fps / self.source_fps
        # x = torch.nn.functional.interpolate(x.unsqueeze(1), size=None, scale_factor=[1, self.interpolation_factor], mode='nearest-exact', align_corners=None, recompute_scale_factor=None, antialias=False)

        # We load the pretrained HF LLM using "ForCausalLM" variant so that we can obtain the
        # pretrained LM head weights.
        # However, for S2S we need to access the activations before LM head directly
        # to feed them to the audio codec head.
        self.tokenizer = AutoTokenizer(self.cfg.pretrained_llm, use_fast=True)
        if 'Qwen2.5' in self.cfg.pretrained_llm:
            # For Qwen, '<|im_start|>' is a common choice for a BOS token.
            # You can check your tokenizer's vocabulary for the best candidate.
            logging.warning("Tokenizer does not have a `bos_token`. Setting it to '<|im_start|>'.")
            self.tokenizer.bos_token = '<|im_start|>'
            self.tokenizer.eos_token = '<|im_end|>'
            if self.cfg.get("use_extra_id_for_pad", False):
                self.tokenizer.pad_token = '<|extra_1|>'

        llm = load_pretrained_hf(self.cfg.pretrained_llm, pretrained_weights=self.cfg.pretrained_weights).train()
        self.llm = llm.model  # fetch PretrainedBaseModel from model "ForCausalLM"
        self.lm_head = llm.lm_head
        # Note: we have to "move out" the token embedding outside of LLM to avoid
        #       messing up FSDP/TP hooks.
        self.embed_tokens = self.llm.embed_tokens
        del self.llm.embed_tokens
        maybe_install_lora(self)

        # Load the pretrained streaming ASR model and copy its parameters into the audio perception module.
        setup_speech_encoder(self)

        if self.cfg.get("use_eou_decoder", None):
            if self.cfg.get("eou_decoder_from_wav", None):
                self.eou_decoder = EOUDecoderFromWav(
                    samples_per_frame=int(self.source_sample_rate / self.source_fps),
                    audio_proj_size=1024,
                    output_dim=2,
                    n_layers=self.cfg.get("eou_decoder_num_layers", 3),
                    d_model=1024,
                    d_ffn=4096,
                    is_causal=True,
                    sliding_window_size=12,
                    max_position_embeddings=self.cfg.speech_decoder.max_length_causal_mask,
                )
            else:
                t_params = {
                    "n_layers": self.cfg.get("eou_decoder_num_layers", 3),  # 3 layers
                    "d_model": 768,
                    "d_ffn": 3072,
                    "sa_n_heads": 12,
                    "kernel_size": 1,
                    "p_dropout": 0.1,
                    "p_dropout_out": 0.0,
                    "has_xattn": False,
                    "is_causal": True,
                    "apply_norm_to_cond": True,
                    "apply_norm_out": True,
                    "max_length_causal_mask": self.cfg.speech_decoder.max_length_causal_mask,
                }
                self.eou_decoder = EOUDecoder(input_dim=self.cfg.get("asr_emb_dim", 512), params=t_params)
            self.eou_embedding = torch.nn.Embedding(2, self.llm.config.hidden_size)

        if self.cfg.get("inference_use_external_eou_predictor", None):
            self.eou_decoder = EOUDecoderFromWav(
                samples_per_frame=int(self.source_sample_rate / self.source_fps),
                audio_proj_size=1024,
                output_dim=2,
                n_layers=self.cfg.get("eou_decoder_num_layers", 3),
                d_model=1024,
                d_ffn=4096,
                is_causal=True,
                sliding_window_size=12,
                max_position_embeddings=self.cfg.speech_decoder.max_length_causal_mask,
            )
            self.eou_embedding = torch.nn.Embedding(2, self.llm.config.hidden_size)

        if self.cfg.get("llm_predict_eou", None):
            if self.cfg.get("llm_use_extra_eou_waveform_encoder", False):
                self.eou_wav_encoder = EOUDecoderFromWav(
                    samples_per_frame=int(self.source_sample_rate / self.source_fps),
                    audio_proj_size=1024,
                    output_dim=self.llm.config.hidden_size,
                    n_layers=self.cfg.get("llm_extra_eou_waveform_encoder_num_layers", 3),
                    d_model=1024,
                    d_ffn=4096,
                    is_causal=True,
                    sliding_window_size=12,
                    max_position_embeddings=self.cfg.speech_decoder.max_length_causal_mask,
                )
            self.eou_embedding = torch.nn.Embedding(2, self.llm.config.hidden_size)
            self.eou_projection = nn.Linear(self.llm.config.hidden_size, 2)

        llm_tokenizer_vocab_items = self.tokenizer.vocab
        # if vocab is a dict it already has the subword and token id, if not, get it from the tokenizer
        if isinstance(llm_tokenizer_vocab_items, dict):
            llm_tokenizer_vocab_items = llm_tokenizer_vocab_items.items()
        else:
            llm_tokenizer_vocab_items = [
                (subword, self.tokenizer.tokenizer._tokenizer.token_to_id(subword))
                for subword in llm_tokenizer_vocab_items
            ]

        self.speech_generation = TransformerARSpeechDecoder(
            speech_decoder_parms=OmegaConf.to_container(self.cfg.speech_decoder),
            lantent_dim=self.llm.config.hidden_size,
            num_audio_codebooks=self._num_codebooks,
            num_audio_tokens_per_codebook=self.speech_vocab_size,
            llm_tokenizer_vocab_items=llm_tokenizer_vocab_items,
        )

        if self.cfg.get("pretrained_s2s_model", None):
            self.init_from_model_from_ckpt(self.cfg.pretrained_s2s_model)

        # load pretrained TTS model
        if self.cfg.get("pretrained_tts", None):
            self.init_speech_generation_from_tts_checkpoint(self.cfg.pretrained_tts)

        # load speech decoder/speech generation module from another checkpoint
        if self.cfg.get("pretrained_tts_from_s2s", None):
            self.init_speech_generation_from_another_s2s_checkpoint(self.cfg.pretrained_tts_from_s2s)

        # restore EOU predictor from another checkpoint
        if self.cfg.get("pretrained_eou_from_s2s", None):
            self.init_eou_from_another_s2s_checkpoint(self.cfg.pretrained_eou_from_s2s)

        self.embed_audio_tokens = torch.nn.ModuleList(
            [
                torch.nn.Embedding(self.speech_vocab_size, self.embed_tokens.embedding_dim)
                for _ in range(self._num_codebooks)
            ]
        )
        self.audio_head = torch.nn.Linear(self.llm.config.hidden_size, self.speech_vocab_size * self._num_codebooks)

        # cached for quicker audio decoding
        self.register_buffer(
            "_control_codes",
            torch.tensor([self.speech_bos_id, self.speech_eos_id, self.speech_delay_id], device=self.device),
        )
        self._use_fsdp = False
        self._use_tp = False

    def init_speech_generation_from_tts_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.speech_generation.state_dict())
            self.speech_generation.load_state_dict(checkpoint_state, strict=True)

    def init_speech_generation_from_another_s2s_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            # filter keys to keep only speech generation keys and also
            checkpoint_state = {
                k.replace("model.speech_decoder.", "").replace("speech_generation.", ""): v
                for k, v in checkpoint_state.items()
                if "model.speech_decoder." in k or "speech_generation." in k
            }
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.speech_generation.state_dict())
            self.speech_generation.load_state_dict(checkpoint_state, strict=True)

    def init_eou_from_another_s2s_checkpoint(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            # filter keys to keep only speech generation keys and also
            checkpoint_state = {
                k.replace("eou_decoder.", ""): v for k, v in checkpoint_state.items() if "eou_decoder." in k
            }
            if self.cfg.get("use_eou_decoder", None) or self.cfg.get("inference_use_external_eou_predictor", None):
                checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.eou_decoder.state_dict())
                self.eou_decoder.load_state_dict(checkpoint_state, strict=True)

    def init_from_model_from_ckpt(self, checkpoint_path):
        if checkpoint_path is not None:
            if '.nemo' in checkpoint_path:
                with tempfile.TemporaryDirectory() as tmpdir:
                    NLPSaveRestoreConnector._unpack_nemo_file(checkpoint_path, tmpdir)
                    checkpoint_path = f"{tmpdir}/model_weights.ckpt"
                    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
            else:
                checkpoint_state = torch.load(checkpoint_path, weights_only=False, map_location='cpu')['state_dict']

            # partial initialization support
            checkpoint_state = set_model_dict_for_partial_init(checkpoint_state, self.state_dict())
            self.load_state_dict(checkpoint_state, strict=True)

    @property
    def speech_vocab_size(self):
        """Return the size of the audio codec codebook including extra speech BOS and EOS tokens."""
        return self._codebook_size + 3

    @property
    def speech_bos_id(self) -> int:
        """Indicates start of utterance generation (not start of inference!)."""
        if self.cfg.get("custom_speech_bos_id", None):
            return self.cfg.get("custom_speech_bos_id")
        return self._codebook_size

    @property
    def speech_eos_id(self) -> int:
        """Indicates end of utterance generation."""
        if self.cfg.get("custom_speech_eos_id", None):
            return self.cfg.get("custom_speech_eos_id")
        return self._codebook_size + 1

    @property
    def speech_delay_id(self) -> int:
        """Indicates start of inference (the very first frame)."""
        if self.cfg.get("custom_speech_delay_id", None):
            return self.cfg.get("custom_speech_delay_id")
        return self._codebook_size + 2

    @property
    def text_vocab_size(self):
        """Return the size of the text tokenizer."""
        return self.tokenizer.vocab_size

    @property
    def text_bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def text_eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def text_pad_id(self) -> int:
        """
        Text pad ID is used as a 'blank' for frames when the model is not speaking
        and for frames where the model is speaking but has already predicted the
        entire text channel's content.

        Example:

            flow:         |---user---||-------assistant--------||-user-|
            text channel:  0000000000  1xxxxxxx0000000000000002  000000

        Where 0 indicates PAD ID, 1 indicates BOS ID, 2 indacates EOS ID,
        and x indicates tokens corresponding to actual text

        """
        return get_pad_id(self.tokenizer)

    def forward(
        self,
        input_embeds: Tensor,
        cache=None,
        input_audio_tokens=None,
        seq_mask=None,
        target_text_tokens=None,
        modality_adapter_emb=None,
        asr_emb=None,
        speaker_encoder_emb=None,
        eou=None,
    ) -> dict[str, Tensor]:
        """
        Separated text and speech prediction:
            - Speech prediction is achieved by a independent AR decoder based on last_hidden_state + audio tokens
            - For KV-cache:
                (1) llm cache depends on input cache is None or Not
                (2) speech_generation cache relys on reset_input_and_kv_cache function.
        """
        out = self.llm(
            inputs_embeds=input_embeds, past_key_values=cache, use_cache=cache is not None, return_dict=True
        )
        B, T = input_embeds.shape[:2]
        text_logits = self.lm_head(out['last_hidden_state'])  # (B, T, text_vocab_size)

        if seq_mask is not None:
            # This is training Mode
            seq_mask = seq_mask[:, :, -1].reshape(seq_mask.size(0), seq_mask.size(1))
            # disable cache in training mode
            if self.speech_generation.use_input_cache:
                self.speech_generation.reset_input_and_kv_cache(use_cache=False)

        # if inference time, uses the target text tokens sampled from the llm backbone
        if self.speech_generation.use_input_cache and not self.training:
            if self.cfg.get("inference_pad_boost", None):
                text_logits[:, :, self.text_pad_id] += self.cfg.inference_pad_boost
            if self.cfg.get("inference_bos_boost", None):
                text_logits[:, :, self.text_bos_id] += self.cfg.inference_bos_boost
            if self.cfg.get("inference_eos_boost", None):
                text_logits[:, :, self.text_eos_id] += self.cfg.inference_eos_boost

            target_text_tokens = torch.argmax(text_logits, dim=-1).view(B, T).contiguous()

            if self.cfg.get('inference_force_bos_eos_follow_eou', None) and eou is not None:
                not_special = (target_text_tokens[:, -1] != self.text_bos_id) & (
                    target_text_tokens[:, -1] != self.text_eos_id
                )
                should_pad = (eou == 0).squeeze(-1) & not_special
                # if EOU is zero, allows text channel only to assume, zero, eou or bos
                target_text_tokens[:, -1] = torch.where(should_pad, self.text_pad_id, target_text_tokens[:, -1])

            if self.cfg.get('convert_pad_to_extra_id_on_speech_decoder', None):
                target_text_tokens[target_text_tokens == self.text_pad_id] = self.tokenizer.tokenizer._tokenizer.token_to_id("<|endoftext|>") # <|endoftext|> token id
        else:
            # Drop BOS tokens with per-token probability (augmentation)
            drop_bos_prob = getattr(self.cfg, "drop_text_bos_prob", 0.0)
            if drop_bos_prob > 0.0:
                bos_mask = (target_text_tokens == self.text_bos_id)
                # Generate random mask only for BOS positions
                drop_bos_mask = torch.rand_like(target_text_tokens, dtype=torch.float) < drop_bos_prob
                target_text_tokens = torch.where(bos_mask & drop_bos_mask, self.text_pad_id, target_text_tokens)

            # Drop EOS tokens with per-token probability (augmentation)
            drop_eos_prob = getattr(self.cfg, "drop_text_eos_prob", 0.0)
            if drop_eos_prob > 0.0:
                eos_mask = (target_text_tokens == self.text_eos_id)
                drop_eos_mask = torch.rand_like(target_text_tokens, dtype=torch.float) < drop_eos_prob
                target_text_tokens = torch.where(eos_mask & drop_eos_mask, self.text_pad_id, target_text_tokens)

        audio_logits, _ = self.speech_generation(
            out['last_hidden_state'].transpose(0, 1),
            seq_mask,
            input_audio_tokens=input_audio_tokens,
            target_text_tokens=target_text_tokens,
            modality_adapter_emb=modality_adapter_emb,
            asr_emb=asr_emb,
            speaker_encoder_emb=speaker_encoder_emb,
        )

        audio_logits = audio_logits.view(B, T, self._num_codebooks, self.speech_vocab_size)

        ans = {
            "text_logits": text_logits,
            "audio_logits": audio_logits,
        }
        if cache is not None:
            ans["cache"] = out["past_key_values"]

        if self.cfg.get("llm_predict_eou", None):
            eou_logits = self.eou_projection(out['last_hidden_state'])
            ans["eou_logits"] = eou_logits
        return ans

    def add_noise_to_batch(
        self,
        batch_audio,
        noise_folder,
        snr_db=20,
        noise_prob_scale_user=0.3,
        noise_prob_scale_user_min_snr=-15,
        noise_prob_scale_user_max_snr=24,
        snr_measure_dur=0.0,
        noise_resample=True,
        noise_prob_low_pass=0.1,
    ):

        batch_size, audio_length = batch_audio.shape

        import glob

        import librosa
        import soundfile as sf
        from scipy.signal import butter, lfilter

        noise_files = [f for f in glob.glob(noise_folder + "/*.wav")]
        if not noise_files:
            raise ValueError(f"No noise files found in {noise_folder}")

        for i in range(batch_size):

            def get_scale_factor(signal, noise, snr_db):
                if snr_measure_dur > 0:
                    signal = signal[: (snr_measure_dur * self.source_sample_rate)]
                    noise = noise[: (snr_measure_dur * self.source_sample_rate)]
                signal_power = torch.mean(signal**2) + 1e-8
                noise_power = torch.mean(noise**2) + 1e-8

                target_noise_power = signal_power / (10 ** (snr_db / 10))
                scaling_factor = torch.sqrt(target_noise_power / noise_power)
                return scaling_factor

            if random.random() < noise_prob_scale_user:
                scaling_factor = get_scale_factor(
                    batch_audio[i],
                    batch_audio[i],
                    random.randint(noise_prob_scale_user_min_snr, noise_prob_scale_user_max_snr),
                )
                batch_audio[i] = batch_audio[i] * scaling_factor

            def get_noise(noise_files):

                noise_path = random.choice(noise_files)
                noise, sr = sf.read(noise_path, dtype='float32')

                # resample noise from sr to self.cfg.data.train_ds.sample_rate
                if noise_resample and sr != self.source_sample_rate:
                    noise = librosa.resample(noise, orig_sr=sr, target_sr=self.source_sample_rate)

                if len(noise.shape) > 1:
                    noise = np.mean(noise, axis=1)

                noise_tensor = torch.tensor(noise, dtype=batch_audio.dtype, device=batch_audio.device)
                scaling_factor = get_scale_factor(batch_audio[i], noise_tensor, snr_db)
                noise_tensor = noise_tensor * scaling_factor
                return noise_tensor

            noise = get_noise(noise_files)
            noise2 = get_noise(noise_files)
            noise3 = get_noise(noise_files)
            noise = torch.cat([noise, noise2, noise3], axis=0)

            if noise.size(0) < audio_length:
                repeat_times = (audio_length // noise.size(0)) + 1
                # For a 1D tensor, we want to repeat its elements.
                # If noise has other dimensions, adjust the repeat_times_tuple accordingly.
                # e.g., if noise is (C, L), and we want to repeat along L,
                # repeat_times_tuple = (1, repeat_times)
                noise = noise.repeat(repeat_times)[:audio_length]
            else:
                # If noise is a PyTorch tensor
                start_idx = torch.randint(0, noise.size(0) - audio_length + 1, (1,)).item()
                # Or if noise was originally a list/numpy array and you want to keep Python's random
                # start_idx = random.randint(0, len(noise) - audio_length)
                noise = noise[start_idx : start_idx + audio_length]

            # Function to create a low-pass filter
            def butter_lowpass(cutoff, fs, order=5):
                nyquist = 0.5 * fs
                normal_cutoff = cutoff / nyquist
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                return b, a

            # Function to apply the low-pass filter to data (tmp impl on cpu)
            def lowpass_filter(data, cutoff, fs, order=5):
                b, a = butter_lowpass(cutoff, fs, order=order)
                b = torch.tensor(b, dtype=torch.float32).cuda()
                a = torch.tensor(a, dtype=torch.float32).cuda()
                # Apply the filter using lfilter function from scipy..numpysig.numpynal (CPU)
                y_cpu = lfilter(b.cpu().numpy(), a.cpu().numpy(), data.cpu().numpy())
                # Convert the filtered data back to torch tensor and move to GPU.numpy
                y_gpu = torch.tensor(y_cpu, dtype=torch.float32).cuda()
                return y_gpu

            if random.random() < noise_prob_low_pass:
                # Define the desired cutoff frequency (in Hz)
                cutoff = 1000.0
                # Apply low-pass filter to the WAV data
                noise = lowpass_filter(noise, cutoff, self.source_sample_rate)

            batch_audio[i] = batch_audio[i] + noise

        return batch_audio

    def prepare_inputs(self, batch: dict):
        """
        Similar to DuplexS2SModel.prepare_inputs, with following changes:
            (1) Add 'input_audio_tokens' and 'seq_mask' in return value for TransformerARSpeechDecoder
            (2) Remove audio codec embedding from 'input_embeds'
        """
        # check if audios has the same batch size
        assert batch["source_audio"].size(0) == batch["target_audio"].size(0)
        assert batch["target_first_turn_audio"].size(0) == batch["target_audio"].size(0)

        if self.cfg.get('use_old_noise_aug', None):
            # ToDo we are applying it in all datasets, old codebase does not applied in real conv data
            noise_prob = 0.99
            noise_min_snr = 20
            noise_max_snr = 50
            noise_path = self.cfg.get(
                'old_noise_aug_path',
                "/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/duplex/dns5/dns5_demand_noise/",
            )
            noise_path_name = "*"
            no_noise_audio = batch["source_audio"].clone()
            if (
                self.training
                and batch["formatter"][0] != 's2s_duplex_overlap_as_s2s_duplex'
                and noise_prob
                and random.random() < noise_prob
            ):
                batch["source_audio"] = self.add_noise_to_batch(
                    batch["source_audio"],
                    os.path.join(noise_path, noise_path_name),
                    snr_db=random.randint(noise_min_snr, noise_max_snr),
                    noise_prob_scale_user=0.3,
                    noise_prob_scale_user_min_snr=-15,
                    noise_prob_scale_user_max_snr=24,
                    snr_measure_dur=0.0,
                    noise_resample=True,
                    noise_prob_low_pass=0.1,
                )
        else:
            # change audio volume randomly
            if self.training and random.random() < self.cfg.get('noise_prob_scale_user', 0.0):
                # prev codebase had 0.0631 and 5.6234 here we round the values
                min_scale_val = self.cfg.get('noise_scale_user_min', 0.0631)  # -15 snr
                max_scale_val = self.cfg.get('noise_scale_user_min', 5.6234)  # 24 snr

                # get a random float value between min and max
                scaling_factor = (
                    torch.rand(batch["source_audio"].size(0), device=batch["source_audio"].device)
                    * (max_scale_val - min_scale_val)
                    + min_scale_val
                )
                batch["source_audio"] = batch["source_audio"] * scaling_factor.unsqueeze(-1)

            # apply low pass filter
            if self.training and random.random() < self.cfg.get('noise_prob_low_pass', 0.0):
                # prev codebase had 0.0631 and 5.6234 here we round the values
                cutoff_freq = self.cfg.get('noise_low_pass_cutoff_freq', 1000.0)
                # note here we are using a biquad filter, older codebase we are using a filter of order 5
                batch["source_audio"] = torchaudio.functional.lowpass_biquad(
                    waveform=batch["source_audio"], sample_rate=self.source_sample_rate, cutoff_freq=cutoff_freq
                )

        source_encoded, source_encoded_lens, asr_emb = self.perception(
            input_signal=batch["source_audio"],
            input_signal_length=batch["source_audio_lens"],
            return_encoder_emb=True,
        )

        # if inference return speaker embedding None and it will uses the cached speaker embedding
        if not self.training:
            speaker_encoder_emb = None
        else:  # if training or eval extract embedding from first agent turn returned by the dataloader
            if self.speech_generation.use_speaker_encoder:
                target_first_turn_audio = batch["target_first_turn_audio"]
                target_first_turn_audio_lens = batch["target_first_turn_audio_lens"]
                speaker_encoder_emb = self.speech_generation.get_speaker_embedding(
                    target_first_turn_audio, target_first_turn_audio_lens, self.target_sample_rate
                )
            else:
                speaker_encoder_emb = None

        target_tokens = batch["target_tokens"]
        if (diff := target_tokens.shape[1] - source_encoded.shape[1]) < 0:
            target_tokens = torch.cat(
                [
                    target_tokens,
                    (
                        torch.ones(source_encoded.shape[0], abs(diff), device=source_encoded.device) * self.text_pad_id
                    ).to(torch.long),
                ],
                dim=-1,
            )
        elif diff > 0:
            target_tokens = target_tokens[:, : source_encoded.shape[1]]

        with fp32_precision(), torch.no_grad():
            target_codes, target_codes_lens = self.audio_codec.encode(
                audio=batch["target_audio"], audio_len=batch["target_audio_lens"]
            )
        target_codes = target_codes.transpose(1, 2)  # (B, K, T) -> (B, T, K)

        if (tl := target_codes.shape[1]) != (sl := source_encoded.shape[1]):
            if tl < sl:
                diff = sl - tl
                source_encoded = source_encoded[:, :tl]
                asr_emb = asr_emb[:, :tl]
                target_tokens = target_tokens[:, :tl]
                torch.clamp_(source_encoded_lens, max=tl)
            else:
                diff = tl - sl
                target_codes = target_codes[:, :sl]
                torch.clamp_(target_codes_lens, max=sl)
            if diff > 2:
                logging.warning(
                    f"A mismatch between source ({sl}) and target ({tl}) sequence length greater than 2 detected. "
                    f"This may indicate significant desynchronization in longer sessions."
                )

        btt = target_tokens[..., None]
        target_codes = torch.where(btt == self.text_bos_id, self.speech_bos_id, target_codes)
        target_codes = torch.where(btt == self.text_eos_id, self.speech_eos_id, target_codes)

        # ToDo: implement in a way that we can set the number of speech delay > 1
        target_codes = torch.cat(
            [
                torch.full(
                    [target_codes.shape[0], 1, target_codes.shape[-1]],
                    fill_value=self.speech_delay_id,
                    device=self.device,
                    dtype=torch.long,
                ),
                target_codes[:, :-1],
            ],
            dim=1,
        )
        # move back text channel by x, in inference it advance the text channel prediction
        # it is the oposite of speech delay applied on text channel
        if self.advance_text_channel_by:
            pad = torch.full(
                (target_tokens.shape[0], self.advance_text_channel_by),
                fill_value=self.text_pad_id,
                device=target_tokens.device,
                dtype=torch.long,
            )
            target_tokens = torch.cat([target_tokens[:, self.advance_text_channel_by :], pad], dim=-1)
            # make sure that eos/bos is in the place (it can cut tokens from the first advance_text_channel_by tokens and this will breaks everything)

        if self.cfg.get("delay_text_eos_by", None):
            target_tokens = delay_eos(target_tokens, self.text_eos_id, self.text_pad_id, shift=self.cfg.delay_text_eos_by)

        input_ids = torch.cat([target_codes, target_tokens[..., None]], dim=-1)
        if self._use_tp:
            tp_world_size = self.device_mesh["tensor_parallel"].size()
            if (remainder := (input_ids.shape[1] - 1) % tp_world_size) != 0:
                input_ids = input_ids[:, :-remainder]
                source_encoded = source_encoded[:, :-remainder]
                asr_emb = asr_emb[:, :-remainder]

        text_inputs = input_ids[:, :-1, -1]  # (B, T-1)
        text_labels = input_ids[:, 1:, -1]  # (B, T-1)
        audio_inputs = input_ids[:, :-1, :-1]  # (B, T-1, K)
        audio_labels = input_ids[:, 1:, :-1]  # (B, T-1, K)

        input_embeds = self.embed_tokens(text_inputs)

        input_embeds.add_(source_encoded[:, :-1] * self.cfg.get("duplex_user_channel_weight", 1.0))

        # create sequence mask
        seq_mask = torch.ones_like(
            torch.cat([text_labels.unsqueeze(-1), audio_labels], dim=-1),
            device=self.device,
            dtype=torch.bool,
        )

        if self.cfg.get("mask_sequence_loss", True):
            # set the mask based on the target_token_lens to disconsider sequence padding in loss
            for i in range(batch["target_token_lens"].size(0)):
                speech_end_idx = batch["target_token_lens"][i]
                seq_mask[i, speech_end_idx:, :] = 0

            # check new mask consistency
            mask_lengths = seq_mask[:, :, 0].sum(-1)
            assert torch.allclose(batch["target_token_lens"].float(), mask_lengths.float(), atol=2.0)

        eou_logits = None
        eou_labels = None
        eou_loss_scale = None
        eou_skip_batch = False
        # compute eou labels and logits. Note we are ignoring silence augmented batches because this can break the EOU predictor
        if self.cfg.get("use_eou_decoder", None):
            # create eou labels
            eou_labels = generate_multiturn_speaking_mask(
                text_labels, bos_token_id=self.text_bos_id, eos_token_id=self.text_eos_id
            ).detach()
            # predict eou logits if it is not a silence augmented batch
            if self.cfg.get("eou_decoder_ignore_sil_batch", False) and "silence_augmented" in batch["formatter"][0]:
                eou_skip_batch = True
            else:
                if self.cfg.get("eou_decoder_from_wav", None):
                    eou_logits, _ = self.eou_decoder(
                        batch["source_audio"].to(source_encoded.dtype), batch["source_audio_lens"]
                    )
                else:
                    eou_logits = self.eou_decoder(
                        asr_emb[:, :-1], seq_mask[:, :, -1].reshape(seq_mask.size(0), seq_mask.size(1))
                    )

                # ensures that logits and labels has the same shape
                if eou_labels.size(1) > eou_logits.size(1):
                    # Pad on the right (end of time axis)
                    pad_len = eou_labels.size(1) - eou_logits.size(1)
                    eou_logits = torch.nn.functional.pad(
                        eou_logits, pad=(0, 0, 0, pad_len), mode='constant', value=0.0
                    )
                else:
                    eou_logits = eou_logits[:, : eou_labels.size(1)]
            if self.cfg.get("eou_structured_noise_aug_enabled", None):
                eou_labels_aug = add_structured_noise_preserve_tail(
                    eou_labels, span_prob=0.05, min_len=2, max_len=3, min_preserve=4
                )
                # add eou embedding to the llm input
                eou_emb = self.eou_embedding(eou_labels_aug)
            else:
                # add eou embedding to the llm input
                eou_emb = self.eou_embedding(eou_labels)

            input_embeds.add_(eou_emb)
        eou_loss_scale = seq_mask[:, :, 0].clone().float()

        if self.cfg.get("llm_predict_eou", None):
            # create eou labels
            eou_labels = generate_multiturn_speaking_mask(
                text_labels, bos_token_id=self.text_bos_id, eos_token_id=self.text_eos_id
            ).detach()
            # input shifted by one
            eou_input = F.pad(eou_labels[:, :-1], (1, 0), value=0.0).clone()
            # add extra delay on eou labels and input to make the eou be predicted before bos/eos
            if self.cfg.get("llm_eou_bos_eos_delay", 0):
                eou_labels = F.pad(
                    eou_labels[:, : -self.cfg.llm_eou_bos_eos_delay], (self.cfg.llm_eou_bos_eos_delay, 0), value=0.0
                )
                eou_input = F.pad(
                    eou_input[:, : -self.cfg.llm_eou_bos_eos_delay], (self.cfg.llm_eou_bos_eos_delay, 0), value=0.0
                )

            eou_emb = self.eou_embedding(eou_input)
            if self.cfg.get("llm_use_extra_eou_waveform_encoder", False):
                wav_eou_emb, _ = self.eou_wav_encoder(
                    batch["source_audio"].to(source_encoded.dtype), batch["source_audio_lens"]
                )
                # ensures that logits and labels has the same shape
                if eou_emb.size(1) > wav_eou_emb.size(1):
                    # Pad on the right (end of time axis)
                    pad_len = eou_emb.size(1) - wav_eou_emb.size(1)
                    wav_eou_emb = torch.nn.functional.pad(
                        wav_eou_emb, pad=(0, 0, 0, pad_len), mode='constant', value=0.0
                    )
                else:
                    wav_eou_emb = wav_eou_emb[:, : eou_emb.size(1)]
                # add eou input emb with the wav encoder embedding
                eou_emb = eou_emb + wav_eou_emb

            input_embeds.add_(eou_emb)
            eou_loss_scale = seq_mask[:, :, 0].clone().float()
            if self.cfg.get("eou_ignore_sil_batch", False) and "silence_augmented" in batch["formatter"][0]:
                eou_skip_batch = True

        # create loss scale mask by copying seq_mask to include mask sequence
        loss_scale = seq_mask.clone().float()

        if self.cfg.get("scale_loss_by", None):
            if self.cfg.scale_loss_by == 'non_sil_t':
                loss_scale[:, :, :1] = torch.where(
                    text_labels.unsqueeze(-1) != self.text_pad_id,
                    self.cfg.get("scale_loss_mask", self.cfg.get("nonsil_weight", 4.0)),
                    loss_scale[:, :, :1],
                )
            elif self.cfg.scale_loss_by == 'custom_nonsil_bos_eos':
                loss_scale[:, :, :1] = torch.where(
                    text_labels.unsqueeze(-1) != self.text_pad_id,
                    self.cfg.get("nonsil_weight", 1.0),
                    loss_scale[:, :, :1],
                )
                loss_scale[:, :, :1] = torch.where(
                    text_labels.unsqueeze(-1) == self.text_bos_id,
                    self.cfg.get("bos_weight", 10.0),
                    loss_scale[:, :, :1],
                )
                loss_scale[:, :, :1] = torch.where(
                    text_labels.unsqueeze(-1) == self.text_eos_id,
                    self.cfg.get("eos_weight", 10.0),
                    loss_scale[:, :, :1],
                )
            elif self.cfg.scale_loss_by == 'dynamic_text_non_sil_4x_and_bos_eos':
                # Set text loss weights
                for i in range(text_labels.size(0)):
                    current_scale = loss_scale[i, :, :1]
                    labels = text_labels.unsqueeze(-1)
                    num_real_padding_tokens = (torch.numel(current_scale) - current_scale.sum()).item()
                    silence_idxs = labels[i, :, :1] == self.text_pad_id

                    # compute dynamic silence/nonsilence factor
                    silence_idxs = silence_idxs * current_scale.bool()
                    num_silence_tokens = silence_idxs.sum().item()
                    num_non_silence = torch.numel(silence_idxs) - num_real_padding_tokens
                    factor = num_silence_tokens / num_non_silence

                    # make silence text tokens 2 x times less relevant in the loss than the silence tokens
                    new_weight = factor / 2
                    loss_scale[i, :, :1] = torch.where(silence_idxs, new_weight, loss_scale[i, :, :1])

                # set eos/bos 6x more important than a speech tokens and 12x more than a silence, this is that high because we will have only one bos/eos per turn and if it is nor right predicted the model will not produce text/speech
                loss_scale[:, :, :1] = torch.where(labels == self.text_bos_id, 6.0, loss_scale[:, :, :1])
                loss_scale[:, :, :1] = torch.where(labels == self.text_eos_id, 6.0, loss_scale[:, :, :1])
            elif self.cfg.scale_loss_by == 'non_sil_4_eos_bos_12':
                loss_scale[:, :, :1] = torch.where(
                    text_labels.unsqueeze(-1) != self.text_pad_id, 4.0, loss_scale[:, :, :1]
                )
                # set eos/bos 3x more important than a speech tokens and 12x more than a silence, this is that high because we will have only one bos/eos per turn and if it is nor right predicted the model will not produce text/speech
                loss_scale[:, :, :1] = torch.where(
                    text_labels.unsqueeze(-1) == self.text_bos_id, 12.0, loss_scale[:, :, :1]
                )
                loss_scale[:, :, :1] = torch.where(
                    text_labels.unsqueeze(-1) == self.text_eos_id, 12.0, loss_scale[:, :, :1]
                )
            elif self.cfg.scale_loss_by == 'non_sil_4_dynamic_eos_bos':
                # Expand text_labels to match the shape of loss_scale: [B, T] → [B, T, 1]
                text_labels_exp = text_labels.unsqueeze(-1)

                # Assign a weight of 4.0 to all non-padding tokens in the loss scale
                # Padding tokens retain their existing value
                loss_scale[:, :, :1] = torch.where(
                    text_labels_exp != self.text_pad_id,  # Condition: not a padding token
                    4.0,  # Assign fixed weight
                    loss_scale[:, :, :1],  # Keep original value otherwise
                )

                # Compute the total loss weight assigned to valid (non-padding) tokens in each sequence
                # Shape: [B] — one scalar value per batch item
                tot_scale_for_valid_tokens = loss_scale[:, :, :1].flatten(1, 2).sum(-1)

                # Count how many BOS tokens are present per sequence
                # Shape: [B]
                num_bos_tokens = (text_labels_exp == self.text_bos_id).flatten(1, 2).sum(-1)

                # Count how many EOS tokens are present per sequence
                # Shape: [B]
                num_eos_tokens = (text_labels_exp == self.text_eos_id).flatten(1, 2).sum(-1)

                # Compute the total number of special tokens (BOS + EOS) for each sequence
                # Shape: [B]
                tot_special_tokens = num_bos_tokens + num_eos_tokens

                # Loop through each item in the batch to reassign loss weight to special tokens
                for i in range(text_labels.size(0)):
                    # Avoid division by zero: only compute new weight if BOS/EOS tokens are present
                    if tot_special_tokens[i] > 0:
                        # Redistribute the total valid token weight equally across BOS and EOS tokens
                        new_weight = tot_scale_for_valid_tokens[i] / tot_special_tokens[i]
                    else:
                        # No special tokens found — set weight to zero
                        new_weight = 0.0

                    # Assign new_weight to BOS tokens in the current sequence
                    loss_scale[i, :, :1] = torch.where(
                        text_labels_exp[i, :, :1] == self.text_bos_id, new_weight, loss_scale[i, :, :1]
                    )

                    # Assign new_weight to EOS tokens in the current sequence
                    loss_scale[i, :, :1] = torch.where(
                        text_labels_exp[i, :, :1] == self.text_eos_id, new_weight, loss_scale[i, :, :1]
                    )

            elif self.cfg.scale_loss_by == 'non_sil_4_dynamic_x_less_eos_bos_speech_text':
                # Expand text_labels to match the shape of loss_scale: [B, T] → [B, T, 1]
                text_labels_exp = text_labels.unsqueeze(-1)

                # Assign a weight of 4.0 to all non-padding tokens in the loss scale
                # Padding tokens retain their existing value
                loss_scale[:, :, :1] = torch.where(
                    text_labels_exp != self.text_pad_id,  # Condition: not a padding token
                    4.0,  # Assign fixed weight
                    loss_scale[:, :, :1],  # Keep original value otherwise
                )

                # Compute the total loss weight assigned to valid (non-padding) tokens in each sequence
                # Shape: [B] — one scalar value per batch item
                text_tot_scale_for_valid_tokens = loss_scale[:, :, :1].flatten(1, 2).sum(-1)
                speech_tot_scale_for_valid_tokens = (
                    loss_scale[:, :, -1:].flatten(1, 2).sum(-1)
                )  # use only the last speech channel because all the channels are identical

                # Count how many BOS tokens are present per sequence
                # Shape: [B]
                num_bos_tokens = (text_labels_exp == self.text_bos_id).flatten(1, 2).sum(-1)

                # Count how many EOS tokens are present per sequence
                # Shape: [B]
                num_eos_tokens = (text_labels_exp == self.text_eos_id).flatten(1, 2).sum(-1)

                # Compute the total number of special tokens (BOS + EOS) for each sequence
                # Shape: [B]
                tot_special_tokens = num_bos_tokens + num_eos_tokens
                # Loop through each item in the batch to reassign loss weight to special tokens
                for i in range(text_labels.size(0)):
                    # Avoid division by zero: only compute new weight if BOS/EOS tokens are present
                    if tot_special_tokens[i] > 0:
                        # Redistribute the total valid token weight equally across BOS and EOS tokens
                        new_weight_text = (text_tot_scale_for_valid_tokens[i] / tot_special_tokens[i]) / self.cfg.get(
                            "dynamic_scale_loss_x", 10.0
                        )
                        new_weight_speech = (
                            speech_tot_scale_for_valid_tokens[i] / tot_special_tokens[i]
                        ) / self.cfg.get("dynamic_scale_loss_x", 10.0)
                    else:
                        # No special tokens found — set weight to zero
                        new_weight_text = 0.0
                        new_weight_speech = 0.0

                    # set text eos/bos scale
                    # Assign new_weight to BOS tokens in the current sequence
                    loss_scale[i, :, :1] = torch.where(
                        text_labels_exp[i, :, :1] == self.text_bos_id, new_weight_text, loss_scale[i, :, :1]
                    )

                    # Assign new_weight to EOS tokens in the current sequence
                    loss_scale[i, :, :1] = torch.where(
                        text_labels_exp[i, :, :1] == self.text_eos_id, new_weight_text, loss_scale[i, :, :1]
                    )
                    # set speech bos/eos scale
                    # Assign new_weight to BOS tokens in the current sequence
                    loss_scale[i, :, 1:] = torch.where(
                        audio_labels[i, :, :] == self.speech_bos_id, new_weight_speech, loss_scale[i, :, 1:]
                    )

                    # Assign new_weight to EOS tokens in the current sequence
                    loss_scale[i, :, 1:] = torch.where(
                        audio_labels[i, :, :] == self.speech_eos_id, new_weight_speech, loss_scale[i, :, 1:]
                    )

            elif self.cfg.scale_loss_by == 'non_sil_4_dynamic_10x_less_eos_bos_speech_text':
                # Expand text_labels to match the shape of loss_scale: [B, T] → [B, T, 1]
                text_labels_exp = text_labels.unsqueeze(-1)

                # Assign a weight of 4.0 to all non-padding tokens in the loss scale
                # Padding tokens retain their existing value
                loss_scale[:, :, :1] = torch.where(
                    text_labels_exp != self.text_pad_id,  # Condition: not a padding token
                    4.0,  # Assign fixed weight
                    loss_scale[:, :, :1],  # Keep original value otherwise
                )

                # Compute the total loss weight assigned to valid (non-padding) tokens in each sequence
                # Shape: [B] — one scalar value per batch item
                text_tot_scale_for_valid_tokens = loss_scale[:, :, :1].flatten(1, 2).sum(-1)
                speech_tot_scale_for_valid_tokens = (
                    loss_scale[:, :, -1:].flatten(1, 2).sum(-1)
                )  # use only the last speech channel because all the channels are identical

                # Count how many BOS tokens are present per sequence
                # Shape: [B]
                num_bos_tokens = (text_labels_exp == self.text_bos_id).flatten(1, 2).sum(-1)

                # Count how many EOS tokens are present per sequence
                # Shape: [B]
                num_eos_tokens = (text_labels_exp == self.text_eos_id).flatten(1, 2).sum(-1)

                # Compute the total number of special tokens (BOS + EOS) for each sequence
                # Shape: [B]
                tot_special_tokens = num_bos_tokens + num_eos_tokens
                # Loop through each item in the batch to reassign loss weight to special tokens
                for i in range(text_labels.size(0)):
                    # Avoid division by zero: only compute new weight if BOS/EOS tokens are present
                    if tot_special_tokens[i] > 0:
                        # Redistribute the total valid token weight equally across BOS and EOS tokens
                        new_weight_text = (text_tot_scale_for_valid_tokens[i] / tot_special_tokens[i]) / 10.0
                        new_weight_speech = (speech_tot_scale_for_valid_tokens[i] / tot_special_tokens[i]) / 10.0
                    else:
                        # No special tokens found — set weight to zero
                        new_weight_text = 0.0
                        new_weight_speech = 0.0

                    # set text eos/bos scale
                    # Assign new_weight to BOS tokens in the current sequence
                    loss_scale[i, :, :1] = torch.where(
                        text_labels_exp[i, :, :1] == self.text_bos_id, new_weight_text, loss_scale[i, :, :1]
                    )

                    # Assign new_weight to EOS tokens in the current sequence
                    loss_scale[i, :, :1] = torch.where(
                        text_labels_exp[i, :, :1] == self.text_eos_id, new_weight_text, loss_scale[i, :, :1]
                    )
                    # set speech bos/eos scale
                    # Assign new_weight to BOS tokens in the current sequence
                    loss_scale[i, :, 1:] = torch.where(
                        audio_labels[i, :, :] == self.speech_bos_id, new_weight_speech, loss_scale[i, :, 1:]
                    )

                    # Assign new_weight to EOS tokens in the current sequence
                    loss_scale[i, :, 1:] = torch.where(
                        audio_labels[i, :, :] == self.speech_eos_id, new_weight_speech, loss_scale[i, :, 1:]
                    )
            else:
                raise ValueError(f"Unknown scale_loss_by: {self.cfg.scale_loss_by}")

        # debug samples:
        if (
            self.cfg.get("debug_dataloader_audios_path", None)
            and self.training
            and "s2s_duplex_overlap_as_s2s_duplex" not in batch["formatter"][0]
        ):

            def count_leading_silence_tokens(tensor: torch.Tensor, silence_token: int = 0) -> int:
                """
                Count the number of consecutive silence tokens at the beginning of a 1D tensor.

                Args:
                    tensor (torch.Tensor): 1D tensor of tokens.
                    silence_token (int): The token considered as silence (default: 0).

                Returns:
                    int: Number of consecutive silence tokens at the beginning.
                """
                if tensor.ndim != 1:
                    raise ValueError("Input tensor must be 1D.")

                count = 0
                for token in tensor:
                    if token.item() == silence_token:
                        count += 1
                    else:
                        break
                return count

            def write_wave(one_audio_signal, file_name, sr=None):
                import numpy as np
                import soundfile as sf

                one_audio_signal = one_audio_signal.cpu().numpy()
                one_audio_signal = one_audio_signal.astype(np.float32)
                if sr is None:
                    sr = self.target_sample_rate
                # one_audio_signal = np.clip(one_audio_signal, -1.0, 1.0)
                sf.write(file_name, one_audio_signal, sr)

            # encode and decode the audio
            with fp32_precision(), torch.no_grad():
                lengths = torch.tensor([batch["target_audio"].shape[1]] * batch["target_audio"].shape[0]).to(
                    self.audio_codec.device
                )
                reconstructed_audio_from_wav, _ = self.audio_codec(audio=batch["target_audio"], audio_len=lengths)
                # reconstruct wav
                audio_labels_ = replace_control_speech_codes(audio_labels, self._control_codes)
                with fp32_precision(), torch.no_grad():
                    lengths = torch.tensor([audio_labels_.shape[1]] * audio_labels_.shape[0]).to(
                        self.audio_codec.device
                    )
                    reconstructed_audio_from_tokens, _ = self.audio_codec.decode(
                        tokens=audio_labels_.transpose(1, 2), tokens_len=lengths
                    )

            for i in range(audio_labels_.shape[0]):
                write_wave(
                    batch["target_audio"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"target_audio_{i}.wav"),
                    sr=self.target_sample_rate,
                )
                write_wave(
                    batch["target_first_turn_audio"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"speaker_ref_{i}.wav"),
                    sr=self.target_sample_rate,
                )
                write_wave(
                    batch["source_audio"][i],
                    os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"source_audio_{i}.wav"),
                    sr=self.source_sample_rate,
                )

                write_wave(
                    reconstructed_audio_from_tokens[i],
                    os.path.join(
                        self.cfg.get("debug_dataloader_audios_path"), f"target_audio_reconstructed_from_tokens_{i}.wav"
                    ),
                    sr=self.target_sample_rate,
                )

                write_wave(
                    reconstructed_audio_from_wav[i],
                    os.path.join(
                        self.cfg.get("debug_dataloader_audios_path"),
                        f"target_audio_reconstructed_from_waveform_{i}.wav",
                    ),
                    sr=self.target_sample_rate,
                )
                if self.cfg.get("use_eou_decoder", None) or self.cfg.get("llm_predict_eou", None):
                    repeat_factor = int(self.target_sample_rate / self.target_fps)
                    eou_wav = (
                        eou_labels[i].unsqueeze(0).unsqueeze(-1).repeat(1, 1, repeat_factor)
                    )  # (B, T, repeat_factor)
                    eou_wav = eou_wav.view(1, -1)  # (B, T * repeat_factor)
                    eou_wav = eou_wav.float() * 0.8  #  make 1 audible and keep 0 as total silence
                    write_wave(
                        eou_wav.squeeze(),
                        os.path.join(self.cfg.get("debug_dataloader_audios_path"), f"eou_{i}.wav"),
                        sr=self.target_sample_rate,
                    )

            num_bos_tokens = (text_labels.unsqueeze(-1) == self.text_bos_id).flatten(1, 2).sum(-1)
            # Count how many EOS tokens are present per sequence
            # Shape: [B]
            num_eos_tokens = (text_labels.unsqueeze(-1) == self.text_eos_id).flatten(1, 2).sum(-1)
            print("Num eos:", num_eos_tokens, "num bos:", num_bos_tokens)
            # check text
            print(
                "text_labels decoded:",
                tokens_to_str(
                    text_labels[-1:], target_codes_lens - 1, tokenizer=self.tokenizer, pad_id=self.text_pad_id
                ),
            )
            print(
                "target labels from dataloader decoded:",
                tokens_to_str(
                    batch["target_tokens"][-1:],
                    target_codes_lens - 1,
                    tokenizer=self.tokenizer,
                    pad_id=self.text_pad_id,
                ),
            )
            print(
                "Number of padding tokens on the begining:",
                count_leading_silence_tokens(text_labels[-1:].squeeze(), self.text_pad_id),
            )

            print(batch["formatter"])
            if audio_labels_.shape[0] > 1:
                exit()

        return {
            "input_embeds": input_embeds,
            "input_lens": source_encoded_lens - 1,
            "output_lens": target_codes_lens - 1,
            "text_labels": text_labels,
            "eou_labels": eou_labels,
            "eou_logits": eou_logits,
            "eou_loss_scale": eou_loss_scale,
            "eou_skip_batch": eou_skip_batch,
            "input_audio_tokens": audio_inputs,
            "audio_labels": audio_labels,
            "seq_mask": seq_mask,
            "loss_scale": loss_scale,
            "perception_emb": source_encoded[:, :-1],
            "asr_emb": asr_emb[:, :-1],
            "speaker_encoder_emb": speaker_encoder_emb,
        }

    def track_param_updates(self, param_filter: str = "speech_generation.text_embeddings", verbose=True):
        if not hasattr(self, "_param_tracker_state"):
            # First-time call: cache current weights
            self._param_tracker_state = {
                name: p.clone().detach() for name, p in self.named_parameters() if param_filter in name
            }
            if verbose:
                print(f"[Tracker] Initialized snapshot for: {[k for k in self._param_tracker_state]}")
            return

        if verbose:
            print(f"\n📊 [Tracker] Comparing parameters with filter '{param_filter}':")

        for name, p in self.named_parameters():
            if param_filter not in name:
                continue
            prev = self._param_tracker_state[name]
            now = p.detach()
            delta = (now - prev).abs().sum()
            mean_delta = delta / p.numel()

            if delta.item() == 0.0:
                print(f"❌ {name:50s} has NOT been updated (Δsum = 0.0)")
            else:
                print(f"✅ {name:50s} Δsum={delta.item():.4e}  Δmean={mean_delta.item():.4e}")

            # update tracker state
            self._param_tracker_state[name] = now.clone().detach()

    def training_step(self, batch: dict, batch_idx: int):
        for m in (self.perception.preprocessor, self.perception.encoder, self.llm, self.speech_generation):
            if is_frozen(m):
                m.eval()

        if self.cfg.get("use_eou_decoder", None):
            if is_frozen(self.eou_decoder):
                self.eou_decoder.eval()

        if self.cfg.get("llm_predict_eou", None):
            if self.cfg.get("llm_use_extra_eou_waveform_encoder", False):
                if is_frozen(self.eou_wav_encoder):
                    self.eou_wav_encoder.eval()

        # self.track_param_updates("speech_generation.")

        inputs = self.prepare_inputs(batch)
        forward_outputs = self(
            inputs["input_embeds"],
            input_audio_tokens=inputs["input_audio_tokens"],
            seq_mask=inputs["seq_mask"],
            target_text_tokens=inputs["text_labels"],
            modality_adapter_emb=inputs["perception_emb"],
            asr_emb=inputs["asr_emb"],
            speaker_encoder_emb=inputs["speaker_encoder_emb"],
        )
        num_frames = inputs["input_lens"].sum()
        with loss_parallel():
            # mask audio logits to ignore sequence padding
            text_logits = forward_outputs["text_logits"]
            if self.cfg.get("mask_sequence_loss", True):
                text_logits = text_logits * inputs["seq_mask"][:, :, 0].unsqueeze(-1)

            text_loss = (
                torch.nn.functional.cross_entropy(
                    text_logits.flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["text_labels"].flatten(0, 1),
                    reduction="none",
                )
                * inputs["loss_scale"][:, :, 0].flatten(0, 1)
            ).sum(-1) / num_frames

            # mask audio logits to ignore sequence padding
            audio_logits = forward_outputs["audio_logits"]
            if self.cfg.get("mask_sequence_loss", True):
                audio_logits = audio_logits * inputs["seq_mask"][:, :, -1].unsqueeze(-1).unsqueeze(-1)

            audio_loss = (
                torch.nn.functional.cross_entropy(
                    audio_logits.flatten(0, 2),  # (B, T, K, Vs) -> (*, Vs)
                    inputs["audio_labels"].flatten(0, 2),
                    reduction="none",
                )
                * inputs["loss_scale"][:, :, 1:].flatten(0, 2)
            ).sum(-1) / (num_frames * self._num_codebooks)

        loss = self.cfg.text_loss_weight * text_loss + self.cfg.audio_loss_weight * audio_loss

        eou_loss = 0.0
        if self.cfg.get("use_eou_decoder", None) and not inputs["eou_skip_batch"]:
            eou_loss = (
                torch.nn.functional.cross_entropy(
                    inputs["eou_logits"].flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["eou_labels"].flatten(0, 1),
                    reduction="none",
                )
                * inputs["eou_loss_scale"].flatten(0, 1)
            ).sum(-1) / num_frames

            loss = loss + eou_loss * self.cfg.get("eou_loss_weight", 2.0)

        if self.cfg.get("llm_predict_eou", None) and not inputs["eou_skip_batch"]:
            eou_loss = (
                torch.nn.functional.cross_entropy(
                    forward_outputs["eou_logits"].flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["eou_labels"].flatten(0, 1),
                    reduction="none",
                )
                * inputs["eou_loss_scale"].flatten(0, 1)
            ).sum(-1) / num_frames

            loss = loss + eou_loss * self.cfg.get("eou_loss_weight", 2.0)

        if self.cfg.get("llm_predict_eou", None) and not inputs["eou_skip_batch"]:
            eou_loss = (
                torch.nn.functional.cross_entropy(
                    forward_outputs["eou_logits"].flatten(0, 1),  # (B, T, Vt) -> (*, Vt)
                    inputs["eou_labels"].flatten(0, 1),
                    reduction="none",
                )
                * inputs["eou_loss_scale"].flatten(0, 1)
            ).sum(-1) / num_frames

            loss = loss + eou_loss * self.cfg.get("eou_loss_weight", 2.0)

        B, T = inputs["input_embeds"].shape[:2]
        ans = {
            "loss": loss,
            "learning_rate": (
                torch.as_tensor(self.trainer.optimizers[0].param_groups[0]['lr'] if self._trainer is not None else 0)
            ),
            "text_loss": text_loss,
            "audio_loss": audio_loss,
            "batch_size": B,
            "sequence_length": T,
            "num_frames": num_frames.to(torch.float32),  # avoid warning
            "padding_ratio": num_frames / (B * T),
        }

        if self.cfg.get("use_eou_decoder", None) or self.cfg.get("llm_predict_eou", None):
            ans["eou_loss"] = eou_loss

        self.log_dict(ans, on_step=True)
        return ans

    def on_train_epoch_start(self) -> None:
        setup_audio_codec(self)  # potentially reloads the audio codec to make sure it's in fp32
        if hasattr(self.speech_generation, "use_speaker_encoder") and self.speech_generation.use_speaker_encoder:
            self.speech_generation.setup_speaker_encoder()  # potentially reloads the speaker encoder to make sure it's in fp32

    def on_validation_epoch_start(self) -> None:
        self.on_train_epoch_start()
        self.results_logger = ResultsLogger(self.validation_save_path).reset()

        self.asr_bleu = ASRBLEU(self.cfg.scoring_asr).reset()
        self.bleu = BLEU().reset()
        tolerance = int(
            self.cfg.get("val_acc_tolerance", 160) / (1000 / self.target_fps)
        )  # 160 ms as default tolerance --> 2 tokens for 12.5FPS and 1 for 25FPS
        self.text_bos_acc = TokenAccuracy(
            token_name="text_bos", token_id=self.text_bos_id, tolerance=tolerance
        ).reset()
        self.text_eos_acc = TokenAccuracy(
            token_name="text_eos", token_id=self.text_eos_id, tolerance=tolerance
        ).reset()

    def on_validation_epoch_end(self, prefix="val") -> None:
        asr_bleu = self.asr_bleu.compute()
        for k, m in asr_bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        bleu = self.bleu.compute()
        for k, m in bleu.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        text_bos_acc = self.text_bos_acc.compute()
        for k, m in text_bos_acc.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)
        text_eos_acc = self.text_eos_acc.compute()
        for k, m in text_eos_acc.items():
            self.log(f"{prefix}_{k}", m.to(self.device), on_epoch=True, sync_dist=True)

    def validation_step(self, batch: dict, batch_idx: int):

        # Update speaker embedding to reflect the one in the prompt during inference
        if self.speech_generation.use_speaker_encoder and self.speech_generation.inference_speaker_reference:
            self.speech_generation.update_inference_speaker_embedding(
                self.speech_generation.inference_speaker_reference
            )

        for name, dataset_batch in batch.items():
            if dataset_batch is None:
                continue  # some dataset is exhausted

            results = self.offline_inference(
                dataset_batch["source_audio"],
                dataset_batch["source_audio_lens"],
            )

            with fp32_precision():  # resample is fragile to bfloat16 default dtype
                asr_hyps = self.asr_bleu.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    pred_audio=resample(results["audio"], 22050, 16000),
                    pred_audio_lens=(results["audio_len"] / 22050 * 16000).to(torch.long),
                )

                self.results_logger.update(
                    name=name,
                    refs=dataset_batch["target_texts"],
                    hyps=results["text"],
                    asr_hyps=asr_hyps,
                    samples_id=dataset_batch['sample_id'],
                    pred_audio=results["audio"],
                    pred_audio_sr=self.target_sample_rate,
                    user_audio=dataset_batch["source_audio"],
                    user_audio_sr=self.source_sample_rate,
                    eou_pred=(
                        results["gen_eou"]
                        if "gen_eou" in results
                        else None
                    ),
                    fps=self.source_fps,
                    results=results if self.cfg.get("dump_tokens_text", False) else None,
                    tokenizer=self.tokenizer,
                )

            self.bleu.update(name=name, refs=dataset_batch["target_texts"], hyps=results["text"])
            self.text_bos_acc.update(name=name, refs=dataset_batch["target_tokens"], hyps=results["tokens_text"])
            self.text_eos_acc.update(name=name, refs=dataset_batch["target_tokens"], hyps=results["tokens_text"])

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.on_validation_epoch_end(prefix="test")

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    def _get_bos_embedding(self) -> torch.Tensor:
        """
        Remove the audio codec embedding for the beginning of AR decoding.
        """
        text_bos = torch.full((1,), fill_value=self.text_pad_id, device=self.device)
        input_embeds = self.embed_tokens(text_bos)
        return input_embeds

    @torch.no_grad()
    def offline_inference(
        self,
        input_signal: torch.Tensor,
        input_signal_lens: torch.Tensor,
        decode_audio: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Autoregressive prediction.

        Args:
            input_signal: a batch of waveforms with shape (B, T) with source sampling rate.
            input_signal_lens: example lengths as number of samples of shape (B,).
            decode_audio: bool, whether to decode audio codes to waveform.

        Returns:
            A dict with keys:
                * "text": generated text, de-tokenized to strings, properly skipping text_pad_id; list of length B.
                * "tokens_text": generated text tokens of shape (B, T2).
                * "tokens_audio": generated audio codes of shape (B, T2, K) where `K=num_codebooks`.
                * "tokens_len" output lengths as number of tokens of shape (B,).
                * "audio": generated waveform of shape (B, T3) (`decode_audio=True`).
                * "audio_len" output lengths as number of waveform samples of shape (B,) (when `decode_audio=True`).
        """
        source_encoded, lengths, asr_emb = self.perception(
            input_signal=input_signal, input_signal_length=input_signal_lens, return_encoder_emb=True
        )
        B, T_local, H = source_encoded.shape

        if self.cfg.get("inference_eou_from_bos_eos", None):
            eou_mask_generator = EfficientBatchStreamingSpeakingMaskGenerator(
                device=source_encoded.device,
                max_length=self.cfg.speech_decoder.max_length_causal_mask,
                batch_size=B,
                bos_token_id=self.text_bos_id,
                eos_token_id=self.text_eos_id,
                eou_window=self.cfg.get("inference_force_follow_external_eou_eou_window", 7),
                eos_lookback=self.cfg.get("inference_force_follow_external_eou_eos_lookback", 7),
                bos_lookback=self.cfg.get("inference_force_follow_external_eou_bos_lookback", 7),
                force_bos_from_eou=self.cfg.get("inference_force_follow_external_eou_bos", False),
            )

        # add eou embedding
        if self.cfg.get("use_eou_decoder", None) or self.cfg.get("inference_use_external_eou_predictor", None):
            # predict eou logits
            if self.cfg.get("eou_decoder_from_wav", None) or self.cfg.get(
                "inference_use_external_eou_predictor", None
            ):
                eou_logits, _ = self.eou_decoder(input_signal.to(source_encoded.dtype), input_signal_lens)
                if source_encoded.size(1) > eou_logits.size(1):
                    # Pad on the right (end of time axis)
                    pad_len = source_encoded.size(1) - eou_logits.size(1)
                    eou_logits = torch.nn.functional.pad(
                        eou_logits, pad=(0, 0, 0, pad_len), mode='constant', value=0.0
                    )
                else:
                    eou_logits = eou_logits[:, : source_encoded.size(1)]
            else:
                mask = torch.ones(
                    (asr_emb.size(0), asr_emb.size(1)),
                    device=asr_emb.device,
                )
                eou_logits = self.eou_decoder(asr_emb, x_mask=mask)

            # if not in training time get eou from the eou decoder
            gen_eou = torch.argmax(eou_logits, dim=-1).view(B, T_local).contiguous()
            # add eou embedding to the llm input
            eou_emb = self.eou_embedding(gen_eou)
            if self.cfg.get("inference_eou_from_bos_eos", None):
                external_eou = gen_eou.clone().float()

        # Determine decoding length and pad if FSDP
        if self._use_fsdp:
            T_tensor = torch.tensor([T_local], device=source_encoded.device)
            dist.all_reduce(T_tensor, op=dist.ReduceOp.MAX)
            T = int(T_tensor.item())
            if T > T_local:

                last_frame_source = source_encoded[:, T_local - 1: T_local, :]
                pad_source = last_frame_source.repeat(1, T - T_local, 1)
                source_encoded = torch.cat([source_encoded, pad_source], dim=1)
                last_frame_asr = asr_emb[:, T_local - 1: T_local, :]
                pad_asr = last_frame_asr.repeat(1, T - T_local, 1)
                asr_emb = torch.cat([asr_emb, pad_asr], dim=1)
                if self.cfg.get("use_eou_decoder", None):
                    last_frame_eou = eou_emb[:, T_local - 1: T_local, :]
                    pad_eou = last_frame_eou.repeat(1, T - T_local, 1)
                    eou_emb = torch.cat([eou_emb, pad_eou], dim=1)
        else:
            T = T_local

        # Apply channel weight
        input_embeds = source_encoded.clone()
        input_embeds *= self.cfg.get("duplex_user_channel_weight", 1.0)

        # create first_eou and add eou embedding
        if self.cfg.get("llm_predict_eou", None):
            first_eou = torch.zeros(B, 1).long().to(source_encoded.device)
            # compute eou_emb only for the first frame
            eou_emb = self.eou_embedding(first_eou)

            if self.cfg.get("llm_use_extra_eou_waveform_encoder", False):
                # compute the eou_wav_encoder for the whole sequence as done to the source_encoded
                wav_eou_emb, _ = self.eou_wav_encoder(input_signal.to(source_encoded.dtype), input_signal_lens)
                # ensures that logits and labels has the same shape
                if source_encoded.size(1) > wav_eou_emb.size(1):
                    # Pad on the right (end of time axis)
                    pad_len = source_encoded.size(1) - wav_eou_emb.size(1)
                    wav_eou_emb = torch.nn.functional.pad(
                        wav_eou_emb, pad=(0, 0, 0, pad_len), mode='constant', value=0.0
                    )
                else:
                    wav_eou_emb = wav_eou_emb[:, : source_encoded.size(1)]
                # add only the first frame to the model
                eou_emb = eou_emb + wav_eou_emb[:, :1]

            input_embeds[:, 0] += eou_emb[:, 0]
            gen_eou = torch.empty(B, T, device=self.device, dtype=torch.long)

        elif self.cfg.get("use_eou_decoder", None):
            first_eou = gen_eou[:, :1]
            input_embeds.add_(eou_emb)
        else:
            first_eou = None

        # This cache is for self.llm
        cache = DynamicCache()
        # Call reset_input_and_kv_cache to enable cache for TransformerARSpeechDecoder
        self.speech_generation.reset_input_and_kv_cache(use_cache=True)
        gen_text = torch.empty(B, T, device=self.device, dtype=torch.long)
        gen_audio = torch.empty(B, T, self._num_codebooks, device=self.device, dtype=torch.long)

        # First step, use speech_delay token
        input_embeds[:, 0] += self._get_bos_embedding()
        first_audio = torch.full(
            [B, 1, self._num_codebooks],
            fill_value=self.speech_delay_id,
            device=self.device,
            dtype=torch.long,
        )
        ans = self(
            input_embeds[:, :1],
            cache=cache,
            input_audio_tokens=first_audio,
            seq_mask=None,
            target_text_tokens=None,  # text input will be sampled from llm backbone
            modality_adapter_emb=source_encoded[:, :1],
            asr_emb=asr_emb[:, :1],
            speaker_encoder_emb=None,  # for inference uses the cached inference_speaker_embedding
            eou=first_eou,
        )
        gen_text[:, 0] = ans["text_logits"][:, -1].argmax(dim=-1)
        gen_audio[:, 0] = ans["audio_logits"][:, -1].argmax(dim=-1)

        if self.cfg.get("llm_predict_eou", None):
            gen_eou[:, 0] = ans["eou_logits"][:, -1].argmax(dim=-1)

        if self.cfg.get("inference_eou_from_bos_eos", None):
            gen_eou = torch.empty(B, T, device=self.device, dtype=torch.long)
            gen_eou[:, 0] = eou_mask_generator.step(
                gen_text[:, 0],
                eou_probs=external_eou[:, 0] if self.cfg.get("inference_force_follow_external_eou", None) else None,
            )[:, 0]

        speech_state = torch.zeros(B, device=self.device, dtype=torch.long)
        # Autoregressive loop
        for t in range(1, T):
            if self.cfg.get("llm_predict_eou", None):
                cond_eou = gen_eou[:, t - 1]
                eou_emb = self.eou_embedding(gen_eou[:, t - 1 : t])
                if self.cfg.get("llm_use_extra_eou_waveform_encoder", False):
                    eou_emb = eou_emb + wav_eou_emb[:, t : t + 1]
                input_embeds[:, t] += eou_emb[:, -1]  # add the last eou emb
            elif self.cfg.get("use_eou_decoder", None):
                cond_eou = gen_eou[:, t]
            elif self.cfg.get("inference_eou_from_bos_eos", None):
                cond_eou = gen_eou[:, t - 1]
            else:
                cond_eou = None

            last_emb = self.embed_tokens(gen_text[:, t - 1])
            input_embeds[:, t] += last_emb

            current_audio = gen_audio[:, t - 1 : t, :]
            ans = self(
                input_embeds[:, t : t + 1],
                cache=ans["cache"],
                input_audio_tokens=current_audio,
                seq_mask=None,
                target_text_tokens=None,  # text input will be sampled from llm backbone
                modality_adapter_emb=source_encoded[:, t : t + 1],
                asr_emb=asr_emb[:, t : t + 1],
                speaker_encoder_emb=None,  # for inference uses the cached inference_speaker_embedding
                eou=cond_eou,
            )
            gen_text[:, t] = ans["text_logits"][:, -1].argmax(dim=-1)
            gen_audio[:, t] = ans["audio_logits"][:, -1].argmax(dim=-1)

            if self.cfg.get("llm_predict_eou", None):
                gen_eou[:, t] = ans["eou_logits"][:, -1].argmax(dim=-1)

            if self.cfg.get("inference_eou_from_bos_eos", None):
                gen_eou[:, t] = eou_mask_generator.step(
                    gen_text[:, t],
                    eou_probs=(
                        external_eou[:, t] if self.cfg.get("inference_force_follow_external_eou", None) else None
                    ),
                )[:, t]

            num_transition_tokens = self.cfg.get('inference_eou_num_transition_tokens', 2)
            if self.cfg.get('inference_force_bos_eos_follow_eou', None) and t >= num_transition_tokens:
                # Check last num_transition_tokens EOU predictions: if all 0 and current is 1 → BOS in text
                prev_zeros = (gen_eou[:, t - num_transition_tokens : t] == 0).all(dim=1)
                curr_is_one = gen_eou[:, t] == 1
                force_bos = prev_zeros & curr_is_one

                # force bos
                gen_text[:, t] = torch.where(force_bos, self.text_bos_id, gen_text[:, t])

                # Check last num_transition_tokens EOU predictions: if all 1 and current is 0 → EOS in text
                prev_ones = (gen_eou[:, t - num_transition_tokens : t] == 1).all(dim=1)
                curr_is_zero = gen_eou[:, t] == 0
                force_eos = prev_ones & curr_is_zero
                # force eos
                gen_text[:, t] = torch.where(force_eos, self.text_eos_id, gen_text[:, t])

                not_special = (gen_text[:, t] != self.text_bos_id) & (gen_text[:, t] != self.text_eos_id)
                should_pad = (gen_eou[:, t] == 0) & not_special

                # if EOU is zero, allows text channel only to assume, zero, eou or bos
                gen_text[:, t] = torch.where(should_pad, self.text_pad_id, gen_text[:, t])

            if self.cfg.get('inference_force_bos_eos_follow_eou_speech_channel', None) and t >= num_transition_tokens:
                not_special = (gen_audio[:, t] != self.speech_bos_id) & (gen_audio[:, t] != self.speech_eos_id)
                should_pad = (gen_eou[:, t] == 0) & not_special[:, 0]

                # if EOU is zero, allows text channel only to assume, zero, eou or bos
                gen_audio[:, t] = torch.where(
                    should_pad.unsqueeze(-1),
                    gen_audio[:, 0],  # first token that supposed to be silence
                    gen_audio[:, t],
                )

            if self.cfg.get('inference_force_speech_state', None):
                # state 0 - silence, state 1 - speech
                speech_state = torch.where(
                    gen_text[:, t] == self.text_bos_id, torch.ones_like(speech_state), speech_state
                )
                speech_state = torch.where(
                    gen_text[:, t] == self.text_eos_id, torch.zeros_like(speech_state), speech_state
                )
                gen_audio[:, t] = torch.where(
                    speech_state.unsqueeze(-1) == 0,
                    gen_audio[:, 0],  # silence
                    gen_audio[:, t],  # speech
                )
            # inference trick force speech decoder eos/bos to make the model more robust
            num_speech_delay = 1
            if self.cfg.get('inference_force_speech_bos', None) and num_speech_delay < gen_text.shape[1]:
                gen_audio[:, t] = torch.where(
                    (gen_text[:, t - num_speech_delay].unsqueeze(-1) == self.text_bos_id)
                    * (torch.sum(gen_audio[:, t - num_speech_delay :] == self.speech_bos_id, 1) == 0),
                    self.speech_bos_id,
                    gen_audio[:, t],
                )

            if self.cfg.get('inference_force_speech_eos', None) and gen_text.shape[
                1
            ] > num_speech_delay + self.cfg.get("advance_text_channel_by", 0):
                # tmp solution: force to stop talking if user interruption is detected
                gen_audio[:, t] = torch.where(
                    (
                        (
                            gen_text[:, t - num_speech_delay - self.cfg.get("advance_text_channel_by", 0)].unsqueeze(
                                -1
                            )
                            == self.text_eos_id
                        )
                    ),
                    self.speech_eos_id,
                    gen_audio[:, t],
                )

        # Trim back to local length if padded
        if self._use_fsdp and T > T_local:
            gen_text = gen_text[:, :T_local]
            gen_audio = gen_audio[:, :T_local]

        ans = {
            "text": tokens_to_str(gen_text, lengths, tokenizer=self.tokenizer, pad_id=self.text_pad_id),
            "tokens_text": gen_text,
            "tokens_audio": gen_audio,
            "tokens_len": lengths,
        }

        if decode_audio:
            gen_audio_codes = replace_control_speech_codes(gen_audio, self._control_codes)
            with fp32_precision(), torch.no_grad():
                predicted_audio, predicted_audio_lens = self.audio_codec.decode(
                    tokens=gen_audio_codes.transpose(1, 2), tokens_len=lengths
                )
            ans["audio"] = predicted_audio
            ans["audio_len"] = predicted_audio_lens

        if (
            self.cfg.get("use_eou_decoder", None)
            or self.cfg.get("llm_predict_eou", None)
            or self.cfg.get("inference_use_external_eou_predictor", None) or self.cfg.get("inference_eou_from_bos_eos", None)
        ):
            ans["gen_eou"] = gen_eou

        # Call reset_input_and_kv_cache to reset cache for TransformerARSpeechDecoder
        self.speech_generation.reset_input_and_kv_cache(use_cache=False)
        return ans

    def backward(self, *args, **kwargs):
        with loss_parallel():
            super().backward(*args, **kwargs)

    def configure_optimizers(self):
        return configure_optimizers(self)

    @property
    def oomptimizer_schema(self) -> dict:
        """
        Return a typing schema for optimal batch size calibration for various
        sequence lengths using OOMptimizer.
        """
        return {
            "cls": dict,
            "inputs": [
                {"name": "source_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "source_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {"name": "target_audio", "type": NeuralType(("B", "T"), AudioSignal()), "seq_length": "input"},
                {"name": "target_audio_lens", "type": NeuralType(("B",), LengthsType()), "seq_length": "input"},
                {
                    "name": "target_tokens",
                    "type": NeuralType(("B", "T"), LabelsType()),
                    "seq_length": "output",
                    "vocab_size": self.tokenizer.vocab_size,
                },
            ],
        }

    def configure_model(self) -> None:
        # TODO(pzelasko): refactor into separate module re-usable across models
        device_mesh = self.device_mesh
        if device_mesh is None:
            return

        llm = self.llm
        if isinstance(llm, PeftModel):
            llm = llm.base_model.model

        if (tp_mesh := device_mesh["tensor_parallel"]).size() > 1:
            self._use_tp = True

            plan = {
                "layers.0": PrepareModuleInput(
                    input_layouts=(Replicate(),),  # , None)
                    desired_input_layouts=(Shard(1),),  # , None)
                    use_local_output=True,
                ),
                "norm": SequenceParallel(),
            }
            parallelize_module(llm, tp_mesh, plan)

            for transformer_block in llm.layers:
                plan = {
                    "input_layernorm": SequenceParallel(),
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                    "self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
                    "post_attention_layernorm": SequenceParallel(),
                    "mlp": PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "mlp.gate_proj": ColwiseParallel(),
                    "mlp.up_proj": ColwiseParallel(),
                    "mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
                    # "pre_feedforward_layernorm": SequenceParallel(),
                    # "post_feedforward_layernorm": SequenceParallel(),
                }

                # Adjust attention module to use the local number of heads
                attn_layer = transformer_block.self_attn
                for attr in ("num_heads", "num_key_value_heads", "hidden_size"):
                    val = getattr(attn_layer, attr)
                    if val % tp_mesh.size() != 0:
                        logging.warning(
                            f"attn_layer.{attr}={val} is not divisible by {tp_mesh.size()=}: "
                            f"set a different tensor parallelism size to avoid errors."
                        )
                    setattr(attn_layer, attr, val // tp_mesh.size())

                parallelize_module(transformer_block, tp_mesh, plan)

            for m in (self.lm_head, self.audio_head):
                parallelize_module(
                    m,
                    tp_mesh,
                    ColwiseParallel(
                        input_layouts=Shard(1),
                        output_layouts=Shard(-1),
                        use_local_output=False,
                    ),
                )

        if (dp_mesh := device_mesh["data_parallel"]).size() > 1:
            assert dp_mesh.ndim == 1
            self._use_fsdp = True

            fsdp_config = {"mesh": dp_mesh}

            for idx, layer in enumerate(llm.layers):
                llm.layers[idx] = fully_shard(layer, **fsdp_config)
            self.embed_tokens = fully_shard(self.embed_tokens, **fsdp_config)
            self.llm = fully_shard(self.llm, **fsdp_config)
            self.lm_head = fully_shard(self.lm_head, **fsdp_config)
            self.perception = fully_shard(self.perception, **fsdp_config)
            if self.cfg.get("use_eou_decoder", None):
                self.eou_decoder = fully_shard(self.eou_decoder, **fsdp_config)
            self.speech_generation = fully_shard(self.speech_generation, **fsdp_config)

    def load_state_dict(self, state_dict, strict: bool = True):
        try:
            super().load_state_dict(state_dict, strict=strict)
        except RuntimeError as e:
            logging.info(f"Error loading model state_dict !! Retrying with partial initialization!")
            model_dict = set_model_dict_for_partial_init(state_dict, self.state_dict())
            super().load_state_dict(model_dict, strict=False)
