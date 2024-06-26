# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Optional, Union

import numpy as np
import torch
from lhotse.dataset.dataloading import resolve_seed
from lhotse.serialization import load_jsonl
from lhotse.utils import Pathlike

from nemo.collections.common.data.lhotse.nemo_adapters import expand_sharded_filepaths
from nemo.collections.common.tokenizers.aggregate_tokenizer import AggregateTokenizer, TokenizerWrapper
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging
from nemo.collections.common.data.lhotse.text_adapters import NeMoSFTExample

"""
Chat template of Gemma:
<start_of_turn>user\nWho are you?<end_of_turn>\n<start_of_turn>model\nI am a large language model.<end_of_turn>\n

Reference:
https://huggingface.co/google/gemma-7b-it/blob/18329f019fb74ca4b24f97371785268543d687d2/tokenizer_config.json#L1507
"""

# the prefix string used in the tokenizer to deal with the added empty token for some of the tokenizers
PREFIX_STR = ""
IGNORE_INDEX = -100
SPEAKER_MAPPING = {
    "User": "user",
    "Assistant": "model",
}


@dataclass
class GemmaSFTJsonlAdapter:
    """
    ``GemmaSFTJsonlAdapter`` is used to read a Gemma LM SFT Chat JSONL file and yield objects of type
    ``GemmaSFTExample`` that can be sampled with Lhotse.

    We expect the following schema (contained in a single line per example)::

        {
            "conversations": [
                {
                    "value": str,
                    "from": "User" | "Assistant",
                    "canonical_form": str,
                    "label": str | null
                },
                ...
            ],
            "mask": "User" | "Assistant",
            "system": str,
            "dataset": str,
            "category": str,
        }

    Refer to examples of this format here:

    * TODO: links to examples?
    * TODO: links to more detailed schema definition?
    """

    paths: Union[Pathlike, list[Pathlike]]
    language: str | None = None
    shuffle_shards: bool = False
    shard_seed: Union[int, Literal["trng", "randomized"]] = "trng"

    def __post_init__(self):
        self.paths = expand_sharded_filepaths(self.paths)

    def __iter__(self) -> Iterator["GemmaSFTExample"]:
        paths = self.paths
        if self.shuffle_shards:
            seed = resolve_seed(self.shard_seed)
            random.Random(seed).shuffle(paths)
        for path in paths:
            for data in load_jsonl(path):
                yield GemmaSFTExample(data, language=self.language)


@dataclass
class GemmaSFTExample(NeMoSFTExample):
    data: dict
    language: str | None = None
    input_ids: np.ndarray | None = None
    context_ids: np.ndarray | None = None
    answer_ids: np.ndarray | None = None
    mask: np.ndarray | None = None
    metadata: dict | None = None

    def tokenize(self, tokenizer: TokenizerWrapper | TokenizerSpec) -> "GemmaSFTExample":
        """
        Create a tokenized variant of this example given a tokenizer (i.e. fill the optional fields).
        Supports BPE tokenizers and aggregate tokenizers.

        The tokenization is compatible with Megatron's :class:`GPTSFTChatDataset`.
        """
        special_tokens = {
            "system_turn_start": None,  # Gemma does not have system instructions: https://ai.google.dev/gemma/docs/formatting#system-instructions
            "turn_start": "<start_of_turn>",
            "label_start": None,
            "end_of_turn": "<end_of_turn>\n",
            "end_of_name": "\n",
        }

        if isinstance(tokenizer, TokenizerWrapper):
            tokenizer = tokenizer._tokenizer
        if isinstance(tokenizer, AggregateTokenizer):
            assert self.language is not None, (
                f"Error: attempted to use AggregateTokenizer for NeMoSFTExample which did not specify language. "
                f"Problematic example: {self}"
            )
            assert self.language in tokenizer.tokenizers_dict, (
                f"Error: attempted to use AggregateTokenizer for NeMoSFTExample with unsupported language: {self.language}. "
                f"The set of supported languages is: {' '.join(tokenizer.tokenizers_dict.keys())}. "
                f"Problematic example: {self}"
            )
            tokenizer = tokenizer.tokenizers_dict[self.language]

        label_start_tokens, name_end_token_ids, num_turn_start_tokens = _build_samples_mapping(
            tokenizer, special_tokens
        )

        tokenized = preprocess(
            source=self.data,
            tokenizer=tokenizer,
            name_end_token_ids=name_end_token_ids,
            label_start_ids=label_start_tokens,
            special_tokens=special_tokens,
            num_turn_start_tokens=num_turn_start_tokens,
        )
        self.input_ids = tokenized["input_ids"].numpy()
        self.context_ids = tokenized["context_ids"].numpy()
        self.answer_ids = tokenized["answer_ids"].numpy()
        self.mask = tokenized["mask"].numpy()
        self.metadata = {k: v for k, v in self.data.items() if k not in ['conversations']}

        return self

    # TODO(pzelasko): for mini-batch sampling purposes, should we consider input_ids or answer_ids
    #                 as representative of the sequence length? Putting input_ids here for now.

    @property
    def tokens(self) -> np.ndarray:
        return self.input_ids

    @property
    def num_tokens(self) -> int:
        return self.input_ids.shape[0]


def _build_samples_mapping(tokenizer, special_tokens):
    # Copied from gpt_sft_chat_dataset.py
    END_NAME_SIGNAL = special_tokens['end_of_name']

    label_start_tokens = None

    id1 = tokenizer.text_to_ids(PREFIX_STR + END_NAME_SIGNAL)
    id2 = tokenizer.text_to_ids(PREFIX_STR)
    name_end_token_ids = id1[len(id2) :]

    id1 = tokenizer.text_to_ids(PREFIX_STR + special_tokens['turn_start'])
    id2 = tokenizer.text_to_ids(PREFIX_STR)
    num_turn_start_tokens = len(id1) - len(id2)

    return label_start_tokens, name_end_token_ids, num_turn_start_tokens


"""
The code below is copied from nemo/collections/nlp/data/language_modeling/megatron/gpt_sft_chat_dataset.py
with minimal modifications in order to avoid importing the NLP collection.

We require this code for on-the-fly text example tokenization in a compatible way with Megatron,
so that we can determine the mini-batch sizes using the token counts.
"""
def preprocess(
    source: dict,
    tokenizer: TokenizerSpec,
    name_end_token_ids: int,
    label_start_ids: list,
    special_tokens: dict,
    num_turn_start_tokens: int,
):
    """
    Given a conversation list. This transform:
    1. Add start and end signals for each turn;
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask user words with IGNORE_INDEX.
    """
    header, conversation, data_type, mask_role = _get_header_conversation_type_mask_role(source, special_tokens)
    assert len(header) == 0, header
    assert label_start_ids is None, label_start_ids

    # tokenize conversations
    input_ids = tokenizer.text_to_ids(conversation)
    target = copy.deepcopy(input_ids)
    # header_tokens = tokenizer.text_to_ids(header)
    # header_len = len(header_tokens)
    header_len = 0
    
    ids = []
    tokenized_lens = []
    # assert torch.equal(torch.tensor(target[:header_len]), torch.tensor(header_tokens))
    for s in source['conversations']:
        # hack to remove the extra empty token in front
        id1 = tokenizer.text_to_ids(PREFIX_STR + s["value"])
        id2 = tokenizer.text_to_ids(PREFIX_STR)
        tokenized_sentence = id1[len(id2) :]
        ids.append(torch.tensor(tokenized_sentence))
        tokenized_lens.append(len(tokenized_sentence))
    speakers = [sentence["from"] for sentence in source['conversations']]
    assert mask_role in speakers, "mask role not in the conversation"
    target = torch.LongTensor(target)
    # not going to train on the header
    # target[:header_len] = IGNORE_INDEX
    input_ids = torch.LongTensor(input_ids)
    _mask_targets(
        target,
        tokenized_lens,
        speakers,
        header_len,
        ids,
        tokenizer,
        mask_role,
        data_type,
        name_end_token_ids,
        special_tokens,
        label_start_ids,
        num_turn_start_tokens,
    )

    # add bos and eos
    bos = torch.LongTensor([tokenizer.bos_id])
    eos = torch.LongTensor([tokenizer.eos_id])
    input_ids = torch.cat([bos, input_ids, eos])
    target = torch.cat([bos, target, eos])

    mask = (target != IGNORE_INDEX).bool()
    assert mask.sum().item() != 0, "mask is empty"
    # Choose the last conversation as answer other history are context
    last_ignore_index_pos = torch.nonzero(target == IGNORE_INDEX)[-1].item() + 1
    context_ids = input_ids[:last_ignore_index_pos]
    answer_ids = input_ids[last_ignore_index_pos:]
    return dict(input_ids=input_ids, mask=mask, context_ids=context_ids, answer_ids=answer_ids)


def _get_header_conversation_type_mask_role(source, special_tokens):
    END_SIGNAL = special_tokens['end_of_turn']
    END_NAME_SIGNAL = special_tokens['end_of_name']
    data_type = None
    mask_role = source.get('mask', 'User')
    header = ""     # Gemma does not use system instructions
    conversation = _add_speaker_and_signal(header, source['conversations'], mask_role, data_type, special_tokens)
    return header, conversation, data_type, mask_role


def _add_speaker_and_signal(header, source, mask_role, gtype, special_tokens):
    TURN_TOKEN = special_tokens['turn_start']
    END_SIGNAL = special_tokens['end_of_turn']
    END_NAME_SIGNAL = special_tokens['end_of_name']
    assert gtype is None, gtype

    """Add speaker and start/end signal on each round."""
    conversation = header
    for i, sentence in enumerate(source):
        sentence_from = sentence["from"]
        sentence["value"] = (
            TURN_TOKEN + SPEAKER_MAPPING[sentence_from] + END_NAME_SIGNAL + sentence["value"] + END_SIGNAL
        )
        conversation += sentence["value"]
        # if the last turn is not masked, add next token start token to the end, which will be included for loss calculation
        if sentence_from != mask_role and i == len(source) - 1:
            conversation += TURN_TOKEN
    return conversation


def _mask_targets(
    target,
    tokenized_lens,
    speakers,
    header_len,
    s_ids,
    tokenizer,
    mask_role,
    gtype,
    name_end_token_ids,
    special_tokens,
    label_start_ids,
    num_turn_start_tokens,
):
    """This function masks the tokens so the loss is computed only on the non-masked role's responses.

    Args:
        target (Tensor): input ids
        tokenized_lens (List[int]): array of lengths of each turns
        speakers (List[str]): array of speakers of each turns
        header_len (int): the system prompt length
        s_ids (List[Tensor]): array of tokenized ids of each turns
        tokenizer (TokenizerSpec): tokenizer object
        mask_role (str): the speaker id to be masked from loss computation
        gtype (str): either 'TEXT_TO_VALUE' or 'VALUE_TO_TEXT'
        name_end_token_ids (int): end of name token ids
        special_tokens (dict): special tokens used for the chat prompt. It has the keys: system_turn_start, turn_start, label_start, end_of_turn
        label_start_ids (list): list of label start token ids,
        num_turn_start_tokens (int): number of tokens of the turn_start str
    """

    assert header_len == 0, header_len
    assert gtype is None, gtype
    assert label_start_ids is None, label_start_ids

    TURN_TOKEN = special_tokens['turn_start']
    END_NAME_SIGNAL = special_tokens['end_of_name']
    name_end_token_ids = torch.tensor(name_end_token_ids)

    cur_idx = header_len
    tgt_len = target.shape[0]
    for i, (tokenized_len, speaker, s_id) in enumerate(zip(tokenized_lens, speakers, s_ids)):
        # note, sentence piece will add extra empty token in front. has to compute the diff
        id1 = tokenizer.text_to_ids(PREFIX_STR)
        id2 = tokenizer.text_to_ids(PREFIX_STR + TURN_TOKEN + SPEAKER_MAPPING[speaker] + END_NAME_SIGNAL)
        skip_name_len = len(id2) - len(
            id1
        )  # s_ids[:skip_name_len] is the name part of the prompt 'TURN_TOKEN + speaker + END_NAME_SIGNAL'

        if cur_idx >= tgt_len:
            break
        # elif cur_idx + tokenized_len < tgt_len:
        #     # Check whether the mask is applied to the correct position, the first token is turn start tokens
        #     if not torch.equal(target[cur_idx + 1 : cur_idx + tokenized_len], s_id[1:]):
        #         logging.warning("a sentence mismatches the corresponding piece " "in the conversation")
        if i == 0:
            # mask the first turn completely to provide at least one turn as context for the rest
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        # elif speaker == mask_role and i == 1 and gtype == 'TEXT_TO_VALUE':
        #     # leave the first turn start tag unmasked, servers severs as the end of turn signal
        #     target[cur_idx + num_turn_start_tokens : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == mask_role and (i > 1):
            # leave the first turn start tag unmasked, which severs as the end of turn signal
            target[cur_idx + num_turn_start_tokens : cur_idx + tokenized_len] = IGNORE_INDEX
        elif speaker == mask_role and (i <= 1):
            # this case should not happen since the second turn is always model
            # mask out everything in the second turn
            target[cur_idx : cur_idx + tokenized_len] = IGNORE_INDEX
        else:
            # mask just the name part
            target[cur_idx : cur_idx + skip_name_len] = IGNORE_INDEX
        cur_idx += tokenized_len
