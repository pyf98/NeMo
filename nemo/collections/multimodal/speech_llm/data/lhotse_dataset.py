from typing import Optional

import torch.utils.data
from lhotse.cut import Cut, CutSet
from lhotse.dataset import AudioSamples
from lhotse.dataset.collation import collate_vectors as collate_vectors_lhotse

from nemo.collections.common.data.lhotse.text_adapters import NeMoSFTExample, SourceTargetTextExample
from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import (
    TextProcessing,
    build_loss_mask,
    ceil_to_nearest,
)
from nemo.collections.multimodal.speech_llm.data.utils import (
    combine_context_and_instruction,
    generate_canary_instruction,
)
from nemo.collections.common.data.lhotse.text_adapters_gemma import GemmaSFTExample
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec


def collate_vectors(items, max_length: int, padding_value):
    vectors = collate_vectors_lhotse(items, padding_value=padding_value)
    if max_length > vectors.size(1):
        vectors = torch.cat(
            [vectors, padding_value * torch.ones(vectors.size(0), max_length - vectors.size(1), dtype=vectors.dtype)],
            dim=1,
        )
    if items[0].shape[0] < 1:
        vectors = vectors.long()
    return vectors


class LhotseAudioQuestionAnswerDataset(torch.utils.data.Dataset):
    """
    This dataset is based on Lhotse ASR dataset from ``audio_to_text_lhotse.py``
    and ``TarredAudioQuestionAnswerDataset`` from ``audio_text_qa_dataset.py``.

    Unlike native NeMo datasets, Lhotse dataset defines only the mapping from
    a CutSet (meta-data) to a mini-batch with PyTorch tensors.
    Specifically, it performs tokenization, I/O, augmentation, and feature extraction (if any).
    Managing data, sampling, de-duplication across workers/nodes etc. is all handled
    by Lhotse samplers instead.

    Args:
        text_processor: TextProcessing object
        default_context: Default question to use if no question is provided
        tokens_to_generate: Number of tokens to generate during inference
        pad_to_max_length: Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        max_seq_length: Maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        context_key: Key to use for the context in your JSONL file
        default_context_key: Key to use for the default context in lhotse yaml
    """

    def __init__(
        self,
        text_processor: TextProcessing,
        default_context: str,
        tokens_to_generate: int,
        pad_to_max_length: bool,
        max_seq_length: int,
        context_key: str = "context",
        default_context_key: str = "default_context",
    ):
        super().__init__()
        self.text_processor = text_processor
        self.load_audio = AudioSamples(fault_tolerant=True)
        self.tokens_to_generate = tokens_to_generate
        self.pad_to_max_length = pad_to_max_length
        self.max_seq_length = max_seq_length

        self.default_context = default_context
        self.context_key = context_key
        self.default_context_key = default_context_key

    def __getitem__(self, all_cuts: CutSet) -> dict[str, torch.Tensor | list[str] | dict]:
        ans = {}

        # convert audio cuts to mini-batch
        cuts = all_cuts.filter(lambda c: isinstance(c, Cut))
        if cuts:
            audio, audio_lens, cuts = self.load_audio(cuts)

            return_batch = {}
            audio_ratio = [1.0] * len(cuts)
            for _, cut in enumerate(cuts):
                if hasattr(cut, self.context_key):
                    cut.context = getattr(cut, self.context_key)
                elif hasattr(cut, self.default_context_key):
                    cut.context = getattr(cut, self.default_context_key)
                else:
                    cut.context = self.default_context

            metadata = []
            for id, cut in enumerate(cuts):
                metadata.append({'audio_filepath': cut.id + '.wav'})

            collated_text_data = collate_text_data(
                cuts=cuts,
                default_context=self.default_context,
                text_processor=self.text_processor,
                tokens_to_generate=self.tokens_to_generate,
                pad_to_max_length=self.pad_to_max_length,
                max_seq_length=self.max_seq_length,
            )
            return_batch.update(
                {
                    "sample_ids": list(cuts.ids),
                    "audio_signal": audio,
                    "audio_signal_length": audio_lens,
                    "audio_ratio": torch.FloatTensor(audio_ratio),
                    "metadata": metadata,
                    **collated_text_data,
                }
            )
            ans.update(return_batch)

        # convert text examples to tensors
        text_examples = all_cuts.filter(lambda c: isinstance(c, (SourceTargetTextExample, NeMoSFTExample)))
        if text_examples:
            pad_id = self.text_processor.pad_id
            text_minibatch = dict(
                text_input_ids=collate_vectors_lhotse([e.input_ids for e in text_examples], padding_value=pad_id),
                text_answer_ids=collate_vectors_lhotse([e.answer_ids for e in text_examples], padding_value=pad_id),
                text_context_ids=collate_vectors_lhotse([e.context_ids for e in text_examples], padding_value=pad_id),
                text_masks=collate_vectors_lhotse([e.mask for e in text_examples], padding_value=0),
            )
            ans.update(text_minibatch)

        return ans


class LhotseAudioChatDataset(torch.utils.data.Dataset):
    """
    This dataset follows the chat template consisting of multi-turn conversations.

    Args:
        
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        audio_locator: str,
        # tokens_to_generate: int,
        # pad_to_max_length: bool,
        # max_seq_length: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.audio_locator = audio_locator
        self.audio_locator_ids = tokenizer.text_to_ids(audio_locator)
        self.load_audio = AudioSamples(fault_tolerant=True)
        # self.text_processor = text_processor
        # self.tokens_to_generate = tokens_to_generate
        # self.pad_to_max_length = pad_to_max_length
        # self.max_seq_length = max_seq_length

    def __getitem__(self, all_cuts: CutSet) -> dict[str, torch.Tensor | list[str] | dict]:
        ans = {}

        # convert audio cuts to mini-batch
        cuts = all_cuts.filter(lambda c: isinstance(c, Cut))
        if cuts:
            audio, audio_lens, cuts = self.load_audio(cuts)

            return_batch = {}
            audio_ratio = [1.0] * len(cuts)
            metadata = []
            sft_examples = []
            for _, cut in enumerate(cuts):
                metadata.append({'audio_filepath': cut.id + '.wav'})

                breakpoint()

                # create chat-style SFT data
                chat_format = getattr(cut, "chat_format", "chat")
                if chat_format == "canary":
                    entry = {
                        "taskname": cut.taskname,
                        "source_lang": cut.source_lang,
                        "target_lang": cut.target_lang,
                        "pnc": cut.pnc,
                        "answer": cut.answer,
                    }
                    sft_data = {
                        "system": "",
                        "mask": "User",
                        "dataset": "",
                        "conversations": [
                            {
                                "from": "User",
                                "value": combine_context_and_instruction(
                                    context=self.audio_locator,
                                    instruction=generate_canary_instruction(entry)
                                ),
                                "canonical_form": "",
                                "label": None
                            },
                            {
                                "from": "Assistant",
                                "value": entry["answer"],
                                "canonical_form": "",
                                "label": None
                            }
                        ],
                    }
                elif chat_format == "sqa":
                    sft_data = {
                        "system": "",
                        "mask": "User",
                        "dataset": "",
                        "conversations": [
                            {
                                "from": "User",
                                "value": combine_context_and_instruction(
                                    context=self.audio_locator,
                                    instruction=cut.question,
                                ),
                                "canonical_form": "",
                                "label": None,
                            },
                            {
                                "from": "Assistant",
                                "value": cut.answer,
                                "canonical_form": "",
                                "label": None
                            }
                        ],
                    }
                elif chat_format == "chat":
                    # data is already in chat format
                    sft_data = {
                        "system": cut.system,
                        "mask": cut.mask,
                        "dataset": cut.dataset,
                        "conversations": cut.conversations,
                    }
                else:
                    raise NotImplementedError(f"Chat format {chat_format} is not supported.")

                sft_examples.append(
                    GemmaSFTExample(
                        data=sft_data,
                        language=cut.supervisions[0].language
                    ).tokenize(self.tokenizer)
                )

            pad_id = self.tokenizer.pad_id
            collated_text_data = dict(
                tokens=collate_vectors_lhotse([e.input_ids for e in sft_examples], padding_value=pad_id),   # this includes bos and eos
                tokens_length=torch.LongTensor([len(e.input_ids) for e in sft_examples]),
                loss_mask=collate_vectors_lhotse([e.mask for e in sft_examples], padding_value=0),  # this includes bos and eos
                contexts=collate_vectors_lhotse([e.context_ids for e in sft_examples], padding_value=pad_id),   # this includes bos
                context_lengths=torch.LongTensor([len(e.context_ids) for e in sft_examples]),
                answers=collate_vectors_lhotse([e.answer_ids for e in sft_examples], padding_value=pad_id),   # this includes eos
            )
            return_batch.update(
                {
                    "sample_ids": list(cuts.ids),
                    "audio_signal": audio,
                    "audio_signal_length": audio_lens,
                    "audio_ratio": torch.FloatTensor(audio_ratio),
                    "metadata": metadata,
                    "audio_locator_ids": torch.LongTensor(self.audio_locator_ids),
                    **collated_text_data,
                }
            )
            ans.update(return_batch)

        # convert text examples to tensors
        text_examples = all_cuts.filter(lambda c: isinstance(c, (SourceTargetTextExample, NeMoSFTExample)))
        if text_examples:
            pad_id = self.tokenizer.pad_id
            text_minibatch = dict(
                text_input_ids=collate_vectors_lhotse([e.input_ids for e in text_examples], padding_value=pad_id),
                text_answer_ids=collate_vectors_lhotse([e.answer_ids for e in text_examples], padding_value=pad_id),
                text_context_ids=collate_vectors_lhotse([e.context_ids for e in text_examples], padding_value=pad_id),
                text_masks=collate_vectors_lhotse([e.mask for e in text_examples], padding_value=0),
            )
            ans.update(text_minibatch)

        return ans


def collate_text_data(
    cuts,
    default_context: str,
    text_processor: TextProcessing,
    tokens_to_generate: int,
    pad_to_max_length: bool,
    max_seq_length: int,
) -> dict:
    """Perform text collation equivalent to nemo/collections/multimodal/data/audio_text_qa_dataset.py:121"""
    batch_size = len(cuts)
    pad_id = text_processor.pad_id
    examples = [
        {
            k: torch.as_tensor(v)
            for k, v in text_processor._process_example(
                context=cut.context,
                output=cut.supervisions[0].text,
            ).items()
        }
        for cut in cuts
    ]
    fields = as_dict(examples)

    def get_max_len(input_list):
        return max([len(x) for x in input_list])

    max_length = tokens_to_generate + max(
        get_max_len(fields["input_ids"]), get_max_len(fields["context_ids"]), get_max_len(fields["answer_ids"])
    )
    # increase max length to nearest multiple of 4 or 8
    if pad_to_max_length:
        max_length = max_seq_length
    else:
        max_length = min(max_seq_length, ceil_to_nearest(max_length, 8))

    all_tokens = collate_vectors(fields["input_ids"], max_length=max_length, padding_value=pad_id)
    full_lengths = torch.LongTensor([len(item) for item in fields["input_ids"]])

    assert max_length <= max_seq_length, f"{max_length=} <= {max_seq_length=}"

    return {
        "tokens": all_tokens[:, :-1],
        "tokens_length": full_lengths - 1,
        "labels": all_tokens[:, 1:],
        "loss_mask": collate_vectors(
            [torch.as_tensor(build_loss_mask(item)) for item in examples], max_length=max_length, padding_value=0
        )[:, 1:],
        "position_ids": torch.arange(max_length, dtype=torch.long).repeat(batch_size, 1),
        "contexts": collate_vectors(fields["context_ids"], max_length=max_length, padding_value=pad_id),
        "context_lengths": torch.LongTensor([len(seq) for seq in fields["context_ids"]]),
        "answers": collate_vectors(fields["answer_ids"], max_length=max_length, padding_value=pad_id),
        "max_length": torch.LongTensor([max_length] * batch_size),
    }


def as_dict(arg: list[dict]) -> dict[str, list]:
    return {k: [item[k] for item in arg] for k in arg[0].keys()}
