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
# pylint: disable=missing-function-docstring,missing-class-docstring
from itertools import groupby
from lhotse.cut import Cut, MixedCut
import torch
from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter
from nemo.collections.common.data.lhotse.text_adapters import NeMoMultimodalConversation, TextTurn


SYSTEM_BOS = "<SPECIAL_10>"
TURN_BOS = "<SPECIAL_11>"
OUTPUT_EOS = "<SPECIAL_12>\n"


class NemotronHPromptFormatter(PromptFormatter):
    NAME = "nemotron-h"
    OUTPUT_ROLE = "assistant"
    INFERENCE_PREFIX = f"{TURN_BOS}Assistant\n"
    THINK_PREFIX = "<think>\n"
    NO_THINK_PREFIX = "<think></think>"
    TEMPLATE = {
        "system": {
            "template": f"{SYSTEM_BOS}System\n|message|\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        "user": {
            "template": f"{TURN_BOS}User\n|message|\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"{INFERENCE_PREFIX}|message|\n{OUTPUT_EOS}",
            "slots": {
                "message": Modality.Text,
            },
        },
    }

    def encode_dialog(self, turns: list[dict]) -> dict[str, torch.Tensor]:
        """Overrides the base class method with following changes:

        - Set loss mask as True only for the last assistant turn.
        - Remove thinking content from previous turns.
        - Add an empty system turn even if it's not present.
        - Add thinking prefix for inference based on "/think" and "/no_think" in user or system turns.

        Reference: https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2#prompt-format

        Args:
            turns (list[dict]): List of turns. Each turn is a dict with "role" and "slots" keys.

        """

        roles = self.get_roles()
        assert len(turns) > 0, "Empty dialog is not supported."
        for turn in turns:
            assert "role" in turn, f"A turn must have have a 'role' key. We received {turn=}"
            assert turn["role"] in roles, f"Found turn with {turn['role']=}, but available roles are {roles}"

        # Preprocess turns based on Nemotron-Nano-9B-v2 prompt format
        # 0) Unify the format of turns to have "role" and "slots" keys.
        for turn in turns:
            if "content" in turn:
                turn["slots"] = {"message": turn.pop("content")}

        # 1) Add an empty system turn even if it's not present. This is for both training and inference.
        if turns[0]["role"] != "system":
            turns.insert(0, {"role": "system", "slots": {"message": ""}})

        # 2) Determine if thinking is enabled in user or system turns.
        # If multiple turns have the tag, we will use the last one.
        # This is for inference only.
        enable_thinking = True
        for turn in turns:
            if turn["role"] == "user" or turn["role"] == "system":
                if "/think" in turn["slots"]["message"]:
                    enable_thinking = True
                elif "/no_think" in turn["slots"]["message"]:
                    enable_thinking = False
                turn["slots"]["message"] = turn["slots"]["message"].replace("/think", "").replace("/no_think", "").strip()

        # 3) Remove thinking content from previous turns. This is for both training and inference.
        for turn in turns[:-1]:
            if turn["role"] == self.OUTPUT_ROLE:
                if "</think>" in turn["slots"]["message"]:
                    turn["slots"]["message"] = turn["slots"]["message"].split("</think>")[1].strip()

        # 4) Add empty thinking content to the last assistant turn if not present. This is for training only.
        turn = turns[-1]
        if turn["role"] == self.OUTPUT_ROLE:
            if "<think>" not in turn["slots"]["message"]:
                turn["slots"]["message"] = self.NO_THINK_PREFIX + turn["slots"]["message"]
            else:
                assert turn["slots"]["message"].startswith("<think>"), turn["slots"]["message"]
                assert "</think>" in turn["slots"]["message"], turn["slots"]["message"]


        turn_tokens = []
        turn_token_counts = []
        turn_mask_values = []

        if self.INSERT_BOS:
            turn_tokens.append(self.tokenizer.bos)
            turn_token_counts.append(1)
            turn_mask_values.append(False)

        if "preamble" in self.TEMPLATE:
            preamble_turns = [idx for idx, t in enumerate(turns) if t["role"] == "preamble"]
            if not preamble_turns:
                turns = [{"role": "preamble", **self.TEMPLATE["preamble"]}] + turns
            else:
                assert (
                    len(preamble_turns) == 1 and preamble_turns[0] == 0
                ), f"Preamble can only be presented at turn 0 but we found preamble turns at indexes {preamble_turns}."

        is_inference = turns[-1]["role"] != self.OUTPUT_ROLE
        for idx, turn in enumerate(turns):
            role = turn["role"]
            expected_slots = self.get_slots(role)
            if "content" in turn and len(expected_slots) == 1:
                # User is leveraging the "standard" API prompting LLM; we'll map "content" value
                # to whatever is the name of the slot, when there's only one slot.
                slot_values = {k: turn["content"] for k in expected_slots.keys()}  # 1-item dict
            else:
                slot_values = turn.get("slots", {})
                if expected_slots:
                    assert slot_values, (
                        f"A turn for role {role} must have have a non-empty value under 'slots' key. "
                        f"We received {turn=}"
                    )
                    self._validate_slot_values(expected_slots, slot_values)
            template = self.get_template(role)
            tokens = self.encode_turn(template, expected_slots, slot_values)
            turn_tokens.extend(tokens)
            turn_token_counts.append(len(tokens))
            # Set loss mask as True only for the last assistant turn.
            turn_mask_values.append(role == self.OUTPUT_ROLE and idx == len(turns) - 1)

        if is_inference and self.INFERENCE_PREFIX is not None:
            inference_prefix_with_thinking = self.INFERENCE_PREFIX
            if enable_thinking:
                inference_prefix_with_thinking = inference_prefix_with_thinking + self.THINK_PREFIX
            else:
                inference_prefix_with_thinking = inference_prefix_with_thinking + self.NO_THINK_PREFIX
            inference_prefix = self._apply_tokenizer(inference_prefix_with_thinking)
            turn_tokens.extend(inference_prefix)
            turn_token_counts.append(len(inference_prefix))
            turn_mask_values.append(False)  # not a training example

        # Insert EOS only when the last turn comes from the OUTPUT_ROLE.
        if self.INSERT_EOS and not is_inference:
            turn_tokens.append(self.tokenizer.eos)
            turn_token_counts[-1] += 1
            turn_mask_values.append(True)

        ans = {"input_ids": torch.tensor(turn_tokens, dtype=torch.long)}
        if turn_mask_values[-1]:
            # The last turn comes from OUTPUT_ROLE, i.e. it's a response from the system.
            # This indicates it's a training example for which we provide context/answer/mask.
            ans["context_ids"] = ans["input_ids"][: -turn_token_counts[-1]]
            ans["answer_ids"] = ans["input_ids"][-turn_token_counts[-1] :]
            ans["mask"] = torch.tensor(
                [
                    turn_mask_values[turn_idx]
                    for turn_idx, turn_len in enumerate(turn_token_counts)
                    for _ in range(turn_len)
                ],
                dtype=torch.bool,
            )
        else:
            ans["context_ids"] = ans["input_ids"]  # context == input for inference
        return ans


# @registered_prompt_format_fn(Cut, NemotronHPromptFormatter)
# def nemotron_h(cut: Cut, prompt: NemotronHPromptFormatter):
#     if isinstance(cut, MixedCut):
#         cut = cut.first_non_padding_cut

#     turns = []

#     system = ""
#     if cut.has_custom("system_prompt"):
#         system = cut.system_prompt
#     turns.append({"role": "system", "content": system})

#     if cut.has_custom("context"):
#         ctx = cut.context
#     else:
#         ctx = ""
#     turns.append({"role": "user", "content": ctx})

#     if (answer := cut.supervisions[0].text) is not None:
#         turns.append({"role": "assistant", "content": answer})

#     return prompt.encode_dialog(turns)
