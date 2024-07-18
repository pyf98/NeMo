import random
import json
from pathlib import Path


AUDIO_LOCATOR = "<unused0>"

CODE2LANG = {
    "fr": 'French',
    "es": 'Spanish',
    "de": 'German',
    "en": 'English',
}

CANARY_TEMPLATES = {
    "asr": "Transcribe the content to <|SLANG|>, <|PNC|>.",
    "ast": "Translate the <|SLANG|> content to <|TLANG|>, <|PNC|>.",
}
for t in ["transcribe"]:
    CANARY_TEMPLATES[t] = CANARY_TEMPLATES["asr"]
for t in ["s2t_translation", "translate"]:
    CANARY_TEMPLATES[t] = CANARY_TEMPLATES["ast"]

PNC = {
    "yes": 'with punctuations and capitalizations',
    "no": 'ignoring punctuations and capitalizations',
}


def generate_canary_instruction(entry: dict):
    task = CANARY_TEMPLATES[entry['taskname']]
    task = task.replace('<|SLANG|>', CODE2LANG[entry['source_lang']])
    task = task.replace('<|TLANG|>', CODE2LANG[entry['target_lang']])
    task = task.replace('<|PNC|>', PNC[entry['pnc']])
    return task


def combine_context_and_instruction(context: str, instruction: str):
    choices = [
        f"Given a context:\n{context}\nPerform this task:\n{instruction}",
        f"Follow the instruction below:\n{instruction}\nYou are given a context:\n{context}",
        f"Here is a sentence:\n{context}\nYou need to follow:\n{instruction}",
        f"With the context in mind:\n{context}\nGenerate responses for the instruction:\n{instruction}",
        f"You need to generate responses to the instruction:\n{instruction}\nYou can use information below:\n{context}",
        f"{context}\nFollow the instruction below based on the previous sentence:\n{instruction}",
        f"With the following information:\n{context}\nHow do you respond to the instruction?\n{instruction}",
        f"Based on the following information:\n{context}\nCan you respond to the instruction?\n{instruction}",
        f"Can you follow this?\n{instruction}\nYou can find information from:\n{context}",
        f"You are given some contextual information:\n{context}\nCan you generate a response?\n{instruction}",
        f"{instruction}\nHere is the context:\n{context}",
        f"{instruction}\nYou may utilize this text:\n{context}",
        f"{context}\nCan you perform the following task based on the previous sentence?\n{instruction}",
        f"Here is your task:\n{instruction}\nYou may find relevant information below:\n{context}",
        f"Based on this sentence:\n{context}\nCan you respond to the prompt?\n{instruction}",
        f"{context}\n{instruction}",
        f"{instruction}\n{context}",
        f"{context} {instruction}",
        f"{instruction} {context}",
    ]
    return random.choice(choices)


def process_manifest_sqa(fin, fout, dataset, label_key="answer"):
    """Process SpeechQA data. Each sample has 'question' and 'answer' fields."""

    Path(fout).parent.mkdir(parents=True, exist_ok=True)

    with open(fin, 'r', encoding='utf-8') as fp_in, open(
        fout, 'w', encoding='utf-8'
    ) as fp_out:
        for line in fp_in:
            entry = json.loads(line)
            entry.update(
                {
                    "system": "",
                    "mask": "User",
                    "dataset": dataset,
                    "conversations": [
                        {"from": "User", "value": combine_context_and_instruction(context=AUDIO_LOCATOR, instruction=entry['question']), "canonical_form": "", "label": None},
                        {"from": "Assistant", "value": entry[label_key], "canonical_form": "", "label": None}
                    ]
                }
            )
            fp_out.write(json.dumps(entry) + "\n")


def process_manifest_canary(fin, fout, dataset, label_key="answer"):
    Path(fout).parent.mkdir(parents=True, exist_ok=True)
    with open(fin, 'r', encoding='utf-8') as fp_in, open(
        fout, 'w', encoding='utf-8'
    ) as fp_out:
        for line in fp_in:
            entry = json.loads(line)
            entry.update(
                {
                    "system": "",
                    "mask": "User",
                    "dataset": dataset,
                    "conversations": [
                        {
                            "from": "User",
                            "value": combine_context_and_instruction(
                                context=AUDIO_LOCATOR,
                                instruction=generate_canary_instruction(entry)
                            ),
                            "canonical_form": "",
                            "label": None
                        },
                        {"from": "Assistant", "value": entry[label_key], "canonical_form": "", "label": None}
                    ]
                }
            )
            fp_out.write(json.dumps(entry) + "\n")
