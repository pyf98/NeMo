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


from pathlib import Path

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from nemo.collections.multimodal.speech_llm.models.modular_models import ModularAudioGPTModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.utils import logging

mp.set_start_method("spawn", force=True)


# alm_ckpt = "/workspace/results/yifanp_speechllm_subsampled_chat_N8/gemma_chat_enc-frozen_llm-ft_joint_text1.0_mbs16_gbs1024/gemma_chat_enc-frozen_llm-ft_joint_text1.0_mbs16_gbs1024/checkpoints/gemma_chat_enc-frozen_llm-ft_joint_text1.0_mbs16_gbs1024--validation_bleu=0.000-step=50000-epoch=1-last.ckpt"
# alm_ckpt = "/workspace/results/stage2/checkpoint/gemma_chat_stage2_enc-frozen_llm-ft_joint_text1.0_lr2e-5_max2k_ultrachat_mbs16_gbs2048--validation_bleu=39.011-step=2000-epoch=0-last.ckpt"
alm_ckpt = "/workspace/results/yifanp_speechllm_subsampled_chat_N8/gemma_chat_gemma2b-it_enc-frozen_llm-lora_speech_mbs1_gbs64/checkpoints/fixed.ckpt"
alm_yaml = Path(alm_ckpt).parent.parent / "version_0/hparams.yaml"
test_cfg = "/data/data_cfg/test_cfg.yaml"

base_cfg = OmegaConf.load("/workspace/code/NeMo/examples/multimodal/speech_llm/conf/modular_audio_gpt_config_eval.yaml")
new_cfg = OmegaConf.create(
    {
        "model": {
            "restore_from_path": '/workspace/pretrained/HF_TO_NEMO/gemma-2b-it',
            "pretrained_audio_model": '/workspace/pretrained/canary-1b.nemo',
            "peft": {
                "restore_from_path": alm_ckpt,
                "restore_from_hparams_path": alm_yaml,
            },
            "data": {
                "test_ds": {
                    "input_cfg": test_cfg,
                    "use_multimodal_sampling": False,    # for text data
                    "global_batch_size": 1,
                    "micro_batch_size": 1,
                    "tokens_to_generate": 512,
                    "batch_size": 1,
                }
            },
            "global_batch_size": 1,
            "micro_batch_size": 1,
        },
        "inference": {
            "greedy": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.4,
            "repetition_penalty": 1.2,
        },
    }
)
cfg = OmegaConf.merge(base_cfg, new_cfg)
# logging.info("\n\n************** Experiment configuration ***********")
# logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
# logging.info("**************************************************\n\n")

trainer = MegatronTrainerBuilder(cfg).create_trainer()

if cfg.model.from_pretrained:
    # Load model from NGC or HuggingFace
    logging.info(f"Loading model from cloud: {cfg.model.from_pretrained}")
    model_cfg = ModularAudioGPTModel.from_pretrained(
        cfg.model.from_pretrained, trainer=trainer, return_config=True
    )
    model_cfg = ModularAudioGPTModel.merge_inference_cfg(cfg, trainer, model_cfg)
    model_file = ModularAudioGPTModel.from_pretrained(
        cfg.model.from_pretrained, trainer=trainer, return_model_file=True
    )
    model = ModularAudioGPTModel.restore_from(
        restore_path=model_file,
        trainer=trainer,
        override_config_path=model_cfg,
        strict=False,
        map_location="cpu",
    )
    if "peft" in model_cfg and model_cfg.peft.get("peft_scheme", None):
        # need this due to the way that MegatronGPTSFTModel doesn't load adapters in model initialization
        model.load_adapters(model_file, map_location="cpu")
else:
    # Load model from a local file
    model_cfg = ModularAudioGPTModel.merge_inference_cfg(cfg, trainer)
    model = ModularAudioGPTModel.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        override_config_path=model_cfg,
        strict=False,
        map_location="cpu",
    )
    model = ModularAudioGPTModel.load_adapters_for_inference(cfg, model_cfg, model)
    model = ModularAudioGPTModel.load_audio_encoder_for_inference(cfg, model_cfg, model)

model.freeze()
if cfg.get("save_as_nemo", None):
    model.setup("predict")  # need to call setup() to load adapters and prepare for saving
    model.save_to(cfg.save_as_nemo)
    logging.info(f"Model saved to {Path(cfg.save_as_nemo).absolute()}, exiting...")
    exit(0)

if not cfg.model.get('use_flash_attention', False):
    cfg.inference.compute_attention_mask = True


# model.setup("predict")
# import torch
# sd = torch.load(alm_ckpt)['state_dict']
# for k, v in sd.items():
#     if k not in model.state_dict():
#         print(f"--- {k} is missing in model")
#     elif not torch.allclose(v.cpu(), model.state_dict()[k].cpu()):
#         print(f"+++ {k} mismatch")

# breakpoint()


# run inference
cfg.inference.greedy = True
cfg.inference.temperature = 0.4
config = OmegaConf.to_container(cfg.inference, resolve=True)
model.set_inference_config(config)

ds = model._build_dataset(cfg.model.data.test_ds, is_train=False)
dl = model.build_data_loader(ds, cfg.model.data.test_ds, consumed_samples=0, is_predict=True, is_eval=False)
response = trainer.predict(model, dl)
print(response[-1]['sentences'][0])
