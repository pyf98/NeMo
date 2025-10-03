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

import torch
from lightning.pytorch import Trainer
from omegaconf import OmegaConf, open_dict

from nemo.collections.speechlm2 import DataModule, SALMDataset
from nemo.collections.speechlm2.models.salm_asr_decoder import SALMWithAsrDecoder
from nemo.core.config import hydra_runner
from nemo.utils.trainer_utils import resolve_trainer_cfg

torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

@hydra_runner(config_path="conf", config_name="salm")
def main(cfg):
    OmegaConf.resolve(cfg)
    torch.distributed.init_process_group(backend="nccl")
    torch.set_float32_matmul_precision("medium")
    trainer = Trainer(**resolve_trainer_cfg(cfg.trainer))

    if os.path.isdir(cfg.pretrained_name):
        # HF format
        model = SALMWithAsrDecoder.from_pretrained(cfg.pretrained_name)
    else:
        # PyTorch Lightning format
        model = SALMWithAsrDecoder.load_from_checkpoint(cfg.pretrained_name, map_location="cpu")

    # Assign additional arguments to the model config
    model.cfg.update(cfg.model)
    # Fill in missing arguments in dataset config
    with open_dict(cfg.data.test_ds): # must open_dict to avoid errors
        cfg.data.test_ds.prompt_format = model.cfg.prompt_format
        cfg.data.test_ds.audio_locator_tag = model.cfg.audio_locator_tag

    dataset = SALMDataset(tokenizer=model.tokenizer)
    datamodule = DataModule(cfg.data, tokenizer=model.tokenizer, dataset=dataset)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
