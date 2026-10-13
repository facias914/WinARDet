# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
from typing import Optional

import torch
from mmengine import Config
from torch.utils.data import ConcatDataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from engine.registry import BUILDER

from ..utils.dataset import collate_fn
from .config import DataConfig


def create_dataloader(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin],
) -> None:
    config_path = config.config_path
    mmconfig = Config.fromfile(config_path)
    train_datasets = []
    for dataset in mmconfig.train_dataset:
        dataset.tokenizer = tokenizer
        dataset.processor = processor
        train_dataset = BUILDER.build(dataset)
        train_datasets.append(train_dataset)
    train_dataset = ConcatDataset(train_datasets)
    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(
            data_source=train_dataset, generator=train_dataloader_generator
        )
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.rollout_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    print(f"Size of train dataloader: {len(train_dataloader)}")
    return train_dataloader, None
