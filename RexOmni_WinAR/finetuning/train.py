import json
import logging
import os
import pathlib
import shutil
import sys
from pathlib import Path

import torch
import transformers

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
import os.path as osp

from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl
from mmengine.config import Config
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer

from engine.argument import DataArguments, ModelArguments, TrainingArguments
from engine.registry import BUILDER
from engine.trainer import replace_qwen2_vl_attention_class
from engine.utils import is_main_process, setup_logger_and_init_log

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    ############################################################
    #                    Step1: Parse Args                     #
    ############################################################
    if is_main_process():
        print(f">>>>>>> Step1: Parse Args <<<<<<<")
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    cfg = Config.fromfile(training_args.config)

    # merge args from cfg to model_args, data_args
    cfg_dict = cfg._cfg_dict
    for key, value in cfg_dict.items():
        if hasattr(model_args, key):
            setattr(model_args, key, value)
        elif hasattr(data_args, key):
            setattr(data_args, key, value)
    if training_args.output_dir is None:
        training_args.output_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(training_args.config))[0]
        )
    # merge training_args to cfg
    for k, v in training_args.__dict__.items():
        if v is not None:
            cfg[k] = v

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    if is_main_process():
        logger = setup_logger_and_init_log(training_args.output_dir, local_rank)

    ############################################################
    #                    Step2: build model                    #
    ############################################################
    if is_main_process():
        print(f">>>>>>> Step2: build model <<<<<<<")

    apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )

    data_args.image_processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    ).image_processor
    data_args.model_type = "qwen2.5vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    # print all the trainable parameters and the amount of them
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main_process():
        logger.info("number of training params:" + str(n_parameters))
        logger.info(
            "Training params:\n"
            + json.dumps(
                {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
                indent=2,
            )
        )
        # log the model structure
        logger.info("Model structure:")
        logger.info(model)

    ############################################################
    #                  Step3: build dataset                    #
    ############################################################
    if is_main_process():
        print(f">>>>>>> Step3: build dataset <<<<<<<")
    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    cfg.train_dataset.data_args = data_args
    cfg.train_dataset.tokenizer = tokenizer
    cfg.data_collator.tokenizer = tokenizer
    train_dataset = BUILDER.build(cfg.train_dataset)
    # log the size of the dataset
    if is_main_process():
        logger.info("Training dataset size: " + str(len(train_dataset)))
    eval_dataset = None
    data_collator = BUILDER.build(cfg.data_collator)
    data_module = dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    ############################################################
    #                  Step4: build trainer                    #
    ############################################################
    if is_main_process():
        print(f">>>>>>> Step4: Start Training <<<<<<<")

    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    shutil.copy2(source_path, template_path)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
