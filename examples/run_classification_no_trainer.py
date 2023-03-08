#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
""" Finetuning short model chunks for long sequence classification"""

import logging
import os
import math
from itertools import chain

from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    set_seed,
    get_scheduler,
    DataCollatorWithPadding,
)
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.utils import flatten_dict
from accelerate import Accelerator
from accelerate.logging import get_logger

from utils import (
    set_wandb_env_vars,
    set_mlflow_env_vars,
)
from strideformer import Strideformer, StrideformerConfig, StrideformerCollator
from data import DataModule


logger = get_logger(__name__)


def training_loop(accelerator, model, optimizer, lr_scheduler, dataloaders, args):

    progress_bar = tqdm(
        range(args.max_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )
    completed_steps = 0

    for epoch in range(args.num_train_epochs):

        model.train()

        if args.report_to is not None:
            epoch_loss = 0

        example_count = 0
        for step, batch in enumerate(dataloaders["train"]):

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs["loss"]

                example_count += batch["labels"].numel()

                # We keep track of the loss at each epoch
                if args.report_to is not None:
                    epoch_loss += loss.detach().float() * batch["labels"].numel()

                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                if (step+1) % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                    completed_steps += 1

            if args.save_strategy == "steps":
                checkpointing_steps = args.save_steps
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if (
                args.logging_strategy == "steps"
                and (step + 1) % args.logging_steps == 0
            ):
                details = {
                    "train_loss": epoch_loss.item() / example_count,
                    "step": completed_steps,
                }
                accelerator.log(
                    details,
                    step=completed_steps,
                )
                logging.info(details)
                progress_bar.set_description(
                    f"Training: Epoch {epoch} | Loss {details['train_loss']} | Step {details['step']}"
                )

            if completed_steps >= args.max_steps:
                break

        if args.do_eval:

            eval_metrics = eval_loop(
                accelerator=accelerator,
                model=model,
                dataloader=dataloaders["validation"],
                prefix="eval",
            )

            eval_metrics.update(
                {
                    "epoch": epoch,
                    "step": completed_steps,
                }
            )

            accelerator.log(eval_metrics, step=completed_steps)
            logging.info(eval_metrics)

        if completed_steps >= args.max_steps:
            break

    return completed_steps


@torch.no_grad()
def eval_loop(accelerator, model, dataloader, prefix):

    model.eval()

    y_preds = []
    y_true = []
    eval_loss = 0
    count = 0

    num_steps = len(dataloader)

    progress_bar = tqdm(
        range(num_steps),
        disable=not accelerator.is_local_main_process,
        desc=f"{prefix}",
    )

    for batch in dataloader:

        outputs = model(**batch)

        eval_loss += outputs["loss"].detach().float()

        # TODO: gather for distributed training

        preds = outputs["logits"].argmax(-1).detach().cpu().tolist()
        labels = batch["labels"].detach().cpu().tolist()
        count += len(labels)

        y_preds.append(preds)
        y_true.append(labels)

        progress_bar.update(1)

    if isinstance(y_preds[0], list):
        y_preds = list(chain(*y_preds))
    y_true = list(chain(*y_true))

    f1_micro = f1_score(y_true, y_preds, average="micro")
    f1_macro = f1_score(y_true, y_preds, average="macro")

    metrics = {
        "loss": eval_loss.item() / count,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }

    metrics_desc = " | ".join(
        [f"{name.capitalize()} {round(score, 4)}" for name, score in metrics.items()]
    )

    progress_bar.set_description(f"{prefix}: {metrics_desc}")

    for name in metrics.keys():
        if not name.startswith(prefix):
            metrics[f"{prefix}_{name}"] = metrics.pop(name)

    return metrics


def num_examples(dataloader: DataLoader) -> int:
    """
    Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
    dataloader.dataset does not exist or has no length, estimates as best it can
    """
    dataset = dataloader.dataset
    # Special case for IterableDatasetShard, we need to dig deeper
    if isinstance(dataset, IterableDatasetShard):
        return len(dataloader.dataset.dataset)
    return len(dataloader.dataset)


def train_samples_steps_epochs(dataloader, args):
    # TODO: make use of num_train_samples and num_train_epochs

    total_train_batch_size = (
        args.train_batch_size * args.gradient_accumulation_steps * args.world_size
    )

    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)

    if args.max_steps > 0:
        max_steps = args.max_steps
        num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
            args.max_steps % num_update_steps_per_epoch > 0
        )
        # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
        # the best we can do.
        num_train_samples = args.max_steps * total_train_batch_size
    else:
        max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        num_train_epochs = math.ceil(args.num_train_epochs)
        num_train_samples = num_examples(dataloader) * args.num_train_epochs

    args.max_steps = max_steps

    return num_train_samples, max_steps, num_train_epochs


def get_optimizer_and_scheduler(model, train_dataloader, args):

    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    num_train_samples, max_steps, num_train_epochs = train_samples_steps_epochs(
        train_dataloader, args
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.get_warmup_steps(max_steps),
        num_training_steps=max_steps,
    )

    return optimizer, lr_scheduler


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    train_args = dict(cfg.training_arguments)
    training_args = TrainingArguments(**train_args)

    mixed_precision = "no"
    if training_args.fp16:
        mixed_precision = "fp16"
    if training_args.bf16:
        mixed_precision = "bf16"

    if "wandb" in training_args.report_to:
        set_wandb_env_vars(cfg)
    if "mlflow" in training_args.report_to:
        set_mlflow_env_vars(cfg)

    accelerator = Accelerator(
        mixed_precision=mixed_precision, 
        log_with=training_args.report_to,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        )

    log_level = training_args.get_process_log_level()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )
    logger.info(accelerator.state, main_process_only=False)

    
    if log_level != -1:
        logger.setLevel(log_level)

    if accelerator.is_local_main_process:
        if log_level == -1:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity(log_level)
            transformers.utils.logging.set_verbosity(log_level)
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_module = DataModule(cfg)

    # If running distributed, this will do something on the main process, while
    #    blocking replicas, and when it's finished, it will release the replicas.
    with training_args.main_process_first(desc="Dataset loading and tokenization"):
        data_module.prepare_dataset()


    use_strideformer = cfg.data.stride is not None and cfg.data.stride > 0

    if use_strideformer:
        collator = StrideformerCollator(tokenizer=data_module.tokenizer)

        # batch_sizes must always be 1 when using strided approach
        if (
            training_args.per_device_train_batch_size != 1
            or training_args.per_device_eval_batch_size != 1
        ):
            logger.warning(
                "Batch size must be 1 when using strided approach. Changing to 1 now."
            )
        training_args.per_device_train_batch_size = 1
        training_args.per_device_eval_batch_size = 1

    else:
        collator = DataCollatorWithPadding(
            tokenizer=data_module.tokenizer, pad_to_multiple_of=cfg.data.pad_multiple
        )

    dataloaders = {}

    dataloaders["train"] = DataLoader(
        data_module.get_dataset("train"),
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=collator,
        drop_last=training_args.dataloader_drop_last,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
    )

    if training_args.do_eval:

        dataloaders["validation"] = DataLoader(
            data_module.get_dataset("validation"),
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=collator,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )

    if training_args.do_predict:

        dataloaders["test"] = DataLoader(
            data_module.get_dataset("test"),
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=collator,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )


    model_config = AutoConfig.from_pretrained(
            cfg.model.model_name_or_path,
            label2id=data_module.label2id,
            id2label=data_module.id2label,
            num_labels=len(data_module.label2id),
    )
    if use_strideformer:
        second_model_config = dict(
            freeze_first_model=cfg.model.freeze_first_model,
            max_chunks=cfg.model.max_chunks,
            num_hidden_layers=cfg.model.num_hidden_layers,
            num_attention_heads=cfg.model.num_attention_heads,
            intermediate_size=cfg.model.intermediate_size,
            hidden_act=cfg.model.hidden_act,
            dropout=cfg.model.dropout,
            layer_norm_eps=cfg.model.layer_norm_eps,
            initializer_range=cfg.model.initializer_range,
            hidden_size=model_config.hidden_size,
            label2id=data_module.label2id,
            id2label=data_module.id2label,
            num_labels=len(data_module.label2id),
            )
        model_config = StrideformerConfig.from_two_configs(
            first_model_config=model_config,
            second_model_config=second_model_config,
        )
        
    if use_strideformer:
        model = Strideformer(
            config=model_config
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.model_name_or_path, config=model_config
        )

    optimizer, lr_scheduler = get_optimizer_and_scheduler(
        model, dataloaders["train"], training_args
    )

    items2prepare = [
        model,
        optimizer,
        lr_scheduler,
        dataloaders["train"],
    ]

    if training_args.do_eval:
        items2prepare.append(dataloaders["validation"])

    if training_args.do_predict:
        items2prepare.append(dataloaders["test"])

    model, optimizer, lr_scheduler, *dataloaders_list = accelerator.prepare(
        *items2prepare
    )

    dataloaders["train"] = dataloaders_list.pop(0)

    if training_args.do_eval:
        dataloaders["validation"] = dataloaders_list.pop(0)

    if training_args.do_predict:
        dataloaders["test"] = dataloaders_list.pop(0)

    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if training_args.report_to:
        if accelerator.is_main_process:
            experiment_config = OmegaConf.to_container(cfg)
            experiment_config.update(model.config.to_diff_dict())
                
            accelerator.init_trackers(cfg.project_name, flatten_dict(experiment_config))
            

    completed_steps = training_loop(
        accelerator,
        model,
        optimizer,
        lr_scheduler,
        dataloaders,
        training_args,
    )

    if training_args.do_predict:

        test_metrics = eval_loop(
            accelerator=accelerator,
            model=model,
            dataloader=dataloaders["test"],
            prefix="test",
        )

        accelerator.log(test_metrics, step=completed_steps)

        logging.info(test_metrics)

    accelerator.end_training()


if __name__ == "__main__":
    main()
