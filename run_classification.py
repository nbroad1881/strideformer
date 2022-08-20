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
""" Finetuning the sbert chunks for long sequence classification"""

import logging
import os
import math
import logging
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
    Trainer,
    TrainingArguments,
    set_seed,
    get_scheduler,
)
from transformers.trainer_pt_utils import IterableDatasetShard
from accelerate import Accelerator

from utils import set_wandb_env_vars, set_mlflow_env_vars
from model import StridedLongformer
from data import DataModule, StridedLongformerCollator


logger = logging.getLogger(__name__)


def tokenize(example, tokenizer, max_length, stride):

    tokenized_example = tokenizer(
        # example["claim"],
        # example["main_text"],
        example["text"],
        padding=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
    )

    tokenized_example["label"] = example["label"]
    tokenized_example["length"] = len(tokenized_example["input_ids"])

    return tokenized_example


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

            outputs = model(**batch)
            loss = outputs["loss"]

            example_count += batch["labels"].numel()

            # We keep track of the loss at each epoch
            if args.report_to is not None:
                epoch_loss += loss.detach().float()

            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(dataloaders["train"]) - 1
            ):
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if args.save_strategy == "steps":
                checkpointing_steps = args.save_steps
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if args.logging_strategy == "steps" and step % args.logging_steps == 0:
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

        preds = outputs["logits"].argmax(-1).detach().cpu().tolist()
        labels = batch["labels"].detach().cpu().tolist()
        count += len(labels)

        y_preds.append(preds)
        y_true.append(labels)
        
        progress_bar.update(1)

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

    len_dataloader = len(dataloader)
    num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

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

    optim_cls, optim_args = Trainer.get_optimizer_cls_and_kwargs(args)

    optimizer = optim_cls(model.parameters(), **optim_args)

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
    del cfg.training_arguments

    training_args = TrainingArguments(**train_args)

    mixed_precision = "no"
    if training_args.fp16:
        mixed_precision = "fp16"
    if training_args.bf16:
        mixed_precision = "bf16"

    accelerator = Accelerator(
        mixed_precision=mixed_precision, log_with=training_args.report_to
    )

    if "wandb" in training_args.report_to:
        set_wandb_env_vars(cfg)
    if "mlflow" in training_args.report_to:
        set_mlflow_env_vars(cfg)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
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

    collator = StridedLongformerCollator(tokenizer=data_module.tokenizer)

    dataloaders = {}

    dataloaders["train"] = DataLoader(
        data_module.get_train_dataset(),
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=collator,
        drop_last=training_args.dataloader_drop_last,
        num_workers=training_args.dataloader_num_workers,
        pin_memory=training_args.dataloader_pin_memory,
    )

    if training_args.do_eval:

        dataloaders["validation"] = DataLoader(
            data_module.get_eval_dataset(),
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=collator,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )

    if training_args.do_predict:

        dataloaders["test"] = DataLoader(
            data_module.get_test_dataset(),
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=collator,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )

    # Start with the config for the sbert model
    model_config = AutoConfig.from_pretrained(cfg.model.model_name_or_path)
    model_config.update(
        {
            "sbert_model": cfg.model.model_name_or_path,
            "num_labels": len(data_module.label2id),
            "label2id": data_module.label2id,
            "id2label": data_module.id2label,
            "hidden_act": cfg.model.hidden_act,
            "intermediate_size": cfg.model.intermediate_size,
            "layer_norm_eps": cfg.model.layer_norm_eps,
            "num_attention_heads": cfg.model.num_attention_heads,
            "num_hidden_layers": cfg.model.num_hidden_layers,
        }
    )

    model = StridedLongformer.from_pretrained(
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

    model, optimizer, lr_scheduler, *dataloaders_list = accelerator.prepare(*items2prepare)
    
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

            accelerator.init_trackers(cfg.project_name, experiment_config)

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