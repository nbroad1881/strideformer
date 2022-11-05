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
""" Breaking long documents into short chunks for long sequence classification"""

import logging
import os
import math

from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    get_scheduler,
    DataCollatorWithPadding,
)
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.training_args import trainer_log_levels
from transformers.utils import flatten_dict
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
import evaluate

from utils import (
    MLflowTracker,
    set_wandb_env_vars,
    set_mlflow_env_vars,
)
from strideformer import Strideformer, StrideformerConfig, StrideformerCollator
from data import DataModule


logger = get_logger(__name__)


def set_logging_verbosity(accelerator, training_args):

    log_level = trainer_log_levels[training_args.log_level]

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


def training_loop(
    accelerator, model, optimizer, lr_scheduler, dataloaders, args, metrics
):

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
                accelerator.backward(loss)
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

                break
                if completed_steps >= args.max_steps:
                    break

        if args.do_eval:

            eval_metrics = eval_loop(
                accelerator=accelerator,
                model=model,
                dataloader=dataloaders["validation"],
                prefix="eval",
                metrics=metrics,
            )

            eval_metrics.update(
                {
                    "epoch": epoch,
                    "step": completed_steps,
                }
            )

            accelerator.log(eval_metrics, step=completed_steps)
            logging.info(eval_metrics)

    return completed_steps


@torch.no_grad()
def eval_loop(accelerator, model, dataloader, prefix, metrics):

    model.eval()

    eval_loss = 0

    num_steps = len(dataloader)

    progress_bar = tqdm(
        range(num_steps),
        disable=not accelerator.is_local_main_process,
        desc=f"{prefix}",
    )

    for batch in dataloader:

        outputs = model(**batch)

        eval_loss += accelerator.gather(outputs["loss"]).detach().float()

        preds, labels = accelerator.gather_for_metrics(
            (outputs["logits"].argmax(-1), batch["labels"])
        )
        metrics.add_batch(predictions=preds, references=labels)

        progress_bar.update(1)

    metric_results = metrics.compute()

    metrics_desc = " | ".join(
        [
            f"{name.capitalize()} {round(score, 4)}"
            for name, score in metric_results.items()
        ]
    )

    progress_bar.set_description(f"{prefix}: {metrics_desc}")

    for name in metric_results.keys():
        if not name.startswith(prefix):
            metric_results[f"{prefix}_{name}"] = metric_results.pop(name)

    return metric_results


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
    training_args = TrainingArguments(**train_args)

    mixed_precision = "no"
    if training_args.fp16:
        mixed_precision = "fp16"
    if training_args.bf16:
        mixed_precision = "bf16"

    log_with = training_args.report_to
    if "wandb" in training_args.report_to:
        set_wandb_env_vars(cfg)
    if "mlflow" in training_args.report_to:
        set_mlflow_env_vars(cfg)
        log_with.remove("mlflow")
        mlflow_tracker = MLflowTracker(
            experiment_name=cfg.mlflow.experiment_name,
            logging_dir=cfg.mlflow.logging_dir or cfg.training_arguments.output_dir,
            run_id=cfg.mlflow.run_id,
            tags=cfg.mlflow.tags,
            nested_run=cfg.mlflow.nested_run,
            run_name=cfg.mlflow.run_name,
            description=cfg.mlflow.description,
        )
        log_with.append(mlflow_tracker)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        log_with=log_with,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    set_logging_verbosity(accelerator, training_args)

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

    USE_STRIDEFORMER = cfg.data.stride is not None

    if USE_STRIDEFORMER:
        collator = StrideformerCollator(
            tokenizer=data_module.tokenizer, max_chunks=cfg.model.max_chunks
        )

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

    if USE_STRIDEFORMER:
        model_config = StrideformerConfig(
            first_model_name_or_path=cfg.model.first_model_name_or_path,
            freeze_first_model=cfg.model.freeze_first_model,
            max_chunks=cfg.model.max_chunks,
            hidden_size=cfg.model.hidden_size,
            num_hidden_layers=cfg.model.num_hidden_layers,
            num_attention_heads=cfg.model.num_attention_heads,
            intermediate_size=cfg.model.intermediate_size,
            hidden_act=cfg.model.hidden_act,
            dropout=cfg.model.dropout,
            layer_norm_eps=cfg.model.layer_norm_eps,
            num_labels=len(data_module.label2id),
            label2id=data_module.label2id,
            id2label=data_module.id2label,
        )
    else:
        model_config = AutoConfig.from_pretrained(
            cfg.model.model_name_or_path,
            num_labels=len(data_module.label2id),
            label2id=data_module.label2id,
            id2label=data_module.id2label,
        )

    if USE_STRIDEFORMER:
        model = Strideformer(config=model_config, first_init=True)
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

            experiment_config.update(model_config.to_diff_dict())

            accelerator.init_trackers(cfg.project_name, flatten_dict(experiment_config))

    metrics = evaluate.load("accuracy")

    completed_steps = training_loop(
        accelerator,
        model,
        optimizer,
        lr_scheduler,
        dataloaders,
        training_args,
        metrics,
    )

    if training_args.do_predict:

        test_metrics = eval_loop(
            accelerator=accelerator,
            model=model,
            dataloader=dataloaders["test"],
            prefix="test",
            metrics=metrics,
        )

        accelerator.log(test_metrics, step=completed_steps)

        logging.info(test_metrics)

    accelerator.end_training()


if __name__ == "__main__":
    main()
