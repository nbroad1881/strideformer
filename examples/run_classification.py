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
import sys

import hydra
import numpy as np
from sklearn.metrics import f1_score
from omegaconf import DictConfig
import datasets
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    DataCollatorWithPadding,
)

from strideformer import Strideformer, StrideformerConfig, StrideformerCollator
from data import DataModule


logger = logging.getLogger(__name__)


def compute_metrics(eval_predictions):
    predictions, labels = eval_predictions
    predictions = np.argmax(predictions, axis=1)

    f1_micro = f1_score(labels, predictions, average="micro")
    f1_macro = f1_score(labels, predictions, average="macro")
    accuracy = (predictions == labels).mean().item()

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "accuracy": accuracy,
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    train_args = dict(cfg.training_arguments)
    training_args = TrainingArguments(**train_args)

    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_module.get_dataset("train") if training_args.do_train else None,
        eval_dataset=data_module.get_dataset("validation") if training_args.do_eval else None,
        tokenizer=data_module.tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:

        train_results = trainer.train()

        logger.info(train_results.metrics)

    if training_args.do_eval:
        eval_results = trainer.evaluate()

        logger.info(eval_results)

    if training_args.do_predict:

        predict_results = trainer.predict(test_dataset=data_module.get_dataset("test"))
        
        logger.info(predict_results.metrics)
        

if __name__ == "__main__":
    main()
