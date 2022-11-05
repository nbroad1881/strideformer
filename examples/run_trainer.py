#!/usr/bin/env python
# coding=utf-8
""" Finetuning short model chunks for long sequence classification"""

import sys
import logging
from functools import partial
from collections import defaultdict

import hydra
from omegaconf import DictConfig, OmegaConf
import datasets
import evaluate
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    AutoModelForSequenceClassification,
    AutoConfig,
    DataCollatorWithPadding,
)
from transformers.utils.logging import get_logger

from strideformer import Strideformer, StrideformerConfig, StrideformerCollator

from utils import (
    set_wandb_env_vars,
    set_mlflow_env_vars,
)
from data import DataModule

logger = get_logger(__name__)

def set_tracking_env_vars(cfg):
    """
    Sets environment variables for tracking tools.
    Currently supports WandB and MLFlow.

    Args:
        cfg (DictConfig):
            Hydra configuration
    """
    if "wandb" in cfg.training_arguments.report_to:
        set_wandb_env_vars(cfg)
    if "mlflow" in cfg.training_arguments.report_to:
        set_mlflow_env_vars(cfg)

def setup_logging(training_args):
    """
    Sets up logging level and prints out some information 
    about device, number of GPUs, and mixed precision.

    Args:
        training_args (TrainingArguments):
            training arguments
    """

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    mixed_precision = "fp16" if training_args.fp16 else "fp32"
    mixed_precision = "bf16" if training_args.use_bf16 else mixed_precision

     # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, mixed-precision: {mixed_precision}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


def get_model_config(cfg, data_module, use_strideformer):
    """
    Returns a StrideformerConfig or regular model configuration.

    Args:
        cfg (DictConfig):
            Hydra configuration
        data_module (DataModule):
            data module that contains label2id and id2label
        use_strideformer (bool):
            whether to use Strideformer

    Returns:
        AutoConfig or StrideformerConfig: model configuration
    """
    if use_strideformer:
        return StrideformerConfig(
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
    return AutoConfig.from_pretrained(
        cfg.model.model_name_or_path,
        num_labels=len(data_module.label2id),
        label2id=data_module.label2id,
        id2label=data_module.id2label,
    )


def get_model(model_config, cfg, use_strideformer):
    """
    Returns a Strideformer or regular model with designated
    configuration.

    Args:
        model_config (AutoConfig or StrideformerConfig):
            model configuration
        cfg (DictConfig):
            Hydra configuration
        use_strideformer (bool):
            whether to use Strideformer

    Returns:
        Strideformer or AutoModelForSequenceClassification: 
            model
    """
    if use_strideformer:
        return Strideformer(config=model_config, first_init=True)
    return AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_name_or_path, config=model_config
    )


def load_compute_metrics():
    """
    Returns:
        function: function that computes metrics for the trainer
    """

    metrics = {
        "accuracy": evaluate.load_metric("accuracy"),
        "f1_micro": evaluate.load_metric("f1"),
        "f1_macro": evaluate.load_metric("f1"),
        "precision": evaluate.load_metric("precision"),
        "recall": evaluate.load_metric("recall"),
    }

    metric_kwargs = defaultdict(lambda: {"average": "micro"})
    metric_kwargs["accuracy"] = {}

    def compute_metrics(eval_pred, metrics, metric_kwargs):
        predictions, labels = eval_pred
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions
        predictions = predictions.argmax(dim=1)

        all_scores = {
            name: metrics[name].compute(
                predictions=predictions, references=labels, **metric_kwargs[name]
            )
            for name in metrics
        }
        return all_scores

    return partial(compute_metrics, metrics=metrics, metric_kwargs=metric_kwargs)


def init_trackers(cfg):
    """
    Initializes tracking tools.

    Args:
        cfg (DictConfig):
            Hydra configuration
    """
    if "wandb" in cfg.training_arguments.report_to:
        import wandb

        wandb.init(
                config=OmegaConf.to_container(cfg),
            )
    if "mlflow" in cfg.training_arguments.report_to:
        import mlflow

        # TODO: add mlflow tracking
        # mlflow.set_experiment(cfg.mlflow.experiment_name)
        # mlflow.start_run(run_name=cfg.mlflow.run_name)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    train_args = dict(cfg.training_arguments)
    training_args = TrainingArguments(**train_args)

    set_tracking_env_vars(cfg)

    setup_logging(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    data_module = DataModule(cfg)

    # If running distributed, this will do something on the main process, while
    #    blocking replicas, and when it's finished, it will release the replicas.
    with training_args.main_process_first(desc="Dataset loading and tokenization"):
        data_module.prepare_dataset()

    USE_STRIDEFORMER = cfg.data.stride is not None

    if USE_STRIDEFORMER:
        collator = StrideformerCollator(tokenizer=data_module.tokenizer)
    else:
        collator = DataCollatorWithPadding(
            tokenizer=data_module.tokenizer, pad_to_multiple_of=cfg.data.pad_multiple
        )

    # batch_sizes must always be 1 when using strided approach
    if USE_STRIDEFORMER and (
        training_args.per_device_train_batch_size != 1
        or training_args.per_device_eval_batch_size != 1
    ):
        logger.warning(
            "Batch size must be 1 when using strided approach. Changing to 1 now."
        )
        training_args.per_device_train_batch_size = 1
        training_args.per_device_eval_batch_size = 1

    model_config = get_model_config(cfg, data_module, USE_STRIDEFORMER)

    model = get_model(model_config, cfg, USE_STRIDEFORMER)

    train_dataset = data_module.get_train_dataset() if training_args.do_train else None
    eval_dataset = data_module.get_eval_dataset() if training_args.do_eval else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=data_module.tokenizer,
        compute_metrics=load_compute_metrics(),
    )

    if training_args.do_train:

        init_trackers(cfg)

        train_results = trainer.train()
        train_metrics = train_results.metrics
        trainer.save_model()

        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()

    if training_args.do_eval:
        eval_results = trainer.evaluate(eval_dataset)
        eval_metrics = eval_results.metrics
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    
    if training_args.do_predict:
        predict_results = trainer.predict(data_module.get_test_dataset())
        predict_metrics = predict_results.metrics
        trainer.log_metrics("predict", predict_metrics)
        trainer.save_metrics("predict", predict_metrics)
