import os

import torch
from transformers.utils import logging

logger = logging.get_logger(__name__)


def set_wandb_env_vars(cfg):
    """
    Set environment variables from the config dict object.
    The environment variables can be picked up by wandb in Trainer.
    """

    os.environ["WANDB_ENTITY"] = getattr(cfg.wandb, "entity", "")
    os.environ["WANDB_PROJECT"] = getattr(cfg.wandb, "project", "")
    os.environ["WANDB_RUN_GROUP"] = getattr(cfg.wandb, "group", "")
    os.environ["WANDB_JOB_TYPE"] = getattr(cfg.wandb, "job_type", "")
    os.environ["WANDB_NOTES"] = getattr(cfg.wandb, "notes", "")
    if cfg.wandb.tags:
        os.environ["WANDB_TAGS"] = ",".join(cfg.wandb.tags)


def set_mlflow_env_vars(cfg):
    """
    Set environment variables from the config object.
    The environment variables can be picked up by mlflow in Trainer.
    """

    os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = cfg.mlflow.log_artifact
    os.environ["MLFLOW_NESTED_RUN"] = cfg.mlflow.nested_run
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow.experiment_name
    os.environ["MLFLOW_FLATTEN_PARAMS"] = cfg.mlflow.flatten_params
    os.environ["MLFLOW_RUN_ID"] = cfg.mlflow.run_id


def reinit_modules(modules, std, reinit_embeddings=False):
    """
    Reinitializes every Linear, Embedding, and LayerNorm module provided.
    """
    for module in modules:
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif reinit_embeddings and isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
