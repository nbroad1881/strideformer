import os
import json
import importlib
from typing import Optional, Any, Union, Dict


import torch
from transformers import TrainingArguments

from accelerate.tracking import GeneralTracker
from accelerate.logging import get_logger

logger = get_logger(__name__)


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

    os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = str(cfg.mlflow.log_artifact)
    os.environ["MLFLOW_NESTED_RUN"] = str(cfg.mlflow.nested_run)
    os.environ["MLFLOW_EXPERIMENT_NAME"] = cfg.mlflow.experiment_name
    os.environ["MLFLOW_FLATTEN_PARAMS"] = str(cfg.mlflow.flatten_params)
    os.environ["MLFLOW_RUN_ID"] = str(cfg.mlflow.run_id)
    os.environ["MLFLOW_TAGS"] = json.dumps(cfg.mlflow.tags)


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


training_args_to_log = {
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "evaluation_strategy",
    "eval_delay",
    "learning_rate",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "max_grad_norm",
    "num_train_epochs",
    "max_steps",
    "lr_scheduler_type",
    "warmup_ratio",
    "warmup_steps",
    "log_level",
    "logging_dir",
    "logging_strategy",
    "logging_steps",
    "save_strategy",
    "save_steps",
    "save_total_limit",
    "seed",
    "bf16",
    "fp16",
    "tf32",
    "dataloader_drop_last",
    "eval_steps",
    "dataloader_num_workers",
    "metric_for_best_model",
    "greater_is_better",
    "label_smoothing_factor",
    "optim",
    "adafactor",
    "group_by_length",
    "resume_from_checkpoint",
    "gradient_checkpointing",
}

if importlib.util.find_spec("mlflow") is not None:
    import mlflow


class MLflowTracker(GeneralTracker):
    """
    A `Tracker` class that supports `mlflow`. Should be initialized at the start of your script.
    Args:
        experiment_name (`str`):
            Name of the experiment.
            Environment variable MLFLOW_EXPERIMENT_NAME has priority over this argument.
        logging_dir (`str`, `os.PathLike`):
            Location for mlflow logs to be stored.
        run_id (`str`):
            If specified, get the run with the specified UUID and log parameters and metrics under that run.
            The run’s end time is unset and its status is set to running, but the run’s other attributes
            (source_version, source_type, etc.) are not changed.
            Environment variable MLFLOW_RUN_ID has priority over this argument.
        tags (`dict`, `str`):
            An optional `dict` of `str` keys and values, or a `str` dump from a `dict`,
            to set as tags on the run. If a run is being resumed, these tags are set on the resumed run.
            If a new run is being created, these tags are set on the new run.
            Environment variable MLFLOW_TAGS has priority over this argument.
        nested_run (`bool`):
            Controls whether run is nested in parent run. True creates a nested run.
            Environment variable MLFLOW_NESTED_RUN has priority over this argument.
        run_name (`str`):
            Name of new run (stored as a mlflow.runName tag). Used only when run_id is unspecified.
        description (`str`):
            An optional string that populates the description box of the run.
            If a run is being resumed, the description is set on the resumed run.
            If a new run is being created, the description is set on the new run.
    """

    name = "mlflow"
    requires_logging_directory = True

    def __init__(
        self,
        experiment_name: Optional[str] = None,
        logging_dir: Optional[Union[str, os.PathLike]] = ".",
        run_id: Optional[str] = None,
        tags: Optional[Union[Dict[str, Any], str]] = None,
        nested_run: Optional[bool] = False,
        run_name: Optional[str] = None,
        description: Optional[str] = None,
    ):

        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        self._MAX_PARAMS_TAGS_PER_BATCH = (
            mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH
        )

        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", experiment_name)
        run_id = os.getenv("MLFLOW_RUN_ID", run_id)
        tags = os.getenv("MLFLOW_TAGS", tags)
        if isinstance(tags, str):
            tags = json.loads(tags)

        nested_run = os.getenv("MLFLOW_NESTED_RUN", nested_run)

        if mlflow.active_run() and not nested_run:
            raise ValueError("Detected active run. `nested_run` must be set to True.")

        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=logging_dir,
            tags=tags,
        )

        self.active_run = mlflow.start_run(
            run_id=run_id,
            experiment_id=experiment_id,
            run_name=run_name,
            nested=nested_run,
            tags=tags,
            description=description,
        )

        logger.debug(f"Initialized mlflow experiment {experiment_name}")
        logger.debug(
            "Make sure to log any initial configurations with `self.store_init_configuration` before training!"
        )

    @property
    def tracker(self):
        return self.active_run

    def save_training_args(self, training_args: TrainingArguments):
        """
        To be used in Azure ML. Since the log limit is 100 parameters,
        this method should be called first to save the training arguments to a
        json file. The method then returns a filtered down dict that can be
        logged as parameters when passed as a configuration argument to
        `init_trackers`.
        Args:
            training_arguments (`TrainingArguments`):
                TrainingArguments to save.
        Returns:
            `dict` of the key/value pairs to be logged as parameters.
        """
        mlflow.log_dict(training_args.to_dict(), "training_arguments.json")

        return {
            k: v
            for k, v in training_args.to_dict().items()
            if k in training_args_to_log
        }

    def store_init_configuration(self, values: dict):
        """
        Logs `values` as hyperparameters for the run. Should be run at the beginning of your experiment.
        Args:
            values (`dict`):
                Values to be stored as initial hyperparameters as key-value pairs.
        """

        for name, value in list(values.items()):
            # internally, all values are converted to str in MLflow
            if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                logger.warning(
                    f'Trainer is attempting to log a value of "{value}" for key "{name}" as a parameter. MLflow\'s'
                    f" log_param() only accepts values shorter than {self._MAX_PARAM_VAL_LENGTH} characters so we dropped this attribute."
                )
                del values[name]

        values_list = list(values.items())

        if os.getenv("AML_CloudName") == "AzureCloud":
            values_list = values_list[:100]

        # MLflow cannot log more than 100 values in one go, so we have to split it
        for i in range(0, len(values_list), self._MAX_PARAMS_TAGS_PER_BATCH):
            mlflow.log_params(
                dict(values_list[i : i + self._MAX_PARAMS_TAGS_PER_BATCH])
            )

        logger.debug("Stored initial configuration hyperparameters to MLflow")

    def log(self, values: dict, step: Optional[int]):
        """
        Logs `values` to the current run.
        Args:
            values (`dict`):
                Values to be logged as key-value pairs.
            step (`int`, *optional*):
                The run step. If included, the log will be affiliated with this step.
        """
        metrics = {}
        for k, v in values.items():
            if isinstance(v, (int, float)):
                metrics[k] = v
            else:
                logger.warning(
                    f'MLflowTracker is attempting to log a value of "{v}" of type {type(v)} for key "{k}" as a metric. '
                    "MLflow's log_metric() only accepts float and int types so we dropped this attribute."
                )

        mlflow.log_metrics(metrics, step=step)
        logger.debug("Successfully logged to mlflow")

    def finish(self):
        """
        End the active MLflow run.
        """
        mlflow.end_run()
