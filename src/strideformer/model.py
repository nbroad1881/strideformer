""" PyTorch Strideformer Model """
import os
from pathlib import Path
from typing import Optional, Tuple, Union, Iterator

import torch
from torch import nn
from transformers import PreTrainedModel, AutoModel, AutoConfig
from transformers.modeling_outputs import ModelOutput

from .config import StrideformerConfig


class StrideformerOutput(ModelOutput):
    """
    Base class for outputs of Strideformer model.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        first_model_last_hidden_states (`torch.FloatTensor`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        second_model_last_hidden_states (`torch.FloatTensor`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
    """

    loss: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    first_model_hidden_states: Optional[torch.FloatTensor] = None
    second_model_hidden_states: Optional[torch.FloatTensor] = None


class Strideformer(PreTrainedModel):
    def __init__(self, config: StrideformerConfig) -> None:
        """
        Initializes Strideformer model with random values.
        Use `from_pretrained` to load pretrained weights.
        """
        super().__init__(config)
        self.config = config

        self.first_model_config = AutoConfig.from_pretrained(config.first_model_name_or_path)
        self.first_model = AutoModel.from_config(self.first_model_config)
        self.max_chunks = config.max_chunks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
        )
        self.second_model = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_hidden_layers
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self._init_weights(self.modules(), self.config.initializer_range)

    @staticmethod
    def mean_pooling(
        output_embeddings: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Mean pool across the `sequence_length` dimension. Assumes that output 
        embeddings have shape `(batch_size, sequence_length, hidden_size)`.
        If batched, there can be pad tokens in the sequence.
        This will ignore padded outputs when doing mean pooling by using
        `attention_mask`.

        Args:
            output_embeddings (`torch.FloatTensor`):
                Embeddings to be averaged across the first dimension.
            attention_mask (`torch.LongTensor`):
                Attention mask for the embeddings. Used to ignore 
                padd tokens from the averaging.

        Returns:
            `torch.FloatTensor`of shape `(batch_size, hidden_size)` that is 
            `output_embeddings` averaged across the 1st dimension.
        """

        token_embeddings = output_embeddings
        if attention_mask is not None:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )

        # this might be wrong
        return torch.sum(token_embeddings, 1) / torch.clamp(
            token_embeddings.sum(1), min=1e-9
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = 1,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], StrideformerOutput]:
        """
        Args:
            input_ids (`torch.Tensor`, *optional*, defaults to None):
                Indices of input sequence tokens in the vocabulary. These can be created
                using the corresponding tokenizer for the first model.
                Shape is `(num_chunks, sequence_length)` where `num_chunks` is `(batch_size*chunks_per_batch)`.
            token_type_ids (`torch.Tensor`, *optional*, defaults to None):
                Some models take token_type_ids. This comes from the tokenizer and gets
                passed to the first model.
            labels (`torch.FloatTensor` or `torch.Tensor`, *optional*, defaults to None):
                The true values. Used for loss calculation.
                Shape is `(batch_size, num_classes)` if multilabel,
                `(batch_size, 1)` for multiclass or regression.
            batch_size (`int`, *optional*, defaults to 1):
                If passing batched inputs, this specifies the shape of input for the second model.
                The first model will get input `(num_chunks, sequence_length)` wherewhere `num_chunks`
                is `(batch_size*chunks_per_batch)`. The output of the first model is `(num_chunks, hidden_size)`.
                This gets reshaped to `(batch_size, chunks_per_batch, hidden_size)`. This means that
                all document sequences must be tokenized to the same number of chunks.
        Returns:
            A `tuple` of `torch.Tensor` if `return_dict` is `False`. 
            A `StrideformerOutput` object if `return_dict` is None or True.
            These containers hold values for loss, logits, and last hidden states for 
            both models.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        token_type_ids = {"token_type_ids": token_type_ids} if token_type_ids is not None else {}

        if self.config.freeze_first_model:
            # No gradients, no training, save memory
            with torch.no_grad():
                first_model_hidden_states = self.first_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **token_type_ids,
                )[0]
        else:
            first_model_hidden_states = self.first_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **token_type_ids,
            )[0]

        # mean pool last hidden state 
        embeddings = self.mean_pooling(first_model_hidden_states, attention_mask=attention_mask)

        second_model_hidden_states = self.second_model(
            embeddings.reshape(batch_size, -1, self.config.hidden_size),
        )  # [batch_size, chunks_per_batch, hidden_size]

        # Classifier uses mean pooling to combine output embeddings into single embedding.
        second_model_chunk_logits = self.classifier(
            second_model_hidden_states
        )  # [batch_size, chunks_per_batch, num_labels]
        logits = second_model_chunk_logits.mean(dim=1)  # [batch_size, num_labels]

        loss = None
        if labels is not None:

            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.config.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not output_hidden_states:
            first_model_hidden_states = None
            second_model_hidden_states = None

        if not return_dict:
            output = (logits, first_model_hidden_states, second_model_hidden_states)
            return ((loss,) + output) if loss is not None else output

        return StrideformerOutput(
            loss=loss,
            logits=logits,
            first_model_hidden_states=first_model_hidden_states,
            second_model_hidden_states=second_model_hidden_states,
        )

    @staticmethod
    def _init_weights(modules: Iterator[nn.Module], std: float = 0.02) -> None:
        """
        Reinitializes every Linear, Embedding, and LayerNorm module provided.
        Args:
            modules (Iterator of `torch.nn.Module`)
                Iterator of modules to be initialized. Typically by calling Module.modules()
            std (`float`, *optional*, defaults to 0.02)
                Standard deviation for normally distributed weight initialization 
        """
        for module in modules:
            if isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
