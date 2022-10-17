""" Strideformer model configuration """
from typing import Optional, List

from transformers import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class StrideformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Strideformer`]. It is used to instantiate a BStrideformerART
    model according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        first_model_name_or_path (`str`, *optional*, defaults to `"sentence-transformers/all-MiniLM-L6-v2"`):
            The model name or path to the first model that is usually a pre-trained sentence transformer.
        freeze_first_model (`bool`, *optional*, defaults to `True`):
            If True, freeze the weights of the first model. Otherwise, train it as well.
        max_chunks (`int`, *optional*, defaults to 64)
            The maximum number of chunks the first model can take.
        hidden_size (`int`, *optional*, defaults to 384):
            The hidden size of the second model. It must match the first model's hidden size.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation when initializing weights using a normal distribution.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of layers for the second model.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the second model.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in second model.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the second model.
        layer_norm_eps (`float`, *optional*, defaults to 1e-7): 
            The epsilon value in LayerNorm
        num_labels (`int`, *optional*, defaults to 2):
            The number of labels for the classifier.

    Example:
    ```python
    >>> from strideformer import StrideformerConfig, Strideformer
    >>> config = StrideformerConfig("sentence-transformers/all-MiniLM-L6-v2", max_chunks=128)
    >>> model = Strideformer(config)
    ```"""
    model_type: str = "strideformer"
    keys_to_ignore_at_inference: List = []

    def __init__(
        self,
        first_model_name_or_path: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
        freeze_first_model: Optional[bool] = True,
        max_chunks: Optional[int] = 64,
        hidden_size: Optional[int] = 384,
        initializer_range: Optional[float] = 0.02,
        num_hidden_layers: Optional[int] = 24,
        num_attention_heads: Optional[int] = 12,
        intermediate_size: Optional[int] = 4096,
        hidden_act: Optional[str] = "gelu",
        dropout: Optional[float] = 0.1,
        layer_norm_eps: Optional[float] = 1e-7,
        num_labels: Optional[int] = 2,
        **kwargs
    ):
        self.first_model_name_or_path = first_model_name_or_path
        self.freeze_first_model = freeze_first_model
        self.max_chunks = max_chunks
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_labels = num_labels

        super().__init__(
            num_labels=num_labels,
            **kwargs,
        )