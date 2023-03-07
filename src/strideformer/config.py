""" Strideformer model configuration """
import copy
from typing import Dict

from transformers import PretrainedConfig, AutoConfig


class StrideformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Strideformer`]. It is used to instantiate a BStrideformerART
    model according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        first_model_config (`PretrainedConfig`, *optional*, defaults to `None`):
            The configuration of the first model. 
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
    >>> from transformers import AutoConfig
    >>> first_model_config = AutoConfig.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    >>> config = StrideformerConfig(first_model_config, max_chunks=128)
    >>> model = Strideformer(config)
    ```

    """
    model_type = "strideformer"
    keys_to_ignore_at_inference = [
        "first_model_hidden_states",
        "second_model_hidden_states",
    ]
    is_composition = True

    def __init__(
        self,
        **kwargs
    ):        
        first_model_config_dict = kwargs.pop("first_model_config")
        first_model_type = first_model_config_dict.pop("model_type")
        self.first_model_config = AutoConfig.for_model(first_model_type, **first_model_config_dict)

        super().__init__(**kwargs)
        


    @classmethod
    def from_two_configs(cls, first_model_config: PretrainedConfig, second_model_config: Dict):
        
        return cls(
            first_model_config=first_model_config.to_dict(),
            **second_model_config
        )

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default *to_dict()* from *PretrainedConfig*.
        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["first_model_config"] = self.first_model_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output