from pathlib import Path

import torch
from torch import nn
from transformers import PreTrainedModel, AutoModel, AutoConfig


class Strideformer(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.short_model = AutoModel.from_pretrained(config.short_model)
        self.short_model_max_chunks = config.short_model_max_chunks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            dim_feedforward=config.intermediate_size,
            nhead=config.num_attention_heads,
            batch_first=True,
            activation=config.hidden_act,
            layer_norm_eps=config.layer_norm_eps,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_hidden_layers
        )
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    @staticmethod
    def mean_pooling(output_embeddings, attention_mask=None):
        """
        If batched, there can be pad tokens in the sequence.
        This will ignore padded outputs when doing mean pooling.
        """

        token_embeddings = output_embeddings
        if attention_mask is not None:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        
        return torch.sum(token_embeddings, 1) / torch.clamp(token_embeddings.size(1), min=1e-9)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        """
        input_ids will be of shape [num_chunks, sequence_length] where num_chunks should only be for one document
        labels is [num_classes] if multilabel, [1] for multiclass or regression
        """
        
        # short model is frozen, so no gradients needed
        with torch.no_grad():
            short_model_output = self.short_model(
                input_ids=input_ids[:self.short_model_max_chunks, :],
                attention_mask=attention_mask[:self.short_model_max_chunks, :],
            )
            embeddings = short_model_output[0].mean(dim=1)

        # embeddings will be of shape [num_chunks, hidden_dim]
        transformer_output = self.transformer(
            embeddings, 
        )

        logits = self.classifier(transformer_output).mean(dim=0)
        # logits will be of shape [num_labels]

        loss = None
        if labels is not None:
            # TODO: Implement multilabel, regression losses

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.unsqueeze(0), labels)

        return {"loss": loss, "logits": logits}

    @classmethod
    def from_pretrained(cls, model_name_or_path, config=None, **kwargs):
        model_name_or_path = Path(model_name_or_path)
        if config is None:
            config = AutoConfig.from_pretrained(model_name_or_path)
        model = Strideformer(config)

        if model_name_or_path.is_dir():
            model.load_state_dict(torch.load(model_name_or_path / "pytorch_model.bin"))
            return model

        model.short_model = AutoModel.from_pretrained(model_name_or_path)

        reinit_modules(model.transformer.modules(), std=config.initializer_range)
        reinit_modules(model.classifier.modules(), std=config.initializer_range)

        return model

    
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