from pathlib import Path

import torch
from torch import nn
from transformers import PreTrainedModel, AutoModel, AutoConfig

from utils import reinit_modules


class StridedLongformer(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.short = AutoModel.from_pretrained(config.short_model)
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
    def mean_pooling(output_embeddings, attention_mask):
        """
        If batched, there can be pad tokens in the sequence.
        This will ignore padded outputs when doing mean pooling.
        """

        token_embeddings = output_embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        lengths=None,
    ):
        """
        input_ids will be of shape [num_chunks, sequence_length] where num_chunks can be across multiple documents
        attention_mask is same shape as input_ids
        labels is [num_documents, num_classes] if multilabel, [num_documents] for multiclass or regression
        lengths is list of length [num_documents]
        """

        # short model is frozen, so no gradients needed
        all_embeddings = []
        with torch.no_grad():
            bs = 32
            # TODO: have a better way to handle batch size for the short model

            for i in range(0, input_ids.size(0), bs):
                short_model_output = self.short_model(
                    input_ids=input_ids[i : i + bs, :],
                    attention_mask=attention_mask[i : i + bs, :],
                )
                embeddings = self.mean_pooling(
                    short_model_output[0], attention_mask[i : i + bs, :]
                )

                all_embeddings.append(embeddings)

        embeddings = torch.vstack(all_embeddings)

        # embeddings will be of shape [num_chunks, hidden_dim]
        # Now need to be reshaped to separate documents

        max_seq_len = max(lengths)
        hidden_dim = embeddings.size(-1)

        docs = []
        start = 0
        for length in lengths:
            temp = embeddings[start : start + length, :]
            z = torch.zeros(
                (max_seq_len - length, hidden_dim), dtype=temp.dtype, device=temp.device
            )
            docs.append(torch.vstack([temp, z]).unsqueeze(0))
            start += length

        new_input = torch.vstack(docs)

        attention_mask = torch.all(new_input == 0, dim=-1).to(new_input.device)

        transformer_output = self.transformer(
            new_input, src_key_padding_mask=attention_mask
        )

        logits = self.mean_pooling(self.classifier(transformer_output), attention_mask)

        loss = None
        if labels is not None:
            # TODO: Implement multilabel, regression losses

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    @classmethod
    def from_pretrained(cls, model_name_or_path, config=None, **kwargs):
        model_name_or_path = Path(model_name_or_path)
        if config is None:
            config = AutoConfig.from_pretrained(model_name_or_path)
        model = StridedLongformer(config)

        if model_name_or_path.is_dir():
            model.load_state_dict(torch.load(model_name_or_path / "pytorch_model.bin"))
            return model

        model.short_model = AutoModel.from_pretrained(model_name_or_path)

        reinit_modules(model.transformer.modules(), std=config.initializer_range)
        reinit_modules(model.classifier.modules(), std=config.initializer_range)

        return model
