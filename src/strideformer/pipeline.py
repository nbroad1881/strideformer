import torch
from transformers import AutoTokenizer

from .model import Strideformer

class Pipeline:

    def __init__(self, model_name_or_path=None, model=None, tokenizer=None, device=None):
        if model_name_or_path is not None:
            self.model = Strideformer.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.model = model
            self.tokenizer = tokenizer
        
        self.model.eval()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model.to(self.device)

    def __call__(self, text, max_length=384, stride=128, return_type="list"):

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            padding=True,
            truncation=True,
        )

        output = self.model(**tokens.to(self.device))

        if return_type == "np":
            return output.logits.detach().cpu().numpy()
        elif return_type == "list":
            return output.logits.detach().cpu().tolist()
            
        return output.logits.detach().cpu()

        