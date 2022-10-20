import torch
from transformers import AutoTokenizer

from .model import Strideformer, StrideformerConfig


class Pipeline:
    """
    Pipeline abstraction to make it easy to use a
    Strideformer model.
    """

    def __init__(
        self, model_name_or_path=None, model=None, tokenizer=None, device="cpu"
    ):
        """
        Load pretrained model and tokenizer from path or objects.

        Args:
            model_name_or_path (`str`, *optional*, defaults to `None`):
                Location of model weights and configuration file.
            model (`Strideformer`, *optional*, defaults to `None`):
                Strideformer model to directly load.
            tokenizer (`PreTrainedTokenizer`, *optional*, defaults to `None`):
                Tokenizer to directly load.
            device (`str`, *optional*, defaults to `"cpu"`):
                String indicating which device the model should run on.
        """
        if model_name_or_path is not None:
            config = StrideformerConfig.from_pretrained(model_name_or_path)
            self.model = Strideformer.from_pretrained(model_name_or_path, config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        else:
            self.model = model
            self.tokenizer = tokenizer

        self.model.eval()
        self.device = torch.device(device)

        self.model.to(self.device)

    @torch.no_grad()
    def __call__(self, text, max_length=384, stride=128, return_type="list"):
        """
        Pass the model a string and it will return predictions. Only works for
        one string and not a list of strings.

        Args:
            text (`str`):
                Text to be classified.
            max_length (`int`, *optional*, defaults to 384):
                Maximum sequence length for the first model.
            stride (`int`, *optional*, defaults to 128):
                Amount of overlap between token chunks.
            return_type (`str`, *optional*, defaults to `"list"`):
                String to determine what type of results to return.
                Options are `"np"`, `"pytorch"`, and `"list"`.

        Returns:
            (`np.array`, `torch.Tensor`, `list`): Classification scores.
        """

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            stride=stride,
            return_overflowing_tokens=True,
            padding=True,
            truncation=True,
        )

        del tokens["overflow_to_sample_mapping"]

        output = self.model(**tokens.to(self.device))

        if return_type == "np":
            return output.logits.detach().cpu().numpy()
        elif return_type == "pt":
            return output.logits.detach().cpu()

        return output.logits.detach().cpu().tolist()
