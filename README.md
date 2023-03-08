# stride-former

Using short models to classify long texts. Still a work-in-progress, but the general skeleton is there. This is very similar to this: [Three-level Hierarchical Transformer Networks for Long-sequence and Multiple Clinical Documents Classification](https://arxiv.org/abs/2104.08444). 

[Here is a good illustration from the paper.](https://www.semanticscholar.org/paper/Three-level-Hierarchical-Transformer-Networks-for-Si-Roberts/2cc7d0a242f8358bf5e3e41defd2e93e6297f74c/figure/0)
## How it works

Since attention in transformers scales quadratically with sequence length, it can become infeasible to do full self-attention on sequences longer than 512 tokens. Models like [Longformer](https://arxiv.org/abs/2004.05150) and [Big Bird](https://arxiv.org/abs/2007.14062) use different attention mechanisms to reduce the quadratic scaling to linear scaling, allowing them to work on sequences up to 4096 tokens. This approach chunks the text into 512 token chunks and then aggregates those embeddings by putting them through another transformer. Because there are two models involved, the first one is a pre-trained sentence transformer to reduce the number of trainable parameters. The second model is a [generic transformer encoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html#torch.nn.TransformerEncoder).  This repo is called stride-former because there is an overlap between chunks of text - typically of 256 tokens, which is referred to as a stride. 

## Is it better than Longformer or Bigbird?

In a very small and un-scientific set of experiments on a small subset of one dataset ([arxiv-classification](https://huggingface.co/datasets/ccdv/arxiv-classification)), the stride-former did best. This dataset has extremely long documents: about half of the documents are over 15k tokens and 25% are over 23k. This is far more than the 4096 token limit of Longformer and Bigbird, and much more than the typical 512 token limit of standard full-attention models. The current default parameters for the stride-former are: chunks of 384 tokens, stride by 128 tokens, and limit the total number of chunks to 128, which means that documents with 32k tokens can be passed through.  

The two rows at the bottom of the table below are the stride-former runs which use sentence transformers to encode the chunks. 

|model               |max_seq_length|stride|eval_f1_micro      |eval_f1_macro      |
|---------------------------------------|-------------------|-----------------|-----------|-------------------|
|microsoft/deberta-v3-base              |512                |0          |0.2871 |0.16918|
|microsoft/deberta-v3-base              |512                |0          |0.2475|0.1494|
|allenai/longformer-base-4096           |4096               |0          |0.6732 |0.6094 |
|allenai/longformer-base-4096           |4096               |0          |0.6831 |0.6295 |
|google/bigbird-roberta-base            |4096               |0          |0.5841 |0.5118  |
|google/bigbird-roberta-base            |4096               |0          |0.6534 |0.6064 |
|sentence-transformers/all-MiniLM-L12-v2|384                |128        |0.7227 |0.6728 |
|sentence-transformers/all-MiniLM-L12-v2|384                |128        |0.6831 |0.6386  |


See the [Weights and Biases report here for more details.](https://wandb.ai/nbroad/stride-former/reports/Stride-former-comparison--VmlldzoyNTUyOTEy?accessToken=p5x55isxp9thu5ktrhrrlmm98c82ckaagcam3r2re43mye8z45763mudidrb4vml)

## How many chunks?

If the max sequence length is `L` and the stride is `S`, and the original sequence length without striding is `N`, then here is how to calculate the number of chunks. Chunk 1 will consist of a sequence starting at token `0` and going until, but not including, `L`. Chunk 2 will start at token `L - S` and end at `2L - S`, chunk `C` will end at `CL - (C - 1)S`. How many chunks does it take for an original sequence of `N` tokens? 

```text
CL - (C - 1)*S > N
C*(L - S) + S > N
C*(L - S) > N - S
C > (N - S)/(L - S)
```

## Installation

```sh
pip install strideformer
```

## Basic usage

```python
from strideformer import StrideformerConfig, Strideformer
from transformers import AutoConfig, AutoTokenizer

label2id = {"positive": 0, "negative": 1}
id2label = {"0": "positive", "1": "negative"}

first_model_name = "sentence-transformers/all-MiniLM-L6-v2"

first_model_config = AutoConfig.from_pretrained(
            first_model_name,
            label2id=label2id,
            id2label=id2label,
            num_labels=len(label2id),
    )
second_model_config = dict(
            freeze_first_model=False, # Train first model?
            max_chunks=128, # Maximum number of chunks to consider
            num_hidden_layers=24,
            num_attention_heads=12,
            intermediate_size=4096,
            hidden_act="gelu",
            dropout=0.1,
            layer_norm_eps=1e-7,
            initializer_range=0.02,
            hidden_size=first_model_config.hidden_size, # Needs to match first model
            label2id=label2id,
            id2label=id2label,
            num_labels=len(label2id),
            )

model_config = StrideformerConfig.from_two_configs(
            first_model_config=first_model_config,
            second_model_config=second_model_config,
        )

model = Strideformer(
            config=model_config
        )

tokenizer = AutoTokenizer.from_pretrained(first_model_name)
# Set the max_length and stride to whatever values you like (256, 128 are good starting points)
inputs = tokenizer(
    ["Here is a bunch of text that should be strided over"],
     return_tensors="pt",
     return_overflowing_tokens=True,
     padding=True,
     truncation=True,
     max_length=5,
     stride=2
     ) 

print(inputs.input_ids.shape)
# torch.Size([10, 5])

outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
print(outputs)
# StrideformerOutput([('loss', None),
#                    ('logits',
#                     tensor([[ 0.4127, -0.2096]], grad_fn=<MeanBackward1>)),
#                    ('first_model_hidden_states', None),
#                    ('second_model_hidden_states', None)])

output_path = "trained_strideformer"
model.save_pretrained(output_path)

# `from_pretrained` is only to be used after training Strideformer
# Use `Strideformer(config)` before training
loaded_model = Strideformer.from_pretrained(output_path)
```


## Basic run

Start by installing the required packages in the examples folder using `pip install -r requirements.txt`. Some additional packages might also be necessary depending on the tracking framework and models chosen (see [here](#optional-packages)). Then change the values inside `examples/conf/config.yaml` to your desired parameters. If you set `data.stride` to <=0 or null, then it will train with a regular approach (no strides).  

- `examples/run_classification.py` uses the Hugging Face Trainer and abstracts much of the training complexities away.  
- `examples/run_classification_no_trainer.py` uses Hugging Face Accelerate and allows for more customization.

```sh
python run_classification.py
```

## Dataset

It is currently set up for two datasets with long texts: [health_fact](https://huggingface.co/datasets/health_fact) and [arxiv-classification](https://huggingface.co/datasets/ccdv/arxiv-classification). `data.py` contains a data module and processors to make it relatively simple to load in any dataset.

## Optional packages

```text
mlflow
wandb
sentencepiece
```
