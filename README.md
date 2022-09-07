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

## Basic run

Start by installing the required packages using `pip install -r requirements.txt`. Some additional packages might also be necessary depending on the tracking framework and models chosen (see [here](#optional-packages)). Then change the values inside `conf/config.yaml` to your desired parameters. If you set `data.stride` to 0 or null, then it will just run normally.  

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
