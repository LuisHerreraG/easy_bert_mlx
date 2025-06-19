# Easy-BERT-MLX

This is a fast implementation for everyone who needs it for BERT in the MLX Framework (Apple's Machine Learning Framework). You can use BERT currently if you go to mlx-examples, but it requires manual convertion of BERT and isn't connected with pip. That's the only reason I made this, to be easily installed.

## Installation

```
pip install git+https://github.com/yourusername/easy_bert_mlx.git
```

That's it!

## Usage

You can use the model by just using

```
model, tokenizer = easy_bert_mlx.model.load_model_huggingface("bert-base-uncased")
```

That's it, it will be downloaded in your HuggingFace cache and can be used like in MLX-Examples.

You can use DistilBERT too, I have tested it.

To truly use its inference though, I will hopefully update the README.md soon! This was made to make a bigger project which I will release soon (hopefully too).

## Final Words

That's all, xoxo.

MKU64.
