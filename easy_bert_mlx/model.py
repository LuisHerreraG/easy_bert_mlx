import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerBase


class TransformerEncoderLayer(nn.Module):
    """
    A transformer encoder layer with (the original BERT) post-normalization.
    """

    def __init__(
        self,
        dims: int,
        num_heads: int,
        mlp_dims: Optional[int] = None,
        layer_norm_eps: float = 1e-12,
        approx = 'none'
    ):
        super().__init__()
        mlp_dims = mlp_dims or dims * 4
        self.attention = nn.MultiHeadAttention(dims, num_heads, bias=True)
        self.ln1 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.ln2 = nn.LayerNorm(dims, eps=layer_norm_eps)
        self.linear1 = nn.Linear(dims, mlp_dims)
        self.linear2 = nn.Linear(mlp_dims, dims)
        self.gelu = nn.GELU(approx=approx)

    def __call__(self, x, mask):
        attention_out = self.attention(x, x, x, mask)
        add_and_norm = self.ln1(x + attention_out)

        ff = self.linear1(add_and_norm)
        ff_gelu = self.gelu(ff)
        ff_out = self.linear2(ff_gelu)
        x = self.ln2(ff_out + add_and_norm)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self, num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None, approx = 'none'
    ):
        super().__init__()
        self.layers = [
            TransformerEncoderLayer(dims, num_heads, mlp_dims, approx=approx)
            for i in range(num_layers)
        ]

    def __call__(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return x


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def __call__(
        self, input_ids: mx.array, token_type_ids: mx.array = None
    ) -> mx.array:
        words = self.word_embeddings(input_ids)
        position = self.position_embeddings(
            mx.broadcast_to(mx.arange(input_ids.shape[1]), input_ids.shape)
        )

        if token_type_ids is None:
            # If token_type_ids is not provided, default to zeros
            token_type_ids = mx.zeros_like(input_ids)

        token_types = self.token_type_embeddings(token_type_ids)

        embeddings = position + words + token_types
        return self.norm(embeddings)


class Bert(nn.Module):
    def __init__(self, config, approx='none'):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = TransformerEncoder(
            num_layers=config.num_hidden_layers,
            dims=config.hidden_size,
            num_heads=config.num_attention_heads,
            mlp_dims=config.intermediate_size,
            approx=approx
        )
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

    def __call__(
        self,
        input_ids: mx.array,
        token_type_ids: mx.array = None,
        attention_mask: mx.array = None,
        output_hidden_states: bool = False,
    ):
        # Get the embedding output.
        x = self.embeddings(input_ids, token_type_ids)
        # Optionally collect hidden states.
        hidden_states = [x] if output_hidden_states else None

        # Prepare attention mask if provided.
        if attention_mask is not None:
            # For example, converting mask (0s,1s) to log scale for attention bias.
            attention_mask = mx.log(attention_mask)
            attention_mask = mx.expand_dims(attention_mask, (1, 2))

        # Run through each encoder layer.
        for layer in self.encoder.layers:
            x = layer(x, attention_mask)
            if output_hidden_states:
                hidden_states.append(x)

        # Compute pooled output (e.g. using the CLS token).
        pooled_output = mx.tanh(self.pooler(x[:, 0]))

        # Return outputs in a dictionary similar to Hugging Face.
        outputs = {
            "last_hidden_state": x,
            "pooler_output": pooled_output,
        }
        if output_hidden_states:
            outputs["hidden_states"] = hidden_states

        return outputs



def load_model(
    bert_model: str, weights_path: str, approx='none'
) -> Tuple[Bert, PreTrainedTokenizerBase]:
    if not Path(weights_path).exists():
        raise ValueError(f"No model weights found in {weights_path}")

    config = AutoConfig.from_pretrained(bert_model)

    # create and update the model
    model = Bert(config)
    model.load_weights(weights_path)

    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    return model, tokenizer


def load_model_huggingface(
        bert_model: str, approx = 'none'
) -> Tuple[Bert, PreTrainedTokenizerBase]:
    """
    Loads a BERT-like model and tokenizer from the Hugging Face Hub.

    This function automatically downloads pre-converted MLX weights from
    the 'mlx-community' organization on Hugging Face.
    """
    # Construct the Hugging Face repo ID for the MLX model
    mlx_repo_id = f"mlx-community/{bert_model}-mlx"

    try:
        weights_path = hf_hub_download(repo_id=mlx_repo_id, filename="weights.npz")
    except Exception as e:
        print(f"Could not find pre-converted MLX weights at {mlx_repo_id}.")
        print(f"Original error: {e}")
        print("Please ensure the model has been converted and uploaded to mlx-community,")
        print("or use convert.py to convert weights manually and load them from a local path.")
        raise e

    config = AutoConfig.from_pretrained(bert_model)

    # Create and update the model
    model = Bert(config, approx)

    # Load the weights
    model.load_weights(weights_path)

    tokenizer = AutoTokenizer.from_pretrained(bert_model)

    return model, tokenizer

def run(bert_model: str, mlx_model: Optional[str], batch: List[str]):
    model, tokenizer = load_model_huggingface(bert_model, mlx_model)

    tokens = tokenizer(batch, return_tensors="np", padding=True)
    tokens = {key: mx.array(v) for key, v in tokens.items()}

    return model(**tokens)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BERT model using MLX.")
    parser.add_argument(
        "--bert-model",
        type=str,
        default="bert-base-uncased",
        help="The huggingface name of the BERT model to use.",
    )
    parser.add_argument(
        "--mlx-model",
        type=str,
        default=None,  # Optional: allows downloading from Hugging Face if None
        help="The path of the stored MLX BERT weights (npz file), or None to download from Hugging Face.",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="This is an example of BERT working in MLX",
        help="The text to generate embeddings for.",
    )
    args = parser.parse_args()
    run(args.bert_model, args.mlx_model, args.text)
