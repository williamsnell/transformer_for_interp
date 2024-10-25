# Transformer from scratch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float, Int
from typing import Literal
import einops
from dataclasses import dataclass

# training
from transformers import AutoTokenizer
from datasets import load_dataset

from tqdm import tqdm
from itertools import product

t.set_printoptions(precision=2, linewidth=160)


Activation = Literal["relu"]

@dataclass
class TransformerConfig:
    n_layers: int
    d_vocab: int
    d_model: int
    n_heads: int
    d_head: int
    d_mlp: int
    n_mlp_layers: int
    mlp_activation: Activation
    context_length: int


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.cfg = cfg

        # TODO: change these from matmul to indexing
        self.embed = Embedding(cfg.d_vocab, cfg.d_model)
        self.pos_embed = Embedding(cfg.context_length, cfg.d_model)

        self.blocks = nn.Sequential(
                *[TransformerBlock(cfg)
                  for _ in range(cfg.n_layers)]
                )

        self.ln_final = nn.RMSNorm(normalized_shape=cfg.d_model, elementwise_affine=False)

        self.unembed = nn.Linear(cfg.d_model, cfg.d_vocab)

        # training parameters
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = t.optim.AdamW(self.parameters())

    def forward(self, x: Int[Tensor, "... seq"]
                ) -> Float[Tensor, "... seq d_vocab"]:
        seq_len = x.shape[-1]

        residual_stream = self.embed(x) + self.pos_embed(t.arange(seq_len))
        final_residual_stream = self.blocks(residual_stream)

        normalized_res = self.ln_final(final_residual_stream)
        
        # what kind of output do we want? Should we softmax here,
        # or just return the unnormalized results (e.g. if we're
        #                                          argmaxing anyway)
        return self.unembed(normalized_res)

    def calc_loss(self, x: Int[Tensor, "... seq"]) -> Tensor:
        prediction = self(x[..., :-1]) # exclude the last token
                                   # since we can't evaluate
                                   # its prediction

        # we use tokens[0 : -1] to predict tokens[1 : end]    
        loss = self.loss(input=prediction, target=x[1:])

        # we may actually want to sum or do something else
        return loss.mean()

    def train(self, batch_size: int, epochs: int, dataset: Tensor):
        token_chunk_size = batch_size * self.cfg.context_length

        for epoch, batch in tqdm(product(
                range(epochs), 
                range(len(dataset) // token_chunk_size))):
            tokens = dataset[batch * token_chunk_size : (batch + 1) * token_chunk_size]
            tokens = t.tensor(tokens).reshape(batch_size, self.cfg.context_length)
                
            self.optimizer.zero_grad()
            loss = self.calc_loss(tokens)
            print(f"{loss=}")
            loss.backward()

            self.optimizer.step()



class Embedding(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(nn.init.kaiming_normal_(t.empty(input_size, output_size)))
        self.bias = nn.Parameter(t.zeros(output_size))

    def forward(self, x):
        return self.weight[x] + self.bias


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()

        self.cfg = cfg

        self.attention_block = AttentionBlock(cfg.d_model, cfg.n_heads, cfg.d_head)

        if cfg.d_mlp > 0:
            self.mlp = MLP(cfg.d_model, cfg.d_mlp, cfg.n_mlp_layers, cfg.mlp_activation)
        else:
            self.mlp = None

    def forward(self, x):
        res_interm = self.attention_block(x)

        if self.mlp is not None:
            res_final = self.mlp(res_interm)
        else:
            res_final = res_interm

        return res_final


class AttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head

        self.ln = nn.RMSNorm(normalized_shape=d_model, elementwise_affine=False)

        # need to batch appropriately to get (batch, seq, qkv, n_heads, d_head)
        self.QKV = nn.Linear(d_model, 3 * d_head * n_heads)
        self.O = nn.Linear(d_head * n_heads, d_model) 
    
    def forward(self, x: Float[Tensor, "... seq d_model"]
                ) -> Float[Tensor, "... seq d_model"]:
        normalized = self.ln(x)

        qkv = self.QKV(normalized) # [(QKV) * d_head * n_heads]
        q, k, v = einops.rearrange(qkv, 
                                   "... seq (n_heads d_head n_qkv) -> ... seq n_heads d_head n_qkv", 
                                   d_head=self.d_head,
                                   n_heads=self.n_heads,
                                   n_qkv=3).unbind(-1)

        attention_dot = einops.einsum(q, k,
                          "... qseq n_heads d_head, ... kseq n_heads d_head -> ... n_heads qseq kseq")
        attention_masked = attention_dot + t.triu(-t.inf * t.ones(*attention_dot.shape[-3:]), diagonal=1)
        attention_score = attention_masked.softmax(dim=-2)

        z = einops.einsum(attention_score, v,
                          "... n_heads qseq kseq, ... kseq n_heads d_head -> ... qseq n_heads d_head")

        return self.O(z.flatten(-2, -1))




class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, n_mlp_layers, mlp_activation):
        super().__init__()
        
        if mlp_activation != "relu":
            raise ValueError("Only relu supported for now")

        self.activation_func = nn.ReLU()

        self.ln = nn.RMSNorm(normalized_shape=d_model, elementwise_affine=False)

        self.embed = nn.Linear(d_model, d_mlp)

        self.hidden = nn.Sequential(
                *[layer for layer in (nn.Linear(d_mlp, d_mlp), self.activation_func)
                  for _ in range(n_mlp_layers)]
                )

        self.unembed = nn.Linear(d_mlp, d_model)

    def forward(self, x):
        normalized = self.ln(x)
        mlp_stream_in = self.activation_func(self.embed(normalized))
        mlp_stream_out = self.hidden(mlp_stream_in)

        return self.unembed(mlp_stream_out)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    cfg = TransformerConfig(
            n_layers=2,
            d_vocab=vocab_size,
            d_model=64,
            n_heads=8,
            d_head=8,
            d_mlp=64*2,
            n_mlp_layers=1,
            mlp_activation='relu',
            context_length=1024
            )

    model = Transformer(cfg)

    print(model((t.rand(10) * vocab_size).int()))

    ds = load_dataset("stas/openwebtext-10k")

    def encode(examples):
        return tokenizer(examples['text'])

    ds = ds.map(encode, batched=True)

    model.train(epochs=5, batch_size=10, dataset=ds['train']['input_ids'])
