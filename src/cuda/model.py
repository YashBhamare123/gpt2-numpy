import src.cuda.nn as nn
import cupy as np
from dataclasses import dataclass

@dataclass
class GPT2Config:
    vocab_size : int = 65
    embed_dim : int = 384
    num_head : int = 6
    num_layers : int = 6
    dropout : bool  = True
    dropout_p : float  = 0.2
    max_sequence_len : int = 256


def get_gpt2(config : GPT2Config = GPT2Config()):
    model = nn.Pipeline()
    model.register_module(nn.Embeddings(config.vocab_size, config.embed_dim), 'embedding')
    model.register_module(nn.PositionalEmbeddings(config.max_sequence_len, config.embed_dim), 'positional_embedding')

    for idx in range(config.num_layers):
        model.register_module(nn.Transformer(config.embed_dim, config.num_head, config.max_sequence_len), f'transformer_{idx}')

    # weight tying
    linear = nn.Linear(config.embed_dim, config.vocab_size)
    linear.weight.params = np.transpose(model.layers['embedding'].weight.params)

    model.register_module(linear, 'final_layer')
    return model

    