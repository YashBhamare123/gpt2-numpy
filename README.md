# GPT-2 from Scratch — NumPy Only

A complete, from-scratch implementation of the GPT-2 transformer architecture using nothing but NumPy for the
forward pass, backward pass, and Adam optimizer. No PyTorch. No TensorFlow. No Jax. Every gradient is derived
by hand and implemented manually. The model trains on NVIDIA H100 GPUs via [Modal](https://modal.com) on the
FineWeb-Edu dataset (1 billion tokens), with GPU acceleration provided by CuPy as a drop-in NumPy replacement.

## Why This Exists

I think gradient computation in deep learning frameworks is a leaky abstraction. You call
`loss.backward()` and gradients appear, but the moment something goes wrong (a NaN in your loss,
an exploding gradient, a shape mismatch deep in some custom layer) you need to understand what's
actually happening underneath. And at that point the abstraction hasn't saved you any complexity,
it's just delayed when you had to deal with it.

So I built this to force myself through every gradient derivation by hand. Every layer in this
project (Linear, LayerNorm, MultiHeadAttention, Softmax, Embeddings, CrossEntropyLoss) has
its backward pass worked out from the math and written as raw NumPy. No autograd, no framework
doing it for me.

## Architecture

The model follows the GPT-2 architecture:

```
                      Input Tokens
                           │
                           ▼
                 ┌───────────────────┐
                 │  Token Embeddings │
                 └────────┬──────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │  Positional Embeddings  │
              └───────────┬─────────────┘
                          │
          ┌───────────────▼───────────────┐
          │       Transformer x N         │
          │                               │
          │   ┌─────────────────────┐     │
          │   │     LayerNorm       │     │
          │   └──────────┬──────────┘     │
          │              ▼                │
          │   ┌─────────────────────┐     │
          │   │  Multi-Head Causal  │     │
     x ──►│   │  Self-Attention     │     │
     │    │   └──────────┬──────────┘     │
     │    │              ▼                │
     │    │   ┌─────────────────────┐     │
     │    │   │  Output Projection  │     │
     │    │   └──────────┬──────────┘     │
     │    │              │                │
     └────│──────► (+) ◄─┘  residual      │
          │         │                     │
          │   ┌─────▼───────────────┐     │
          │   │     LayerNorm       │     │
          │   └──────────┬──────────┘     │
          │              ▼                │
          │   ┌─────────────────────┐     │
     x ──►│   │   FFN               │     │
     │    │   │  Linear ► ReLU      │     │
     │    │   │  Linear             │     │
     │    │   └──────────┬──────────┘     │
     │    │              │                │
     └────│──────► (+) ◄─┘  residual      │
          │         │                     │
          └─────────┼─────────────────────┘
                    │
                    ▼
          ┌───────────────────┐
          │   Final Linear    │
          │  (weight-tied)    │
          └────────┬──────────┘
                   │
                   ▼
          ┌───────────────────┐
          │ Cross-Entropy Loss│
          └───────────────────┘
```

Default configuration:

| Parameter          | Value |
|--------------------|-------|
| Embedding Dim      | 384   |
| Attention Heads    | 6     |
| Transformer Layers | 6     |
| Max Sequence Length | 256   |
| Vocab Size         | 65 (char-level) / 50,257 (GPT-2 BPE) |
| Dropout            | 0.2   |

## What Is Implemented by Hand

There is no automatic differentiation anywhere in this codebase. Every layer has a `forward` and a
`backward` method, both written by hand. Everything lives in [nn.py](src/nn.py).

### Forward Pass

All tensor operations use `np.einsum` directly.

#### Linear ([forward](src/nn.py#L54-L60))

Multiplies input by a weight matrix and adds a bias.

#### Multi-Head Attention ([forward](src/nn.py#L272-L300))

Projects the input into Q, K, V via three linear layers, reshapes into heads, computes scaled dot-product
attention with a causal mask, applies softmax, aggregates the values, and concatenates the heads back
together.

#### Softmax ([forward](src/nn.py#L110-L116))

Subtracts the row-wise maximum before exponentiating to avoid overflow, then normalizes.

#### LayerNorm ([forward](src/nn.py#L210-L229))

Computes mean and variance across the feature dimension, normalizes, and applies a learnable scale and
shift. Caches the inverse standard deviation for the backward pass.

#### Embeddings ([forward](src/nn.py#L150-L152))

Looks up rows in a learnable weight table by token index.

#### Positional Embeddings ([forward](src/nn.py#L175-L180))

Adds a learnable position table to the token embeddings.

#### Cross-Entropy Loss ([forward](src/nn.py#L436-L444))

Computes log-softmax and negative log-likelihood in one pass using the log-sum-exp trick for
numerical stability.

#### FFN ([forward](src/nn.py#L376-L387))

Two linear layers with a ReLU and optional dropout in between.

#### Transformer Block ([forward](src/nn.py#L412-L420))

Pre-norm attention with a residual connection, followed by pre-norm FFN with another residual connection.

### Backward Pass

Each backward method takes the upstream gradient and returns the downstream gradient. Parameter gradients
are stored on the layer and picked up by the optimizer.

The gradient flows through the full model in reverse:

```
          ┌───────────────────┐
          │ Cross-Entropy Loss │
          └────────┬──────────┘
                   │  dL/d(logits) = softmax(logits) - one_hot(labels)
                   ▼
          ┌───────────────────┐
          │   Final Linear    │  ◄── gradient for weights, bias, and input
          └────────┬──────────┘
                   │
          ┌────────▼──────────────────────┐
          │       Transformer x N         │
          │                               │
     dL ──│──────► (+)                    │
     │    │         │                     │
     │    │   ┌─────▼───────────────┐     │
     │    │   │   FFN backward      │     │
     │    │   └──────────┬──────────┘     │
     │    │              ▼                │
     │    │   ┌─────────────────────┐     │
     │    │   │ LayerNorm backward  │     │
     │    │   └──────────┬──────────┘     │
     │    │              │                │
     └────│──────► (+) ◄─┘  residual      │
          │         │                     │
     dL ──│──────► (+)                    │
     │    │         │                     │
     │    │   ┌─────▼───────────────┐     │
     │    │   │  Output Projection  │     │
     │    │   │     backward        │     │
     │    │   └──────────┬──────────┘     │
     │    │              ▼                │
     │    │   ┌─────────────────────┐     │
     │    │   │  Attention backward │     │
     │    │   └──────────┬──────────┘     │
     │    │              ▼                │
     │    │   ┌─────────────────────┐     │
     │    │   │ LayerNorm backward  │     │
     │    │   └──────────┬──────────┘     │
     │    │              │                │
     └────│──────► (+) ◄─┘  residual      │
          │         │                     │
          └─────────┼─────────────────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │ Positional Embeddings  │  ◄── accumulate into position table
       └───────────┬────────────┘
                   ▼
       ┌────────────────────────┐
       │    Token Embeddings    │  ◄── scatter into embedding rows
       └────────────────────────┘
```

#### Linear ([backward](src/nn.py#L62-L79))

Produces three gradients: one for the input, one for the weight matrix, one for the bias.
The weight and bias gradients are summed over all batch and sequence positions.

#### Multi-Head Attention ([backward](src/nn.py#L303-L328))

The hardest backward in the project. Reshapes the upstream gradient back into per-head form, then
walks it backward through the value aggregation, softmax, scaled dot-product, and each of the
Q/K/V projections. The three input gradients are summed since Q, K, V all came from the same input.

#### Softmax ([backward](src/nn.py#L119-L124))

Uses the identity that the vector-Jacobian product of softmax can be written with just element-wise
ops and one dot product per row, so the full Jacobian never needs to be materialized.
A [Jacobian-based variant](src/nn.py#L95-L104) is also included for reference.

#### LayerNorm ([backward](src/nn.py#L232-L251))

The tricky one. Since the mean and variance depend on every element of the input, the gradient has
three terms: the direct term through the affine transform, a mean correction term, and a variance
correction term.

#### Embeddings ([backward](src/nn.py#L155-L158))

No downstream gradient (tokens are discrete). Scatters the upstream gradient into the correct rows
of the embedding table using `np.add.at`.

#### Positional Embeddings ([backward](src/nn.py#L184-L187))

Passes the gradient through (the forward is an addition) and accumulates into the position table.

#### Cross-Entropy Loss ([backward](src/nn.py#L447-L451))

Produces `softmax(logits) - one_hot(labels)`, divided by the number of predictions.
This is the starting gradient for the whole backward pass.

#### Transformer Block ([backward](src/nn.py#L423-L429))

Walks the gradient back through the FFN, LayerNorm, attention, and both residual connections.
The residual paths mean the upstream gradient gets added in at each skip connection.

### Adam Optimizer ([step](src/optim.py#L18-L39))

Written from the definition in [Kingma & Ba (2015)](https://arxiv.org/abs/1412.6980).
Keeps a momentum buffer (moving average of gradients) and a velocity buffer (moving average of
squared gradients) for each parameter. Both are bias-corrected every step. The update divides
corrected momentum by the square root of corrected velocity, giving adaptive per-element learning rates

## Training on H100 GPUs

The project includes a full cloud training pipeline using [Modal](https://modal.com) to run on NVIDIA H100 GPUs.
The CUDA backend swaps NumPy for [CuPy](https://cupy.dev), which provides an identical array API but executes
operations on the GPU. The entire `src/cuda/` directory mirrors `src/` with CuPy as the backend. The neural
network logic, gradient computations, and optimizer remain structurally identical.

### Cloud Training Pipeline (`train_modal.py`)

- **Dataset**: FineWeb-Edu (first 1B tokens), tokenized with the GPT-2 BPE tokenizer (tiktoken, vocab size 50,257)
- **Infrastructure**: Modal serverless GPU (H100), persistent volume for dataset caching and checkpoints
- **Batch Size**: 16 sequences of 512 tokens each (8,192 tokens per step)
- **Monitoring**: TensorBoard logging of training loss, validation loss, tokens/second, and learning rate,
  served via a Modal web endpoint
- **Checkpointing**: Model saved every 1,000 steps to persistent storage
- **Validation**: 16-step validation sweep over a held-out partition, evaluated every 100 training steps

To launch training:

```bash
modal run train_modal.py
```

To view TensorBoard:

```bash
modal serve train_modal.py
```

## Project Structure

```
.
|-- src/
|   |-- nn.py              # All layers: Linear, Softmax, LayerNorm, MultiHeadAttention,
|   |                      #   Embeddings, FFN, Transformer, CrossEntropyLoss, Pipeline
|   |-- optim.py           # Adam optimizer
|   |-- model.py           # GPT-2 architecture assembly and config
|   |-- loader.py          # NumPy data loader
|   |-- utils.py           # State dict serialization
|   |-- cuda/
|       |-- nn.py          # CuPy (GPU) mirror of nn.py
|       |-- optim.py       # CuPy (GPU) mirror of optim.py
|       |-- model.py       # CuPy (GPU) model assembly
|       |-- loader.py      # Sharded data loader with CPU to GPU transfers
|
|-- train.py               # Local training on Tiny Shakespeare (NumPy, CPU)
|-- train_cuda.py          # Local training on Tiny Shakespeare (CuPy, GPU)
|-- train_modal.py         # Cloud training on FineWeb-Edu (H100 via Modal)
|-- pyproject.toml         # Project metadata and dependencies
```

## Running Locally

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### CPU Training (Tiny Shakespeare)

```bash
uv run train.py
```

This downloads the Tiny Shakespeare dataset, builds a character-level tokenizer (65 tokens), constructs a
6-layer GPT-2, and trains with Adam (lr=1e-4, beta1=0.9, beta2=0.98) for 10 epochs.

### GPU Training (Local, CuPy)

Requires an NVIDIA GPU with CUDA and CuPy installed:

```bash
uv run train_cuda.py
```

## Design Decisions

**Einsum everywhere.** Nearly every matrix operation is expressed as an `np.einsum` call. This makes the
tensor contractions explicit and self-documenting. You can read the subscript string to understand exactly
which dimensions are being contracted, broadcast, or preserved. It also makes the correspondence between
the forward and backward passes immediately visible.

**Two softmax implementations.** `JacobianSoftmax` materializes the full Jacobian matrix and contracts it
with the upstream gradient. `Softmax` uses the algebraically equivalent but far more memory-efficient
formulation `s * (dy - sum(s * dy))`. Both are included: the Jacobian version for pedagogical clarity,
the efficient version for actual training.

**Weight tying.** The final linear projection shares its weight matrix (transposed) with the token embedding
layer, following the original GPT-2 design. This reduces parameter count and improves training dynamics.

**GPT-2 initialization.** All weights are initialized from `N(0, 0.02)` following the GPT-2 paper. Biases
are initialized to zero.

**No external autograd.** The entire point. If you want to understand what `loss.backward()` actually does,
read `nn.py`.

## Acknowledgments

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al.
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), Radford et al. (GPT-2)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), Kingma & Ba
- [Andrej Karpathy's Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), HuggingFace
