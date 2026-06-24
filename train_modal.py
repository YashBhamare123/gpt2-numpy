import os

import modal


app = modal.App("fineweb-edu-training")
volume = modal.Volume.from_name("fineweb-edu-training", create_if_missing=True)
volume_path = "/vol"
token_path = f"{volume_path}/fineweb_edu_first_1b_gpt2_tokens.bin"
partial_token_path = f"{token_path}.partial"
checkpoint_path = f"{volume_path}/fineweb_edu_first_1b_model"
tensorboard_path = f"{volume_path}/tensorboard"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .pip_install(
        "cupy-cuda12x>=13.0.0",
        "datasets>=3.0.0",
        "tensorboard>=2.18.0",
        "tensorboardX>=2.6.0",
        "tiktoken>=0.8.0",
        "tqdm>=4.0.0",
    )
    .add_local_python_source("src")
)


def download_dataset():
    import numpy as np
    import tiktoken
    from datasets import load_dataset
    from tqdm import tqdm

    if os.path.exists(token_path) and os.path.getsize(token_path) == 2_000_000_000:
        return

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    tokens = np.memmap(partial_token_path, dtype=np.uint16, mode="w+", shape=(1_000_000_000,))
    token_count = 0

    for sample in tqdm(dataset, desc="tokenizing FineWeb-Edu", unit="documents"):
        encoded = tokenizer.encode_ordinary(sample["text"])
        encoded.append(tokenizer.eot_token)
        encoded = encoded[:len(tokens) - token_count]
        tokens[token_count : token_count + len(encoded)] = encoded
        token_count += len(encoded)

        if token_count == len(tokens):
            break

    tokens.flush()
    del tokens
    assert token_count == 1_000_000_000, f"only downloaded {token_count} tokens"
    os.replace(partial_token_path, token_path)
    volume.commit()


def get_batch(tokens, start, batch_size, max_sequence_len, np):
    import numpy

    batch = numpy.asarray(
        tokens[start : start + batch_size * max_sequence_len + 1],
        dtype=numpy.int32,
    )
    batch = np.asarray(batch)
    x = batch[:-1].reshape(batch_size, max_sequence_len)
    labels = batch[1:].reshape(batch_size, max_sequence_len)
    return x, np.expand_dims(labels, -1)


def validate(pipe, cross_entropy, tokens, validation_start, batch_size, max_sequence_len, np):
    validation_loss = 0
    validation_steps = 16
    tokens_per_step = batch_size * max_sequence_len

    pipe.eval()
    for step in range(validation_steps):
        start = validation_start + step * tokens_per_step
        x, labels = get_batch(tokens, start, batch_size, max_sequence_len, np)
        out = pipe(x)
        validation_loss += cross_entropy.forward(out, labels).item()

    pipe.train()
    return validation_loss / validation_steps


@app.function(
    image=image,
    gpu="H100",
    timeout=24 * 60 * 60,
    volumes={volume_path: volume},
)
def train():
    import cupy as np
    import numpy
    import time
    from tensorboardX import SummaryWriter
    from tqdm import tqdm

    import src.cuda.nn as nn
    import src.cuda.optim as optim
    from src.cuda.model import GPT2Config, get_gpt2

    download_dataset()

    batch_size = 16
    max_sequence_len = 512
    epochs = 1
    validation_tokens = 10_000_000
    validation_every = 100
    tokens = numpy.memmap(token_path, dtype=numpy.uint16, mode="r")
    config = GPT2Config(vocab_size=50257, max_sequence_len=max_sequence_len)
    pipe = get_gpt2(config)
    for parameter in pipe.parameters():
        assert parameter.params.dtype == np.float32

    optimizer = optim.Adam(pipe.parameters(), 1e-4, 0.9, 0.98)
    cross_entropy = nn.CrossEntropyLoss()
    tokens_per_step = batch_size * max_sequence_len
    validation_start = len(tokens) - validation_tokens
    total_steps = (validation_start - 1) // tokens_per_step
    writer = SummaryWriter(tensorboard_path)
    start_time = time.perf_counter()

    for epoch in range(epochs):
        pipe.train()
        for step in tqdm(range(total_steps), desc=f"epoch {epoch}"):
            start = step * tokens_per_step
            x, labels = get_batch(tokens, start, batch_size, max_sequence_len, np)

            out = pipe(x)
            out_loss = cross_entropy.forward(out, labels)
            d_Ly = cross_entropy.backward()
            pipe.backward(d_Ly)

            optimizer.step()
            optimizer.zero_grad()

            elapsed = time.perf_counter() - start_time
            token_count = (step + 1) * tokens_per_step
            writer.add_scalar("training/loss", out_loss.item(), step)
            writer.add_scalar("training/tokens", token_count, step)
            writer.add_scalar("training/tokens_per_second", token_count / elapsed, step)
            writer.add_scalar("training/learning_rate", optimizer.lr, step)

            if step % validation_every == 0:
                validation_loss = validate(
                    pipe,
                    cross_entropy,
                    tokens,
                    validation_start,
                    batch_size,
                    max_sequence_len,
                    np,
                )
                writer.add_scalar("validation/loss", validation_loss, step)
                writer.flush()
                volume.commit()
                print(f"training loss : {out_loss}")
                print(f"validation loss : {validation_loss}")

            if step and step % 1_000 == 0:
                pipe.save(checkpoint_path)
                writer.flush()
                volume.commit()

    pipe.save(checkpoint_path)
    writer.flush()
    writer.close()
    volume.commit()
    print("Done with training")


@app.function(
    image=image,
    scaledown_window=30 * 60,
    volumes={volume_path: volume},
)
@modal.web_server(6006, label="fineweb-edu-tensorboard")
def tensorboard():
    import subprocess

    volume.reload()
    subprocess.Popen([
        "tensorboard",
        "--logdir", tensorboard_path,
        "--host", "0.0.0.0",
        "--port", "6006",
        "--reload_interval", "15",
    ])


@app.local_entrypoint()
def main():
    train.remote()
