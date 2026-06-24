import numpy as np
import cupy as cp
import os
import random

class DataLoader:
    def __init__(self, data_dir, batch_size, max_seq_len):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        # load shard paths
        self.shards = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".bin")
        ]

        # memory-map all shards
        self.data = [
            np.memmap(shard, dtype=np.uint16, mode='r')
            for shard in self.shards
        ]

    def __iter__(self):
        return self.get_batch()

    def get_batch(self):
        while True:
            x_batch = []
            y_batch = []

            for _ in range(self.batch_size):

                # pick random shard
                shard = random.choice(self.data)

                # pick random start
                idx = random.randint(0, len(shard) - self.max_seq_len - 1)

                x = shard[idx : idx + self.max_seq_len]
                y = shard[idx + 1 : idx + self.max_seq_len + 1]

                x_batch.append(x)
                y_batch.append(y)

            # batch on cpu
            x_batch = np.stack(x_batch)
            y_batch = np.stack(y_batch)

            # move entire batch at once to gpu
            yield cp.asarray(x_batch, order='C'), cp.asarray(y_batch, order='C')