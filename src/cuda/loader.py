import cupy as np

class DataLoader:
    def __init__(self, dataset: np.ndarray, batch_size: int, max_seq_len: int):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        usable_len = (len(dataset) - 1) // batch_size * batch_size
        x = dataset[:usable_len]
        labels = dataset[1:usable_len + 1]
        self.x = x.reshape(batch_size, -1)
        self.labels = labels.reshape(batch_size, -1)
        self.num_chunks = self.x.shape[1] // max_seq_len

    def __iter__(self):
        return self.get_batch()

    def __len__(self):
        return self.num_chunks

    def get_batch(self):
        chunk_order = np.random.permutation(self.num_chunks)
        for chunk_idx in chunk_order:
            start = chunk_idx * self.max_seq_len
            end = start + self.max_seq_len
            x_batch = self.x[:, start:end]
            label_batch = self.labels[:, start:end]
            yield self.process(x_batch), self.process(label_batch)

    def process(self, x):
        return x