import numpy as np

class Operation:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, d_Ly):
        raise NotImplementedError


class GradTensor:
    def __init__(self, params, grad):
        self.params = params
        self.shape = params.shape
        self.grad = grad

    def _zero_grad(self):
        self.grad = np.zeros_like(self.params)

class GradLayer(Operation):
    def parameters(self):

        # returning all parameters in a list (either a grad layer or a grad tensor)
        params = []
        for att_name, att_value in self.__dict__.items():
            if isinstance(att_value, GradLayer):
                params.extend(att_value.parameters())
            elif isinstance(att_value, GradTensor):
                params.append(att_value.params)
        
        return params


class Linear(GradLayer):
    def __init__(self, in_features, out_features, bias = True):

        # He uniform initialization schema
        params = np.random.uniform(-np.sqrt(6/in_features), np.sqrt(6/in_features), size = (in_features, out_features))
        if bias:
            bias_ts = np.zeros((in_features, 1))
            self.bias = GradTensor(bias_ts, None)
        else:
            self.bias = None

        # all weights are grad tensors
        self.weight = GradTensor(params, None)

    
    def forward(self, x):
        self.x = x
        out = x @ self.weight
        if self.bias:
            out += self.bias
        
        return out

    def backward(self, d_Ly):
        d_Lx = d_Ly @ np.transpose(self.weight.params)
        d_Lw = np.transpose(self.x) @ d_Ly

        # update gradient as its not used elsewhere
        if self.bias:
            d_Lb = np.sum(d_Ly, axis=0, keepdims=True)
            self.bias.grad = d_Lb

        # we return downstream gradient
        return d_Lx


class JacobianSoftmax(Operation):
    # x : (B, H, S, D)
    def forward(self, x):
        self.x = x
        # subtracting the maximum for numerical stability
        out = np.exp(x - np.max(x, axis= - 1, keepdims= True))
        scale = np.sum(out, axis = -1, keepdims= True)
        out = out / scale
        self.s = out
        return out

    # out : (B, H, S, D)
    def backward(self, d_Ly):
        # all elements excect diagonal are -SiSj,
        s = np.expand_dims(self.s, axis = -1)
        out = - 1 * (s @ np.permute_dims(s, axes = [0, 1, 3, 2]))

        # diagonal is Si - SiSi
        eye = np.eye(self.s.shape[-1])
        out += np.einsum("...i, ij -> ...ij", self.s, eye)
        out = np.einsum("...ij, ...j -> ...i", out, d_Ly)
        return out

class Softmax(Operation):
    # x : (B H S D)
    def forward(self, x):
        self.x = x
        out = np.exp(x - np.max(x, axis= - 1, keepdims= True))
        scale = np.sum(out, axis = -1, keepdims= True)
        out = out / scale
        self.s = out
        return out
    
    # d_Ly : (B H S D)
    def backward(self, d_Ly):
        const = np.einsum('...i, ...j -> ...', self.s, d_Ly)
        offset_d_Ly = d_Ly - const
        out = np.einsum('...i, ...i -> ...i', self.s, offset_d_Ly)
        return out


class ReLU(Operation):
    def forward(self, x):
        self.x = x
        out = np.vectorize(lambda a : np.max(a, 0))(x)
        return out

    def backward(self, d_Ly):
        jac = np.vectorize(lambda a : 1 if max(a, 0) > 0 else 0)(self.x)
        return np.einsum('...i, ...i -> ...i', d_Ly, jac)


class Embeddings(GradLayer):
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        params = np.random.uniform(-np.sqrt(6/vocab_size), np.sqrt(6/vocab_size), size = (vocab_size, embedding_dim))
        self.weight = GradTensor(params, None)
    
    def forward(self, x):
        self.x = x
        return self.weight.params[x]

    # return nothing as this is the last layer (before this there are just tokens)
    def backward(self, d_Ly):
        self.weight.grad = np.zeros_like(self.weight.params)
        np.add.at(self.weight.grad, self.x, d_Ly)
        return None


# we chose to use learnable positional embeddings here
class PositionalEmbeddings(GradLayer):
    def __init__(self, max_seq_length, embedding_dim):
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim

        params = np.random.uniform(-np.sqrt(6/max_seq_length), np.sqrt(6/max_seq_length), size = (max_seq_length, embedding_dim))
        self.weight = GradTensor(params, None)
    
    # x : (B N D)
    def forward(self, x):
        seq_len = x.shape[-2]
        self.seq_len = seq_len
        assert self.max_seq_length >= seq_len
        self.x = x
        return x + self.weight.params[:seq_len][None, :, :]
    
    # d_ly : (B N D)
    # return d_Ly as the forward pass is an addition operation
    def backward(self, d_Ly):
        self.weight.grad = np.zeros_like(self.weight.params)
        np.add.at(self.weight.grad, np.arange(0, self.seq_len), np.einsum('bnd -> nd', d_Ly))
        return d_Ly



# implemented as shown in https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
class LayerNorm(GradLayer):
    def __init__(self, dim, element_wise_affine = True):
        params = np.random.uniform(-np.sqrt(6/dim), np.sqrt(6/dim), size = (dim))
        bias = np.random.uniform(-np.sqrt(6/dim), np.sqrt(6/dim), size = (dim))

        # weight and bias both are common across seq length and batch dim
        if element_wise_affine:
            self.weight = GradTensor(params, None)
            self.bias = GradTensor(bias, None)
        self.element_wise_affine = element_wise_affine
        self.dim = dim

    # x : (B N D)
    def forward(self, x):
        assert x.shape()[-1] == self.dim 
        self.x = x
        mean = np.einsum('...j -> ...', x) / self.dim
        u = x - np.expand_dims(mean, axis = -1)
        v = np.einsum('...i -> ...', u**2) / self.dim

        # add a small constant for numerical stability
        e = 1e-5
        sd = np.sqrt(v + e)

        # store 1/sd as division is expensive
        self.inverse_sd = 1/sd
        out = u / np.expand_dims(sd, axis = -1)

        if (self.element_wise_affine):
            scaled_out = np.einsum('...i, j -> ...i',out, self.weight.params)
            out = scaled_out + np.expand_dims(self.bias.params, axis = (0, 1))

        return out

    # simplify the jacobian expr and implement the mem efficient backward pass
    # d_Ly : (B N D)
    def backward(self, d_Ly):
        mean = np.einsum('...j -> ...', self.x) /self.dim
        u = self.x - np.expand_dims(mean, axis = -1)

        if self.element_wise_affine:
            d_Lw = np.einsum('...i, ...j -> ...i', u, d_Ly)

            # sum over the batch and seq len dim
            d_Lw = np.einsum('bnd -> d', d_Lw)
            self.weight.grad = d_Lw

            d_Lb = np.ones_like(d_Lw)
            self.bias.grad = d_Lb
        
        d = np.einsum('...i, j -> ...i', d_Ly, self.weight.params)
        mean_d = np.einsum('...i -> ...', d) / self.dim
        scaled_u = u * self.inverse_sd
        tmp = scaled_u * np.expand_dims(np.einsum('...i, ...j -> ...', scaled_u, d), axis = -1) / self.dim
        gradient = self.inverse_sd * (d - np.expand_dims(mean_d, axis = -1) - tmp)
        return gradient



















