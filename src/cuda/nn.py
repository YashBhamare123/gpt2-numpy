import cupy as np
import numpy 

from src.utils import create_state_dict

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
        self.grad = np.zeros_like(self.params, dtype=np.float32)

class GradLayer(Operation):
    def parameters(self):

        # returning all parameters in a list (either a grad layer or a grad tensor)
        params = []
        for att_name, att_value in self.__dict__.items():
            if isinstance(att_value, GradLayer):
                params.extend(att_value.parameters())
            elif isinstance(att_value, GradTensor):
                params.append(att_value)
        
        return params


class Linear(GradLayer):
    def __init__(self, in_features, out_features, bias = True):

        # He uniform initialization schema
        # params = np.random.uniform(-np.sqrt(6/in_features), np.sqrt(6/in_features), size = (in_features, out_features)).astype(np.float32)

        # GPT2 Paper Schema
        params = np.random.normal(0, 0.02, size = (in_features, out_features)).astype(np.float32)
        if bias:
            bias_ts = np.zeros((out_features), dtype=np.float32)
            self.bias = GradTensor(bias_ts, None)
        else:
            self.bias = None

        # all weights are grad tensors
        self.weight = GradTensor(params, None)

    # x : (... N D)
    def forward(self, x):
        self.x = x
        out = np.einsum('...ij, jk -> ...ik', x, self.weight.params)
        if self.bias:
            out += self.bias.params
        
        return out
    # d_Ly : (... N D)
    def backward(self, d_Ly):
        d_Lx = np.einsum('...jk, ik -> ...ji', d_Ly, self.weight.params)
        d_Lw = np.einsum('...kj, ...ki -> ...ji', self.x, d_Ly)

        # update gradient as its not used elsewhere
        if self.bias:

            # flatten the gradient to accumutate it
            d_Lb = np.reshape(d_Ly, (-1, d_Ly.shape[-1]))
            d_Lb = np.sum(d_Lb, axis=0, keepdims=True)
            self.bias.grad = d_Lb
        
        d_Lw = np.reshape(d_Lw, (-1, d_Lw.shape[-2], d_Lw.shape[-1]))
        d_Lw = np.sum(d_Lw, axis = 0, keepdims= True)
        self.weight.grad = d_Lw

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
        eye = np.eye(self.s.shape[-1], dtype=np.float32)
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
        const = np.einsum('...i, ...i -> ...', self.s, d_Ly)
        offset_d_Ly = d_Ly - np.expand_dims(const, axis = -1)

        out = np.einsum('...i, ...i -> ...i', self.s, offset_d_Ly)
        return out



class ReLU(Operation):
    def forward(self, x):
        self.x = x
        out = np.maximum(x, 0)
        return out

    def backward(self, d_Ly):
        jac = np.vectorize(lambda a : 1 if max(a, 0) > 0 else 0)(self.x)
        return np.einsum('...i, ...i -> ...i', d_Ly, jac)



class Embeddings(GradLayer):
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # params = np.random.uniform(-np.sqrt(6/vocab_size), np.sqrt(6/vocab_size), size = (vocab_size, embedding_dim))
        # GPT2 Paper Schema
        params = np.random.normal(0, 0.02, size = (vocab_size, embedding_dim)).astype(np.float32)
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

        # params = np.random.uniform(-np.sqrt(6/max_seq_length), np.sqrt(6/max_seq_length), size = (max_seq_length, embedding_dim))

        # GPT2 Paper Schema
        params = np.random.normal(0, 0.02, size = (max_seq_length, embedding_dim)).astype(np.float32)
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
        # params = np.random.uniform(-np.sqrt(6/dim), np.sqrt(6/dim), size = (dim)).astype(np.float32)


        # GPT2 Paper Schema
        params = np.random.normal(0, 0.02, size = (dim)).astype(np.float32)
        bias = np.zeros_like(params).astype(np.float32)

        # weight and bias both are common across seq length and batch dim
        if element_wise_affine:
            
            self.weight = GradTensor(params, None)
            self.bias = GradTensor(bias, None)
        self.element_wise_affine = element_wise_affine
        self.dim = dim

    # x : (B N D)
    def forward(self, x):
        assert x.shape[-1] == self.dim 
        self.x = x
        mean = np.einsum('...j -> ...', x) / self.dim
        u = x - np.expand_dims(mean, axis = -1)
        v = np.einsum('...i -> ...', u**2) / self.dim

        # add a small constant for numerical stability
        e = 1e-5
        sd = np.sqrt(v + e)

        # store 1/sd as division is expensive
        self.inverse_sd = np.expand_dims(1/sd, axis= -1)
        out = u / np.expand_dims(sd, axis = -1)

        if (self.element_wise_affine):
            scaled_out = np.einsum('...i, i -> ...i',out, self.weight.params)
            out = scaled_out + np.expand_dims(self.bias.params, axis = (0, 1))

        return out

    # d_Ly : (B N D)
    def backward(self, d_Ly):
        mean = np.einsum('...j -> ...', self.x) /self.dim
        u = self.x - np.expand_dims(mean, axis = -1)
        scaled_u = u * self.inverse_sd

        if self.element_wise_affine:
            d_Lw = np.einsum('...i, ...i -> ...i', scaled_u, d_Ly)

            # sum over the batch and seq len dim
            d_Lw = np.einsum('bni -> i', d_Lw)
            self.weight.grad = d_Lw

            d_Lb = np.einsum('bni -> i', d_Ly)
            self.bias.grad = d_Lb
        
        d = np.einsum('...i, i -> ...i', d_Ly, self.weight.params)
        mean_d = np.einsum('...i -> ...', d) / self.dim
        tmp = scaled_u * np.expand_dims(np.einsum('...i, ...i -> ...', scaled_u, d), axis = -1) / self.dim
        gradient = self.inverse_sd * (d - np.expand_dims(mean_d, axis = -1) - tmp)
        return gradient



# implement multihead attention
class MultiHeadAttention(GradLayer):
    def __init__(self, embed_dim, num_heads, max_sequence_len, causal: bool = True):
        
        assert embed_dim % num_heads == 0, "embed dim is not divisible by num heads"
        self.query = Linear(embed_dim, embed_dim)
        self.key = Linear(embed_dim, embed_dim)
        self.value = Linear(embed_dim, embed_dim)

        self.dim = embed_dim
        self.num_heads = num_heads
        self.softmax = Softmax()

        self.attention_mask = np.triu(
            np.full((1, 1, max_sequence_len, max_sequence_len), -np.inf, dtype=np.float32),
            k = 1,
        )
        self.attention_scale = np.float32(1 / (embed_dim // num_heads) ** 0.5)
        self.causal = causal

    # x : (B N D) -> (B N D)
    def forward(self, x):
        B, N, D = x.shape
        self.x = x

        # multiheads after and before attention are the same thing (tiled matmuls)
        q = self.query.forward(x)
        k = self.key.forward(x)
        v = self.value.forward(x)

        q = np.permute_dims(np.reshape(q, (B, N, self.num_heads, D // self.num_heads)), (0, 2, 1, 3))
        k = np.permute_dims(np.reshape(k, (B, N, self.num_heads, D // self.num_heads)), (0, 2, 1, 3))
        v = np.permute_dims(np.reshape(v, (B, N, self.num_heads, D // self.num_heads)), (0, 2, 1, 3))

        self.q = q
        self.k = k

        qkT = np.einsum('...ij, ...kj -> ...ik', q, k) * self.attention_scale
        if (self.causal):
            qkT = qkT + self.attention_mask[:, :, :N, :N]

        qkT_softmax = self.softmax.forward(qkT)


        self.v = v
        self.qkT_softmax = qkT_softmax

        delta = np.einsum('...jk, ...ki -> ...ji', qkT_softmax, v)
        delta = np.reshape(np.permute_dims(delta, (0, 2, 1, 3)), (B, N, D))
        return delta
    
    # d_Ly : (B N D)
    def backward(self, d_Ly):
        B, N, D = d_Ly.shape

        # expand upstream gradient across heads
        d_Ly = np.permute_dims(np.reshape(d_Ly, (B, N, self.num_heads, D // self.num_heads)), (0, 2, 1, 3))
        
        d_L_qkT_softmax = np.einsum('...ij, ...kj -> ...ik', d_Ly, self.v)
        d_L_v = np.einsum('...ij, ...ik -> ...jk', self.qkT_softmax, d_Ly)

        d_L_qkT = self.softmax.backward(d_L_qkT_softmax * self.attention_scale)

        d_L_q = np.einsum('...ik, ...kj -> ...ij', d_L_qkT, self.k)
        d_L_k = np.einsum('...ki, ...kj -> ...ij', d_L_qkT, self.q)

        # remove the head dimension (concat across embed_dim)
        d_L_q = np.reshape(np.permute_dims(d_L_q, (0, 2, 1, 3)), (B, N, D))
        d_L_k = np.reshape(np.permute_dims(d_L_k, (0, 2, 1, 3)), (B, N, D))
        d_L_v = np.reshape(np.permute_dims(d_L_v, (0, 2, 1, 3)), (B, N, D))

        d_Lx_q = self.query.backward(d_L_q)
        d_Lx_k = self.key.backward(d_L_k)
        d_Lx_v = self.value.backward(d_L_v)

        # sum the contributions from q, k, v for the downstream gradient
        d_Lx = d_Lx_q + d_Lx_k + d_Lx_v
        return d_Lx


class Dropout(Operation):
    # x : (B N D)
    def __init__(self, p : float = 0.2):
        self.training = False
        self.p = p

    def forward(self, x):

        B, N, D = x.shape
        if self.training:
            mask = (np.random.random(size = (B, N, D)) > self.p).astype(np.float32)
            self.mask = mask

            # scale the outputs by expected value so that the sum of numbers stays in the same range
            return np.einsum('...ij, ...ij -> ...ij', x, mask) * (1/ (1 - self.p))
        else:
            return x
        
    # d_Ly : (B N D)
    def backward(self, d_Ly):

        # redundant if, but kept for clarity
        if (self.training):
            gradient = np.einsum('...ij, ...ij -> ...ij', d_Ly, self.mask) * (1/ (1 - self.p))
            return gradient

        else:
            return d_Ly


# Standard FFN Block
class FFN(GradLayer):
    def __init__(self, embed_dim, hidden_dim, dropout : bool = False, p : float = 0.2):
        self.linear1 = Linear(embed_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, embed_dim)
        self.relu= ReLU()
        if (dropout):
            self.dropout1 = Dropout(p)
            self.dropout2 = Dropout(p)
        else:
            self.dropout1 = None
            self.dropout2 = None

        self.p = p

    def forward(self, x):
        out = self.linear1.forward(x)       
        out = self.relu.forward(out)

        if self.dropout1:
            out = self.dropout1.forward(out)
        
        out = self.linear2.forward(out)
        if self.dropout2:
            out = self.dropout2.forward(out)
        
        return out


    def backward(self, d_Ly):
        if self.dropout2:
            d_Ly= self.dropout2.backward(d_Ly)

        gradient = self.linear2.backward(d_Ly)
        
        if self.dropout1:
            gradient = self.dropout1.backward(gradient)
        
        gradient = self.relu.backward(gradient)
        gradient = self.linear1.backward(gradient)
        return gradient


class Transformer(GradLayer):
    def __init__(self, embed_dim, num_heads, max_sequence_len):
        self.layernorm1 = LayerNorm(embed_dim)
        self.linear1 = Linear(embed_dim, embed_dim)
        self.attention1 = MultiHeadAttention(embed_dim, num_heads, max_sequence_len)
        self.layernorm2 = LayerNorm(embed_dim)
        self.ffn1 = FFN(embed_dim, embed_dim * 4)

    def forward(self, x):
        out = self.layernorm1.forward(x)
        out = self.attention1.forward(out)
        out = self.linear1.forward(out)
        out = x + out
        out_2 = self.layernorm2.forward(out)
        out_2 = self.ffn1.forward(out_2)
        out = out_2 + out
        return out

    # backprop thru residual pathway as well
    def backward(self, d_Ly):
        grad_1 = self.ffn1.backward(d_Ly)
        grad_1 = self.layernorm2.backward(grad_1)
        grad_2 = self.linear1.backward(d_Ly + grad_1)
        grad_2 = self.attention1.backward(grad_2)
        grad_2 = self.layernorm1.backward(grad_2)
        return d_Ly + grad_2 + grad_1
        

class CrossEntropyLoss:

    # x : (B S vocab_size)
    # labels : (B S 1)
    def forward(self, x, labels):
        B, N, vocab_size = x.shape
        self.x = x
        self.labels = labels
        out = np.take_along_axis(x, labels, axis =-1)
        x_max = np.max(x, axis=-1, keepdims=True)
        out = out - x_max - np.log(np.expand_dims(np.einsum('...i -> ...', np.exp(x - x_max)), axis= -1))
        out =  - np.einsum('bnd -> d', out) / (B * N)    # d is just 1 here
        return out

    # no upstream gradient as this is the final gradient layer
    def backward(self):
        B, N, V = self.x.shape
        o = Softmax().forward(self.x)
        batch_idx = np.arange(B)[:, None]
        token_idx = np.arange(N)[None, :]
        o[batch_idx, token_idx, self.labels.squeeze(-1)] -= 1
        return o / (B*N)


# model wrapper
class Pipeline:
    def __init__(self):
        self.layers = {}
        self.layer_order = []

    def register_module(self, layer, name):
        self.layers[name] = layer
        self.layer_order.append(name)


    def forward(self, x):
        for layer_name in self.layer_order:
            x = self.layers[layer_name].forward(x)
        return x


    def backward(self, d_Ly):
        for layer_name in reversed(self.layer_order):
            d_Ly = self.layers[layer_name].backward(d_Ly)
        return d_Ly


    def __call__(self, x):
        return self.forward(x)


    # true for training, false for eval
    def train(self):
        for layer in self.layers.values():
            self._set_mode(layer, True)


    def eval(self):
        for layer in self.layers.values():
            self._set_mode(layer, False)


    # recursive helper to set all modes to training
    def _set_mode(self, layer, mode):
        for attr_name, attr in layer.__dict__.items():
            if (attr_name == 'training'):
                layer.__dict__['training'] = mode

            elif isinstance(attr, GradLayer) or isinstance(attr, Operation):
                self._set_mode(attr, mode)
            

    # return all learnable parameters (ie GradTensors)
    def parameters(self):
        params = []
        for layer in self.layers.values():
            if isinstance(layer, GradLayer):
                params.extend(layer.parameters())
            elif isinstance(layer, GradTensor):
                params.append(layer)          
        return params


    def _load_layer(self, state_dict, layer, prev_name):
        for attr_name, attr_value in layer.__dict__.items():
            if isinstance(attr_value, GradLayer):
                layer.__dict__[attr_name] = self._load_layer(state_dict, attr_value, prev_name + '.' + attr_name)
            elif isinstance(attr_value, GradTensor):
                layer.__dict__[attr_name].params = state_dict[prev_name + '.' + attr_name]['params']
        
        return layer


    # load all layers matched by keys in from the state dict
    def load(self, state_dict):
        for layer_name, layer_val in self.layers.items():
            self.layers[layer_name] = self._load_layer(state_dict, layer_val, layer_name)
        
    def _create_layer_dict(self, layer : GradLayer, prev_name, state_dict : dict) -> dict:
        for attr_name, attr_value in layer.__dict__.items():
            if isinstance(attr_value, GradLayer):
                state_dict = self._create_layer_dict(attr_value, prev_name + "." + attr_name, state_dict)
            elif isinstance(attr_value, GradTensor):
                state_dict[prev_name + "." + attr_name] = {
                    "params" : attr_value.params,
                    "shape" : attr_value.shape
                }
        return state_dict

    # save all layer parameters and shapes
    def save(self, path : str):
        state_dict = {}
        for layer_name, layer_val in self.layers.items():
            if isinstance(layer_val, GradLayer):
                state_dict.update(self._create_layer_dict(layer_val, layer_name, state_dict))
        
        np.savez(path, **state_dict)


if __name__ == "__main__":
    pipe = Pipeline()

    inp = np.random.uniform(0, 1, (16, 20, 20))

    linear1 = Linear(20, 40)
    dropout1 = Dropout()
    linear2 = Linear(40, 20)
    linear3 = Linear(20, 20)
    attn1 = MultiHeadAttention(20, 2, 100)
    relu = ReLU()
    ff1 = FFN(20, 40, dropout= True)

    pipe.register_module(linear1, 'linear1')
    pipe.register_module(dropout1, 'dropout1')
    pipe.register_module(linear2, 'linear2')
    pipe.register_module(linear3, 'linear3')
    pipe.register_module(relu, 'relu')
    pipe.register_module(attn1, 'attn1')
    pipe.register_module(ff1, 'ff1')

    pipe.save('./model.npz')

    state_dict = create_state_dict('./model.npz')
    
    pipe.load(state_dict)
    pipe.eval()


    out = pipe(inp)
    print(pipe.layers['dropout1'].training)
    print(pipe.layers['ff1'].dropout2.training)
