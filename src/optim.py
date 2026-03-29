import numpy as np

# implemented as in shown in https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
class Adam():
    def __init__(self, parameters, lr : float, b1 : float, b2 : float):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.parameters = parameters
        self.t = 1

        # storing prevs
        self.momentum = [np.zeros_like(p.params) for p in self.parameters]
        self.velocity = [np.zeros_like(p.params) for p in self.parameters]


    # step in the direction of calculated gradients
    def step(self):
        for idx, p in enumerate(self.parameters):
            try:
                if p.grad.shape != p.params.shape:
                    p.grad = p.grad.squeeze(0)
                assert p.grad.shape == p.params.shape, f"shape dont match in grad: {p.grad.shape} and params : {p.params.shape}"
                
                v = self.b2 * self.velocity[idx] + (1 - self.b2) * p.grad * p.grad
                m = self.b1 * self.momentum[idx] + (1 - self.b1) * p.grad

                m_cap = m / (1 - self.b1 ** self.t)
                v_cap = v / (1 - self.b2 ** self.t)

                # element wise division
                delta = self.lr * m_cap / (v_cap ** 0.5 + 1e-8)
                p.params -= delta

                self.velocity[idx] = v
                self.momentum[idx] = m
                self.t += 1
            except AssertionError as e:
                print(e)

    def zero_grad(self):
        for p in self.parameters:
            p.grad.fill(0)








    

