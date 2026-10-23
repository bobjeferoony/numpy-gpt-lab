"""A byte-level causal transformer. All model math and gradients use NumPy."""
import numpy as np


def unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, n in enumerate(shape):
        if n == 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Tensor:
    def __init__(self, data, parents=(), backward=lambda g: None):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.parents, self._backward = parents, backward

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        def back(g):
            self.grad += unbroadcast(g, self.data.shape)
            other.grad += unbroadcast(g, other.data.shape)
        return Tensor(self.data + other.data, (self, other), back)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        def back(g):
            self.grad += unbroadcast(g * other.data, self.data.shape)
            other.grad += unbroadcast(g * self.data, other.data.shape)
        return Tensor(self.data * other.data, (self, other), back)

    def __matmul__(self, other):
        def back(g):
            self.grad += unbroadcast(g @ other.data.swapaxes(-1, -2), self.data.shape)
            other.grad += unbroadcast(self.data.swapaxes(-1, -2) @ g, other.data.shape)
        return Tensor(self.data @ other.data, (self, other), back)

    def transpose(self, axes):
        def back(g):
            self.grad += g.transpose(np.argsort(axes))
        return Tensor(self.data.transpose(axes), (self,), back)

    def reshape(self, shape):
        def back(g):
            self.grad += g.reshape(self.data.shape)
        return Tensor(self.data.reshape(shape), (self,), back)

    def take(self, indices):
        def back(g):
            np.add.at(self.grad, indices, g)
        return Tensor(self.data[indices], (self,), back)

    def relu(self):
        def back(g):
            self.grad += g * (self.data > 0)
        return Tensor(np.maximum(0, self.data), (self,), back)

    def norm(self):
        centered = self.data - self.data.mean(axis=-1, keepdims=True)
        inv = 1 / np.sqrt((centered ** 2).mean(axis=-1, keepdims=True) + 1e-5)
        y = centered * inv
        def back(g):
            self.grad += inv * (g - g.mean(axis=-1, keepdims=True)
                                - y * (g * y).mean(axis=-1, keepdims=True))
        return Tensor(y, (self,), back)

    def softmax(self):
        e = np.exp(self.data - self.data.max(axis=-1, keepdims=True))
        p = e / e.sum(axis=-1, keepdims=True)
        def back(g):
            self.grad += p * (g - (g * p).sum(axis=-1, keepdims=True))
        return Tensor(p, (self,), back)

    def backward(self):
        order, seen = [], set()
        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t.parents:
                visit(p)
            order.append(t)
        visit(self)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            t._backward(t.grad)


def cross_entropy(logits, targets, mask):
    z = logits.data.reshape(-1, 257)
    targets, mask = targets.ravel(), mask.ravel()
    shifted = z - z.max(axis=-1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    count = max(float(mask.sum()), 1)
    value = -(logp[np.arange(len(targets)), targets] * mask).sum() / count
    def back(g):
        dz = np.exp(logp)
        dz[np.arange(len(targets)), targets] -= 1
        logits.grad += (dz * mask[:, None] / count * g).reshape(logits.data.shape)
    return Tensor(value, (logits,), back)


class GPT:
    def __init__(self, width=48, context=192, heads=4, layers=2, seed=42):
        if width % heads:
            raise ValueError('Width must be divisible by heads')
        self.config = dict(width=width, context=context, heads=heads, layers=layers, seed=seed)
        self.p = {}
        rng = np.random.default_rng(seed)
        def weight(name, shape, kind='random'):
            data = (rng.normal(0, .02, shape) if kind == 'random' else
                    np.ones(shape) if kind == 'ones' else np.zeros(shape))
            self.p[name] = Tensor(data)
        weight('token', (257, width)); weight('position', (context, width))
        for i in range(layers):
            for n in ('n1', 'n2'):
                weight(f'{i}.{n}g', (width,), 'ones')
                weight(f'{i}.{n}b', (width,), 'zeros')
            for n, shape in [('qkv', (width, 3*width)), ('proj', (width, width)),
                             ('up', (width, 4*width)), ('down', (4*width, width))]:
                weight(f'{i}.{n}', shape)
                weight(f'{i}.{n}b', (shape[1],), 'zeros')
        weight('finalg', (width,), 'ones'); weight('finalb', (width,), 'zeros')
        weight('output', (width, 257))

    def forward(self, ids):
        b, t = ids.shape
        c, h = self.config['width'], self.config['heads']
        if t > self.config['context']:
            raise ValueError('Input exceeds model context')
        p = self.p
        x = p['token'].take(ids) + p['position'].take(np.arange(t))
        for i in range(self.config['layers']):
            prefix = str(i) + '.'
            def norm(y, n):
                return y.norm() * p[prefix+n+'g'] + p[prefix+n+'b']
            z = norm(x, 'n1') @ p[prefix+'qkv'] + p[prefix+'qkvb']
            # Separate q/k/v using a reshape and differentiable indexing.
            z = z.reshape((b, t, 3, h, c//h)).transpose((2, 0, 3, 1, 4))
            q, k, v = (z.take(j) for j in range(3))
            scores = (q @ k.transpose((0, 1, 3, 2))) * ((c//h)**-.5)
            mask = np.where(np.triu(np.ones((t, t)), 1), -1e9, 0)
            a = (scores + mask).softmax() @ v
            a = a.transpose((0, 2, 1, 3)).reshape((b, t, c))
            x = x + (a @ p[prefix+'proj'] + p[prefix+'projb'])
            z = (norm(x, 'n2') @ p[prefix+'up'] + p[prefix+'upb']).relu()
            x = x + (z @ p[prefix+'down'] + p[prefix+'downb'])
        return (x.norm() * p['finalg'] + p['finalb']) @ p['output']

    def zero_grad(self):
        for p in self.p.values():
            p.grad.fill(0)

    def save(self, path):
        np.savez_compressed(path, **{k: v.data for k, v in self.p.items()},
                            config=np.array(list(self.config.values())))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(**dict(zip(('width','context','heads','layers','seed'), map(int, data['config']))))
            for k in model.p:
                model.p[k].data[:] = data[k]
        return model

    def generate(self, code, tokens=160, temperature=0.0):
        prompt = list(('BUG:\n' + code + '\nFIX:\n').encode('utf-8'))
        if len(prompt) >= self.config['context']:
            raise ValueError('Code is too long for this model context; use a shorter function.')
        ids, output = prompt[:], []
        rng = np.random.default_rng()
        for _ in range(tokens):
            logits = self.forward(np.array([ids[-self.config['context']:]])).data[0, -1]
            if temperature <= 0:
                token = int(logits.argmax())
            else:
                z = logits / temperature
                probs = np.exp(z-z.max()); probs /= probs.sum()
                token = int(rng.choice(257, p=probs))
            if token == 256:
                break
            ids.append(token); output.append(token)
        return bytes(output).decode('utf-8', errors='replace')
