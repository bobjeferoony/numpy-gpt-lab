import tempfile
import unittest
from pathlib import Path
import numpy as np
from gpt import GPT, cross_entropy


class ModelTests(unittest.TestCase):
    def test_gradients(self):
        model = GPT(width=8, context=5, heads=2, layers=1)
        x = np.array([[1, 2, 1, 4]])
        y = np.array([[2, 1, 4, 256]])
        mask = np.array([[0., 1., 1., 1.]])
        model.zero_grad()
        cross_entropy(model.forward(x), y, mask).backward()
        rng = np.random.default_rng(7)
        # Check every parameter family, including embeddings, attention and norms.
        for name,p in model.p.items():
            direction = rng.normal(size=p.data.shape)
            direction /= np.linalg.norm(direction)
            analytic = (p.grad * direction).sum()
            original = p.data.copy()
            eps = 1e-5
            p.data[:] = original + eps*direction
            plus = float(cross_entropy(model.forward(x), y, mask).data)
            p.data[:] = original - eps*direction
            minus = float(cross_entropy(model.forward(x), y, mask).data)
            p.data[:] = original
            self.assertAlmostEqual(analytic, (plus-minus)/(2*eps), delta=2e-5, msg=name)

    def test_causality_and_save(self):
        m = GPT(width=8, context=5, heads=2, layers=1)
        a = m.forward(np.array([[1,2,3,4]])).data
        b = m.forward(np.array([[1,2,9,8]])).data
        np.testing.assert_allclose(a[:,:2], b[:,:2], atol=1e-12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'test.npz'
            m.save(path)
            np.testing.assert_array_equal(a, GPT.load(path).forward(np.array([[1,2,3,4]])).data)

    def test_learning(self):
        m = GPT(width=8, context=4, heads=2, layers=1)
        x, y, mask = np.array([[1,2,3]]), np.array([[2,3,256]]), np.ones((1,3))
        before = float(cross_entropy(m.forward(x), y, mask).data)
        for _ in range(30):
            m.zero_grad()
            loss = cross_entropy(m.forward(x), y, mask)
            loss.backward()
            for p in m.p.values():
                p.data -= .03*p.grad
        self.assertLess(float(cross_entropy(m.forward(x), y, mask).data), before*.8)


if __name__ == '__main__':
    unittest.main()
