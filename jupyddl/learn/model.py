"""A small feed-forward network, trainable without leaving the standard library.

The core of this package has no runtime dependencies and this module keeps that
promise: the reference implementation of the forward and backward passes is
plain Python lists. When NumPy happens to be importable the same equations run
batched, which is worth one to two orders of magnitude on a real training run —
so ``pip install jupyddl[learn]`` is a speed option, never a requirement.

The two paths are held together by ``tests/test_learn.py``, which asserts they
produce the same gradients to floating-point tolerance on a fixed seed. A fast
path that has quietly drifted from the reference is worse than no fast path.

The network is deliberately small. The bet a learned heuristic makes is that it
recovers more search than it costs to evaluate, and it is evaluated once per
generated state — tens of thousands of times a second. A wide network loses
that bet before it has said anything.
"""

from __future__ import annotations

import json
import math
import random
from typing import Optional

try:  # pragma: no cover - exercised by whichever path is installed
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

__all__ = ["MLP", "numpy_available"]


def numpy_available() -> bool:
    """Whether the batched fast path is usable in this environment."""
    return _np is not None


def _he_scale(fan_in: int) -> float:
    """He initialisation: keeps activation variance stable through ReLU."""
    return math.sqrt(2.0 / max(1, fan_in))


class MLP:
    """A ReLU multilayer perceptron with a non-negative scalar output.

    The output passes through softplus. A heuristic that returns a negative
    cost-to-go is not merely inaccurate, it breaks the assumptions of every
    planner consuming it — greedy search will chase the negative values and A*
    loses its admissibility argument outright. Clamping with ``max(0, x)``
    would do it too, but it has zero gradient below the threshold, so a unit
    that lands there early never recovers.
    """

    def __init__(self, sizes, seed: int = 0, weights=None, biases=None):
        self.sizes = tuple(sizes)
        if len(self.sizes) < 2:
            raise ValueError("an MLP needs at least an input and an output size")
        if self.sizes[-1] != 1:
            raise ValueError("a heuristic model has a scalar output")
        if weights is None:
            rng = random.Random(seed)
            self.weights = [
                [
                    [rng.gauss(0.0, _he_scale(fan_in)) for _ in range(fan_in)]
                    for _ in range(fan_out)
                ]
                for fan_in, fan_out in zip(self.sizes, self.sizes[1:])
            ]
            self.biases = [[0.0] * fan_out for fan_out in self.sizes[1:]]
        else:
            self.weights = [[list(row) for row in layer] for layer in weights]
            self.biases = [list(b) for b in biases]
        self._np_cache: Optional[tuple] = None

    # -- shape helpers -----------------------------------------------------
    @property
    def input_size(self) -> int:
        return self.sizes[0]

    @property
    def num_parameters(self) -> int:
        return sum(len(row) for layer in self.weights for row in layer) + sum(
            len(b) for b in self.biases
        )

    def get_flat(self) -> list:
        """Every parameter as one vector (what the derivative-free stage tunes)."""
        flat = []
        for layer in self.weights:
            for row in layer:
                flat.extend(row)
        for bias in self.biases:
            flat.extend(bias)
        return flat

    def set_flat(self, flat) -> None:
        """Inverse of :meth:`get_flat`."""
        index = 0
        for layer in self.weights:
            for row in layer:
                width = len(row)
                row[:] = flat[index : index + width]
                index += width
        for bias in self.biases:
            width = len(bias)
            bias[:] = flat[index : index + width]
            index += width
        if index != len(flat):
            raise ValueError(
                f"expected {self.num_parameters} parameters, got {len(flat)}"
            )
        self._np_cache = None

    # -- inference ---------------------------------------------------------
    def __call__(self, vector) -> float:
        """Predict for a single feature vector.

        This is the path the planner takes, once per generated state, so it
        stays a plain loop: for one sample the NumPy call overhead exceeds the
        arithmetic it would save.
        """
        activation = vector
        last = len(self.weights) - 1
        for index, (layer, bias) in enumerate(zip(self.weights, self.biases)):
            out = []
            for row, b in zip(layer, bias):
                total = b
                for w, a in zip(row, activation):
                    total += w * a
                out.append(total if index == last else (total if total > 0.0 else 0.0))
            activation = out
        return _softplus(activation[0])

    # -- training ----------------------------------------------------------
    def forward_batch(self, batch):
        """Predictions plus the activations the backward pass needs.

        Returns ``(outputs, cache)``. ``outputs`` are post-softplus.
        """
        if _np is not None:
            return self._forward_numpy(batch)
        return self._forward_python(batch)

    def backward_batch(self, cache, d_outputs):
        """Parameter gradients, given d(loss)/d(output) for each sample.

        The loss lives in the caller. Keeping it there is what lets the same
        network be trained against a regression target and against a ranking
        objective without the model knowing the difference.
        """
        if _np is not None:
            return self._backward_numpy(cache, d_outputs)
        return self._backward_python(cache, d_outputs)

    # -- reference implementation -----------------------------------------
    def _forward_python(self, batch):
        activations = [batch]
        pre_activations = []
        current = batch
        last = len(self.weights) - 1
        for index, (layer, bias) in enumerate(zip(self.weights, self.biases)):
            pre = []
            for sample in current:
                row_out = []
                for row, b in zip(layer, bias):
                    total = b
                    for w, a in zip(row, sample):
                        total += w * a
                    row_out.append(total)
                pre.append(row_out)
            pre_activations.append(pre)
            if index == last:
                current = pre
            else:
                current = [[v if v > 0.0 else 0.0 for v in row] for row in pre]
                activations.append(current)
        raw = [row[0] for row in current]
        outputs = [_softplus(v) for v in raw]
        return outputs, (activations, pre_activations, raw)

    def _backward_python(self, cache, d_outputs):
        activations, pre_activations, raw = cache
        # d(softplus)/dx = sigmoid(x)
        delta = [[d * _sigmoid(r)] for d, r in zip(d_outputs, raw)]

        grad_w = [[[0.0] * len(row) for row in layer] for layer in self.weights]
        grad_b = [[0.0] * len(bias) for bias in self.biases]

        for index in range(len(self.weights) - 1, -1, -1):
            layer = self.weights[index]
            inputs = activations[index]
            gw = grad_w[index]
            gb = grad_b[index]
            for sample_delta, sample_input in zip(delta, inputs):
                for j, d in enumerate(sample_delta):
                    if d == 0.0:
                        continue
                    gb[j] += d
                    row = gw[j]
                    for k, a in enumerate(sample_input):
                        row[k] += d * a
            if index == 0:
                break
            pre = pre_activations[index - 1]
            new_delta = []
            for sample_delta, sample_pre in zip(delta, pre):
                back = [0.0] * len(sample_pre)
                for j, d in enumerate(sample_delta):
                    if d == 0.0:
                        continue
                    row = layer[j]
                    for k in range(len(back)):
                        back[k] += d * row[k]
                new_delta.append(
                    [b if p > 0.0 else 0.0 for b, p in zip(back, sample_pre)]
                )
            delta = new_delta
        return grad_w, grad_b

    # -- batched fast path -------------------------------------------------
    def _numpy_params(self):
        if self._np_cache is None:
            self._np_cache = (
                [_np.array(layer, dtype=_np.float64) for layer in self.weights],
                [_np.array(bias, dtype=_np.float64) for bias in self.biases],
            )
        return self._np_cache

    def _forward_numpy(self, batch):
        weights, biases = self._numpy_params()
        current = _np.array(batch, dtype=_np.float64)
        activations = [current]
        pre_activations = []
        last = len(weights) - 1
        for index, (layer, bias) in enumerate(zip(weights, biases)):
            pre = current @ layer.T + bias
            pre_activations.append(pre)
            if index == last:
                current = pre
            else:
                current = _np.maximum(pre, 0.0)
                activations.append(current)
        raw = current[:, 0]
        outputs = _np.logaddexp(0.0, raw)
        return outputs.tolist(), (activations, pre_activations, raw)

    def _backward_numpy(self, cache, d_outputs):
        weights, _ = self._numpy_params()
        activations, pre_activations, raw = cache
        d_out = _np.array(d_outputs, dtype=_np.float64)
        delta = (d_out / (1.0 + _np.exp(-raw)))[:, None]

        grad_w = [None] * len(weights)
        grad_b = [None] * len(weights)
        for index in range(len(weights) - 1, -1, -1):
            inputs = activations[index]
            grad_w[index] = (delta.T @ inputs).tolist()
            grad_b[index] = delta.sum(axis=0).tolist()
            if index == 0:
                break
            back = delta @ weights[index]
            delta = back * (pre_activations[index - 1] > 0.0)
        return grad_w, grad_b

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "sizes": list(self.sizes),
            "weights": [[list(row) for row in layer] for layer in self.weights],
            "biases": [list(b) for b in self.biases],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MLP":
        return cls(data["sizes"], weights=data["weights"], biases=data["biases"])

    def copy(self) -> "MLP":
        return MLP(self.sizes, weights=self.weights, biases=self.biases)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        shape = "x".join(str(s) for s in self.sizes)
        return f"MLP({shape}, {self.num_parameters} parameters)"


class Adam:
    """Adam, with the usual defaults, over the nested weight/bias structure."""

    def __init__(self, model: MLP, lr: float = 0.01, betas=(0.9, 0.999), eps=1e-8):
        self.model = model
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0
        self.m_w = [[[0.0] * len(row) for row in layer] for layer in model.weights]
        self.v_w = [[[0.0] * len(row) for row in layer] for layer in model.weights]
        self.m_b = [[0.0] * len(b) for b in model.biases]
        self.v_b = [[0.0] * len(b) for b in model.biases]

    def step(self, grad_w, grad_b) -> None:
        self.step_count += 1
        bias1 = 1.0 - self.beta1**self.step_count
        bias2 = 1.0 - self.beta2**self.step_count
        for layer_index, layer in enumerate(self.model.weights):
            gw = grad_w[layer_index]
            mw = self.m_w[layer_index]
            vw = self.v_w[layer_index]
            for j, row in enumerate(layer):
                grad_row = gw[j]
                m_row = mw[j]
                v_row = vw[j]
                for k in range(len(row)):
                    g = grad_row[k]
                    m_row[k] = self.beta1 * m_row[k] + (1 - self.beta1) * g
                    v_row[k] = self.beta2 * v_row[k] + (1 - self.beta2) * g * g
                    row[k] -= (
                        self.lr
                        * (m_row[k] / bias1)
                        / (math.sqrt(v_row[k] / bias2) + self.eps)
                    )
        for layer_index, bias in enumerate(self.model.biases):
            gb = grad_b[layer_index]
            mb = self.m_b[layer_index]
            vb = self.v_b[layer_index]
            for j in range(len(bias)):
                g = gb[j]
                mb[j] = self.beta1 * mb[j] + (1 - self.beta1) * g
                vb[j] = self.beta2 * vb[j] + (1 - self.beta2) * g * g
                bias[j] -= (
                    self.lr * (mb[j] / bias1) / (math.sqrt(vb[j] / bias2) + self.eps)
                )
        self.model._np_cache = None


def _softplus(x: float) -> float:
    # log(1 + e^x), written so a large x does not overflow.
    if x > 30.0:
        return x
    if x < -30.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def save_model(model: MLP, path: str) -> None:  # pragma: no cover - thin
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(model.to_dict(), handle)


def load_model(path: str) -> MLP:  # pragma: no cover - thin
    with open(path, encoding="utf-8") as handle:
        return MLP.from_dict(json.load(handle))
