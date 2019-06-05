# reimplement efficientnet0~7
## test on mnist
```python
# use EfficientnetB0
Train accuracy: 0.982
Test accuracy: 0.9668
epoch: 98
loss tf.Tensor(1.4803262, shape=(), dtype=float32)
loss tf.Tensor(1.4801246, shape=(), dtype=float32)
loss tf.Tensor(1.4810265, shape=(), dtype=float32)
loss tf.Tensor(1.4791241, shape=(), dtype=float32)
loss tf.Tensor(1.4801202, shape=(), dtype=float32)
loss tf.Tensor(1.4809062, shape=(), dtype=float32)
loss tf.Tensor(1.4795817, shape=(), dtype=float32)
loss tf.Tensor(1.472376, shape=(), dtype=float32)
Train accuracy: 0.98225
Test accuracy: 0.9654
epoch: 99
loss tf.Tensor(1.4809715, shape=(), dtype=float32)
loss tf.Tensor(1.4805433, shape=(), dtype=float32)
loss tf.Tensor(1.4826591, shape=(), dtype=float32)
loss tf.Tensor(1.478586, shape=(), dtype=float32)
loss tf.Tensor(1.477656, shape=(), dtype=float32)
loss tf.Tensor(1.4782047, shape=(), dtype=float32)
loss tf.Tensor(1.4772171, shape=(), dtype=float32)
loss tf.Tensor(1.4746509, shape=(), dtype=float32)
Train accuracy: 0.9822166666666666
Test accuracy: 0.9644
```
## How to import efficientnet
```python
from model import EfficientNetB0
model = EfficientNetB0(classes=10, weights="imagenet", include_top=False)
```
### reference
1. https://arxiv.org/abs/1905.11946
2. https://github.com/qubvel/efficientnet
3. https://github.com/mingxingtan/efficientnet
4. https://github.com/mingxingtan/mnasnet
