# Filter_Response_Norm
A Pytorch implementation of the "Filter Response Normalization Layer: Eliminating Batch Dependence in the Training of Deep Neural Networks"
[paper](https://arxiv.org/abs/1911.09737)

I have tested this implementation on **PyTorch 1.1.0**.
But I guess that it might work after **Pytorch 0.4.1**.
Simplely import it and use it like official batch normalization
```
from FilterResponseNorm import *

input = torch.randn(20, 100, 32, 32)
m = FilterResponseNorm(100)
output = m(input)
```
